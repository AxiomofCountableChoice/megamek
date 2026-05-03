import socket
import struct
import msgpack
import time
import torch
import numpy as np

from torch_geometric.data import HeteroData

class MegaMekEnvironment:
    """
    A Gym-like environment wrapper that manages headless MegaMek via TCP.
    It synchronously maintains a connection and parses MessagePack state payloads
    into PyTorch Geometric HeteroData objects.
    """
    def __init__(self, host='localhost', port=12346, device=None):
        self.host = host
        self.port = port
        self.sock = None
        self._connected = False
        self.device = device
        self._max_attempts = 50
        # Static Topology Cache
        # Populated once per game match to prevent redundant IPC overhead
        self.static_hex_features = None 
        self.static_hex_adjacency_edges = None
        
        # Connect immediately
        self.connect()

    def connect(self):
        if self._connected:
            return

        # If not already connected proceed to connect 
        print(f"Connecting to MegaMek RLServer at {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        attempt = 0
        while attempt < self._max_attempts:
            try:
                self.sock.connect((self.host, self.port))
                self._connected = True
                print("Connected successfully!")
                break
            except ConnectionRefusedError:
                print("Waiting for MegaMek server to start...")
                time.sleep(2)
            attempt += 1
        if attempt == self._max_attempts:
            raise ConnectionError("Failed to connect to MegaMek server.")
                
    def reset(self):
        """
        Wait for the next match state payload and return the initial state.
        If self.device is provided (e.g. 'cuda:0' or a torch.device object), moves the parsed HeteroData graph and its tensors to this device.
        """
        if not self._connected:
            self.connect()
            
        print("Awaiting initial state payload...")
        payload = self._receive_payload()
        if payload is None:
            raise ConnectionError("Server disconnected during reset.")
            
        if payload.get("context") == "TOPOLOGY":
            self.board_width = payload.get('width', 0)
            self.feature_dims = payload.get('feature_dims', {"hex": 14, "unit": 37, "weapon": 10})
            print(f"Received TOPOLOGY payload. Parsing Board shape ({payload.get('width')}x{payload.get('height')})...")
            nodes = payload.get("hex_nodes", [])
            edges = payload.get("hex_edges", [])
            
            if nodes:
                t = torch.tensor(nodes, dtype=torch.float32)
                self.static_hex_features = t
            else:
                self.static_hex_features = torch.empty((0, 5), dtype=torch.float32)
                
            if edges:
                edge_array = np.array(edges, dtype=np.int64)
                self.static_hex_adjacency_edges = edge_array
            else:
                self.static_hex_adjacency_edges = None
                    
            print("Topology cached. Awaiting actual initial state...")
            payload = self._receive_payload()
            if payload is None:
                 raise ConnectionError("Server disconnected while waiting for STATE.")
                    
        data_graph, mask = self._parse_to_heterodata(payload)
        if self.device is not None:
            data_graph = data_graph.to(self.device)
            self.static_hex_features = self.static_hex_features.to(self.device)
            if self.static_hex_adjacency_edges is not None:
                self.static_hex_adjacency_edges = self.static_hex_adjacency_edges.to(self.device)
            
        return data_graph, mask

    def step(self, action_dict):
        """
        Submit an action dict (e.g. {"selected_path_index": 0}) and await the next state.
        Dynamically shifts the next payload onto self.device natively if set.
        """
        res_bytes = msgpack.packb(action_dict, use_bin_type=True)
        self.sock.sendall(struct.pack('>I', len(res_bytes)))
        self.sock.sendall(res_bytes)
        
        # Await next state
        payload = self._receive_payload()
        if payload is None:
            # Done True
            return None, None, True 
            
        state, mask = self._parse_to_heterodata(payload)
        if (self.device is not None) and (state is not None):
            state = state.to(self.device)
            
        # Using dummy reward/done for now
        return state, mask, False

    def _receive_payload(self):
        # Read 4-byte length prefix
        length_buf = self.sock.recv(4)
        if not length_buf:
            return None
        
        payload_len = struct.unpack('>I', length_buf)[0]
        
        # Read exact payload length
        data = b''
        while len(data) < payload_len:
            packet = self.sock.recv(payload_len - len(data))
            if not packet:
                return None
            data += packet
            
        return msgpack.unpackb(data, raw=False)

    def _parse_to_heterodata(self, payload):
        """
        Converts the dynamic dictionary state into a PyTorch Geometric Graph.
        Utilizes Delta Caching for static terrain geometries.
        """
        context = payload.get("context", "UNKNOWN")
        raw_state = payload.get("state", {})
        mask_raw = payload.get("mask", {})
        mask = mask_raw if isinstance(mask_raw, dict) else {}
        
        data = HeteroData()
        data.context = context
        
        if context == "MOVEMENT_BC":
            phase_type = 0.0
        elif context == "WEAPON_BC":
            phase_type = 1.0
        elif context == "PHYSICAL_BC":
            phase_type = 2.0
        else:
            phase_type = -1.0
        
        # 1. Global Phase Features ($p \rightarrow z_{context}$)
        phase_str = raw_state.get('phase_main', "UNKNOWN")
        turn = raw_state.get('turn_number', 0)
        
        # Dummy embedding for phase
        data.global_context = torch.tensor([turn, len(phase_str)], dtype=torch.float32)
        
        # 2. Dynamic Entity Nodes ($V_U$)
        raw_entities = raw_state.get("entities", [])
        unit_dim = getattr(self, 'feature_dims', {}).get("unit", 37)
        if raw_entities:
            t = torch.tensor(raw_entities, dtype=torch.float32)
            pad = torch.zeros((t.size(0), unit_dim - t.size(1)), dtype=torch.float32)
            data['unit'].x = torch.cat([t, pad], dim=1)
        else:
            data['unit'].x = torch.empty((0, unit_dim), dtype=torch.float32)
        
        # 3. Static Hex/Topology Cache ($V_H$ and $E_{adj}$)
        if self.static_hex_features is not None:
            data['hex'].x = self.static_hex_features
        else:
            hex_dim = getattr(self, 'feature_dims', {}).get("hex", 14)
            data['hex'].x = torch.empty((0, hex_dim), dtype=torch.float32)
        
        if self.static_hex_adjacency_edges is not None:
            edge_arr = self.static_hex_adjacency_edges
            for dir_idx in range(6):
                dir_mask = edge_arr[:, 2] == dir_idx
                data['hex', f'hexAdj_{dir_idx}', 'hex'].edge_index = edge_arr[dir_mask, :2].T.detach().clone().to(torch.long)
        else:
            for dir_idx in range(6):
                data['hex', f'hexAdj_{dir_idx}', 'hex'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 4. Weapon Nodes ($V_W$)
        raw_weapons = raw_state.get("weapons", [])
        weapon_dim = getattr(self, 'feature_dims', {}).get("weapon", 10)
        if raw_weapons:
            w_tensor = torch.tensor(raw_weapons, dtype=torch.float32)
            # MegaMek uses Integer.MIN_VALUE for WeaponType.DAMAGE_NA, RANGE_NA, etc.
            # We must mask these out to prevent exploding gradients.
            w_tensor[w_tensor < -1000.0] = 0.0
            data['weapon'].x = w_tensor
        else:
            data['weapon'].x = torch.empty((0, weapon_dim), dtype=torch.float32)
            
        equips = raw_state.get("equips_edges", [])
        if equips:
            e_arr = np.array(equips, dtype=np.int64).T
            data['weapon', 'equips', 'unit'].edge_index = torch.tensor(e_arr, dtype=torch.long)
        else:
            data['weapon', 'equips', 'unit'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 4. Ephemeral Edges ($E_{occ}$)
        width = getattr(self, "board_width", 0)
        mech_indices = []
        hex_indices = []
        
        for i, ent in enumerate(raw_entities):
            ex, ey = ent[1], ent[2]
            if ex >= 0 and ey >= 0 and width > 0:
                hex_idx = int(ey * width + ex)
                mech_indices.append(i)
                hex_indices.append(hex_idx)
                
        if mech_indices:
            data['unit', 'occupies', 'hex'].edge_index = torch.tensor([mech_indices, hex_indices], dtype=torch.long)
        else:
            data['unit', 'occupies', 'hex'].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 4.5 Ephemeral Threat and LOS Edges
        def add_ephemeral_edges(key, src, dst, out_type):
            edges = raw_state.get(key, [])
            if edges:
                e_arr = np.array(edges, dtype=np.int64).T
                data[src, out_type, dst].edge_index = torch.tensor(e_arr, dtype=torch.long)
            else:
                data[src, out_type, dst].edge_index = torch.empty((2, 0), dtype=torch.long)
                
        add_ephemeral_edges("los_target_edges", "unit", "unit", "LOSTarget")
        add_ephemeral_edges("los_threat_edges", "unit", "hex", "LOSThreat")
        add_ephemeral_edges("partial_cover_edges", "unit", "hex", "partialCover")
        add_ephemeral_edges("movement_threat_edges", "unit", "hex", "movementThreat")
        for tmm in range(5):
            add_ephemeral_edges(f"move_type_tmm_{tmm}_edges", "unit", "hex", f"moveTypeTMM_{tmm}")

        # 5. Dynamic Action Nodes ($V_A$) for Autoregressive Trees
        valid_paths = mask.get("valid_paths", [])
        active_entity_idx = mask.get("active_entity_index", -1)
        
        if valid_paths and context == "MOVEMENT_BC":
            # Extract target action logically if provided by the Offline pipeline
            target_action = payload.get("target_action", {})
            true_selected_idx = target_action.get("selected_path_index", -1)
            
            # Mathematical Mapping:
            # Step k=0: Choose Target Hex. C_0 = {unique dest_index values}
            # Step k=1: Choose Path Mode. C_1 = {(facing, mp, jump) | dest_index == chosen_dest_index}
            
            tier0_hexes = {} # Map dest_index -> list of original valid_path indices
            for i, p in enumerate(valid_paths):
                d_idx = p.get("dest_index", -1)
                if d_idx not in tier0_hexes:
                    tier0_hexes[d_idx] = []
                tier0_hexes[d_idx].append(i)
                
            unique_hex_list = list(tier0_hexes.keys())
            
            # Formulate Tier 0 candidates
            action_features = []
            action_target_hex_idx = []
            action_target_unit_idx = []
            action_target_weapon_idx = []
            action_source_unit_idx = []
            step_indices = []
            
            # Ground-truth targets
            true_a0_idx = -1
            true_a1_idx = -1
            
            true_dest_index = -1
            if true_selected_idx != -1 and true_selected_idx < len(valid_paths):
                true_dest_index = valid_paths[true_selected_idx].get("dest_index", -1)
            
            # --- TIER 0 ---
            for c_idx, dest_idx in enumerate(unique_hex_list):
                action_idx = len(action_features)
                step_indices.append(0) # k=0
                
                # Abstract representation for "Hex Selection". Feature values are 0 since the spatial target edge provides context.
                action_features.append([0.0, 0.0, 0.0, 0.0, 0.0, phase_type])
                
                action_target_hex_idx.append(dest_idx if dest_idx != -1 else -1)
                action_target_unit_idx.append(-1)
                action_target_weapon_idx.append(-1)
                action_source_unit_idx.append(active_entity_idx if active_entity_idx != -1 else -1)
                    
                if dest_idx == true_dest_index:
                    true_a0_idx = action_idx
                    
            # --- TIER 1 ---
            # Conditioned entirely on the True Prefix A_{<1} (Teacher Forcing assumption)
            if true_dest_index != -1:
                valid_tier1_paths = tier0_hexes[true_dest_index]
                for raw_path_idx in valid_tier1_paths:
                    p_dict = valid_paths[raw_path_idx]
                    
                    action_idx = len(action_features)
                    step_indices.append(1) # k=1
                    
                    facing = float(p_dict.get("dest_facing", 0))
                    mp_used = float(p_dict.get("mp_used", 0))
                    is_jump = 1.0 if p_dict.get("is_jump", False) else 0.0
                    action_features.append([mp_used, facing, is_jump, 0.0, 1.0, phase_type])
                    
                    # Target is still the hex to ground it spatially
                    action_target_hex_idx.append(true_dest_index)
                    action_target_unit_idx.append(-1)
                    action_target_weapon_idx.append(-1)
                    action_source_unit_idx.append(active_entity_idx if active_entity_idx != -1 else -1)
                        
                    if raw_path_idx == true_selected_idx:
                        true_a1_idx = action_idx
            
            # Populate PyG object
            if action_features:
                data['action'].x = torch.tensor(action_features, dtype=torch.float32)
                data['action'].step_idx = torch.tensor(step_indices, dtype=torch.long)
                data['action'].target_hex_idx = torch.tensor(action_target_hex_idx, dtype=torch.long)
                data['action'].target_unit_idx = torch.tensor(action_target_unit_idx, dtype=torch.long)
                data['action'].target_weapon_idx = torch.tensor(action_target_weapon_idx, dtype=torch.long)
                data['action'].source_unit_idx = torch.tensor(action_source_unit_idx, dtype=torch.long)
                
                if true_a0_idx != -1 and true_a1_idx != -1:
                    data.y_sequence = torch.tensor([true_a0_idx, true_a1_idx], dtype=torch.long)
            else:
                data['action'].x = torch.empty((0, 6), dtype=torch.float32)
                data['action'].step_idx = torch.empty((0,), dtype=torch.long)
                data['action'].target_hex_idx = torch.empty((0,), dtype=torch.long)
                data['action'].target_unit_idx = torch.empty((0,), dtype=torch.long)
                data['action'].target_weapon_idx = torch.empty((0,), dtype=torch.long)
                data['action'].source_unit_idx = torch.empty((0,), dtype=torch.long)
                
        elif "valid_twists" in mask and context == "WEAPON_BC":
            valid_twists = mask.get("valid_twists", [])
            target_action = payload.get("target_action", {})
            chosen_attacks = target_action.get("attacks", [])
            chosen_twist = target_action.get("torso_twist", 0) # Default to 0
            
            action_features = []
            action_target_hex_idx = []
            action_target_unit_idx = []
            action_target_weapon_idx = []
            action_source_unit_idx = []
            action_type_flags = [] # 0: Twist, 1: Target, 2: Weapon, 3: END
            action_twist_context = [] # Store which twist this node belongs to
            
            # Action nodes:
            # 1. Torso Twist Nodes (always 3: -1, 0, 1)
            # 2. Target nodes (tied to a twist)
            # 3. Weapon nodes (tied to a target and twist)
            # 4. END node
            
            twist_values = [-1, 0, 1]
            twist_node_indices = {}
            for tv in twist_values:
                twist_node_indices[tv] = len(action_features)
                action_features.append([tv, 0.0, 0.0, 0.0, 0.0, phase_type]) # Torso Twist Feature
                action_target_hex_idx.append(-1)
                action_target_unit_idx.append(-1)
                action_target_weapon_idx.append(-1)
                action_source_unit_idx.append(active_entity_idx if active_entity_idx != -1 else -1)
                action_type_flags.append(0)
                action_twist_context.append(tv)
            
            for twist_mask in valid_twists:
                tv = twist_mask.get("twist", 0)
                valid_targets = twist_mask.get("valid_targets", [])
                
                for tm in valid_targets:
                    target_entity_index = tm.get("target_entity_index", -1)
                    
                    # Append Target Node
                    action_features.append([1.0, 0.0, 0.0, 0.0, 1.0, phase_type]) # Target Feature
                    action_target_hex_idx.append(-1)
                    action_target_unit_idx.append(target_entity_index)
                    action_target_weapon_idx.append(-1)
                    action_source_unit_idx.append(active_entity_idx if active_entity_idx != -1 else -1)
                    action_type_flags.append(1)
                    action_twist_context.append(tv)
                    
                    valid_weapons = tm.get("valid_weapons", [])
                    for wm in valid_weapons:
                        weapon_id = wm.get("weapon_id", -1)
                        to_hit = float(wm.get("to_hit", 0.0))
                        
                        # Append Weapon Node
                        action_features.append([0.0, 1.0, to_hit, 0.0, 2.0, phase_type]) # Weapon Feature
                        action_target_hex_idx.append(-1)
                        action_target_unit_idx.append(-1)
                        action_target_weapon_idx.append(weapon_id)
                        action_source_unit_idx.append(active_entity_idx if active_entity_idx != -1 else -1)
                        action_type_flags.append(2)
                        action_twist_context.append(tv)
                    
            # Append END node
            end_node_idx = len(action_features)
            action_features.append([0.0, 0.0, 1.0, 0.0, 3.0, phase_type]) # END Feature
            action_target_hex_idx.append(-1)
            action_target_unit_idx.append(-1)
            action_target_weapon_idx.append(-1)
            action_source_unit_idx.append(active_entity_idx if active_entity_idx != -1 else -1)
            action_type_flags.append(3)
            action_twist_context.append(-99) # End applies everywhere
            
            # Map chosen attacks to node sequence
            true_sequence_indices = []
            if target_action: # Only build sequence if we have ground truth
                # Step 0: Torso Twist
                true_sequence_indices.append(twist_node_indices.get(chosen_twist, twist_node_indices[0]))
                
                if chosen_attacks:
                    # Group by target
                    target_to_weapons = {}
                    for att in chosen_attacks:
                        t_idx = att.get("target_entity_index", -1)
                        w_id = att.get("weapon_id", -1)
                        if t_idx not in target_to_weapons:
                            target_to_weapons[t_idx] = []
                        target_to_weapons[t_idx].append(w_id)
                    
                    # Reconstruct sequence Target -> W1 -> W2 -> Target -> W3 -> END
                    for t_idx, w_ids in target_to_weapons.items():
                        # Find Target Node Index in the correct twist context
                        try:
                            t_node_idx = next(i for i, (type_flag, tgt_idx, ctx) in enumerate(zip(action_type_flags, action_target_unit_idx, action_twist_context)) 
                                            if type_flag == 1 and tgt_idx == t_idx and ctx == chosen_twist)
                            true_sequence_indices.append(t_node_idx)
                            
                            # Find Weapon Node Indices in the correct twist context
                            for w_id in w_ids:
                                w_node_idx = next(i for i, (type_flag, tgt_w_idx, ctx) in enumerate(zip(action_type_flags, action_target_weapon_idx, action_twist_context)) 
                                                if type_flag == 2 and tgt_w_idx == w_id and ctx == chosen_twist)
                                true_sequence_indices.append(w_node_idx)
                        except StopIteration:
                            continue
                
                # END node is always the last action
                true_sequence_indices.append(end_node_idx)
            
            if action_features:
                data['action'].x = torch.tensor(action_features, dtype=torch.float32)
                # Step indices don't make sense statically here because it's variable length
                data['action'].step_idx = torch.zeros((len(action_features),), dtype=torch.long)
                data['action'].target_hex_idx = torch.tensor(action_target_hex_idx, dtype=torch.long)
                data['action'].target_unit_idx = torch.tensor(action_target_unit_idx, dtype=torch.long)
                data['action'].target_weapon_idx = torch.tensor(action_target_weapon_idx, dtype=torch.long)
                data['action'].source_unit_idx = torch.tensor(action_source_unit_idx, dtype=torch.long)
                
                if true_sequence_indices and target_action: # Only append y_sequence if we have a teacher action
                    data.y_sequence = torch.tensor(true_sequence_indices, dtype=torch.long)
            else:
                data['action'].x = torch.empty((0, 6), dtype=torch.float32)
                data['action'].step_idx = torch.empty((0,), dtype=torch.long)
                data['action'].target_hex_idx = torch.empty((0,), dtype=torch.long)
                data['action'].target_unit_idx = torch.empty((0,), dtype=torch.long)
                data['action'].target_weapon_idx = torch.empty((0,), dtype=torch.long)
                data['action'].source_unit_idx = torch.empty((0,), dtype=torch.long)

        elif "valid_targets" in mask and context == "PHYSICAL_BC":
            valid_targets = mask.get("valid_targets", [])
            target_action = payload.get("target_action", {})
            chosen_attacks = target_action.get("attacks", [])
            
            action_features = []
            action_target_hex_idx = []
            action_target_unit_idx = []
            action_target_weapon_idx = []
            action_source_unit_idx = []
            action_type_flags = [] # 1: Target, 2: Physical Attack, 3: END
            action_type_context = [] # Store physical action type for lookup
            
            # Action nodes: Target -> Physical Action -> END
            
            for tm in valid_targets:
                target_entity_index = tm.get("target_entity_index", -1)
                
                # Append Target Node
                action_features.append([1.0, 0.0, 0.0, 0.0, 1.0, phase_type]) # Target Feature
                action_target_hex_idx.append(-1)
                action_target_unit_idx.append(target_entity_index)
                action_target_weapon_idx.append(-1)
                action_source_unit_idx.append(active_entity_idx if active_entity_idx != -1 else -1)
                action_type_flags.append(1)
                action_type_context.append(-1)
                
                valid_attacks = tm.get("valid_attacks", [])
                for am in valid_attacks:
                    action_type = am.get("action_type", -1)
                    to_hit = float(am.get("to_hit", 0.0))
                    
                    # Append Physical Action Node
                    action_features.append([0.0, 1.0, to_hit, float(action_type), 2.0, phase_type]) # Physical Action Feature encodes action_type
                    action_target_hex_idx.append(-1)
                    action_target_unit_idx.append(-1)
                    action_target_weapon_idx.append(-1)
                    action_source_unit_idx.append(active_entity_idx if active_entity_idx != -1 else -1) 
                    action_type_flags.append(2)
                    action_type_context.append(action_type)
                    
            # Append END node
            end_node_idx = len(action_features)
            action_features.append([0.0, 0.0, 1.0, 0.0, 3.0, phase_type]) # END Feature
            action_target_hex_idx.append(-1)
            action_target_unit_idx.append(-1)
            action_target_weapon_idx.append(-1)
            action_source_unit_idx.append(active_entity_idx if active_entity_idx != -1 else -1)
            action_type_flags.append(3)
            action_type_context.append(-1)
            
            # Map chosen attacks to node sequence
            true_sequence_indices = []
            if target_action: # Only build sequence if we have ground truth
                if chosen_attacks:
                    # Physical phase typically only allows 1 attack against 1 target, but we'll loop just in case
                    # We will output Target -> Action -> Target -> Action -> END
                    for att in chosen_attacks:
                        t_idx = att.get("target_entity_index", -1)
                        p_action = att.get("physical_action_type", -1)
                        
                        try:
                            # Find Target Node
                            t_node_idx = next(i for i, (type_flag, tgt_idx) in enumerate(zip(action_type_flags, action_target_unit_idx)) 
                                            if type_flag == 1 and tgt_idx == t_idx)
                            true_sequence_indices.append(t_node_idx)
                            
                            # Find Physical Action Node (using action_type_context)
                            a_node_idx_refined = next(i for i in range(t_node_idx+1, len(action_type_flags))
                                            if action_type_flags[i] == 2 and action_type_context[i] == p_action)
                            
                            true_sequence_indices.append(a_node_idx_refined)
                        except StopIteration:
                            continue
                            
                # END node is always the last action
                true_sequence_indices.append(end_node_idx)
            
            if action_features:
                data['action'].x = torch.tensor(action_features, dtype=torch.float32)
                data['action'].step_idx = torch.zeros((len(action_features),), dtype=torch.long)
                data['action'].target_hex_idx = torch.tensor(action_target_hex_idx, dtype=torch.long)
                data['action'].target_unit_idx = torch.tensor(action_target_unit_idx, dtype=torch.long)
                data['action'].target_weapon_idx = torch.tensor(action_target_weapon_idx, dtype=torch.long)
                data['action'].source_unit_idx = torch.tensor(action_source_unit_idx, dtype=torch.long)
                
                if true_sequence_indices and target_action: # Only append y_sequence if we have a teacher action
                    data.y_sequence = torch.tensor(true_sequence_indices, dtype=torch.long)
            else:
                data['action'].x = torch.empty((0, 6), dtype=torch.float32)
                data['action'].step_idx = torch.empty((0,), dtype=torch.long)
                data['action'].target_hex_idx = torch.empty((0,), dtype=torch.long)
                data['action'].target_unit_idx = torch.empty((0,), dtype=torch.long)
                data['action'].target_weapon_idx = torch.empty((0,), dtype=torch.long)
                data['action'].source_unit_idx = torch.empty((0,), dtype=torch.long)

        else:
            data['action'].x = torch.empty((0, 6), dtype=torch.float32)
            data['action'].step_idx = torch.empty((0,), dtype=torch.long)
            data['action'].target_hex_idx = torch.empty((0,), dtype=torch.long)
            data['action'].target_unit_idx = torch.empty((0,), dtype=torch.long)
            data['action'].target_weapon_idx = torch.empty((0,), dtype=torch.long)
            data['action'].source_unit_idx = torch.empty((0,), dtype=torch.long)

        return data, mask

    def close(self):
        if self.sock:
            self.sock.close()
            self._connected = False
            print("Environment socket closed.")
