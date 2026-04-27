import socket
import struct
import msgpack
import time
import torch
import numpy as np

# We conditionally import torch_geometric so it doesn't crash if not installed yet.
try:
    from torch_geometric.data import HeteroData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("Warning: torch_geometric not found. Please install PyTorch Geometric for full functionality.")

class MegaMekEnvironment:
    """
    A Gym-like environment wrapper that manages headless MegaMek via TCP.
    It synchronously maintains a connection and parses MessagePack state payloads
    into PyTorch Geometric HeteroData objects.
    """
    def __init__(self, host='localhost', port=12346):
        self.host = host
        self.port = port
        self.sock = None
        self._connected = False
        
        # Static Topology Cache
        # Populated once per game match to prevent redundant IPC overhead
        self.static_hex_features = None 
        self.static_hex_adjacency_edges = None

    def connect(self):
        print(f"Connecting to MegaMek RLServer at {self.host}:{self.port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        while True:
            try:
                self.sock.connect((self.host, self.port))
                self._connected = True
                print("Connected successfully!")
                break
            except ConnectionRefusedError:
                print("Waiting for MegaMek server to start...")
                time.sleep(2)
                
    def reset(self, device=None):
        """
        Wait for the next match state payload and return the initial state.
        If device is provided (e.g. 'cuda:0' or a torch.device object), moves the parsed HeteroData graph and its tensors to this device.
        """
        if not self._connected:
            self.connect()
            
        print("Awaiting initial state payload...")
        payload = self._receive_payload()
        if payload is None:
            raise ConnectionError("Server disconnected during reset.")
            
        if payload.get("context") == "TOPOLOGY":
            print(f"Received TOPOLOGY payload. Parsing Board shape ({payload.get('width')}x{payload.get('height')})...")
            if HAS_PYG:
                nodes = payload.get("hex_nodes", [])
                edges = payload.get("hex_edges", [])
                
                if nodes:
                    t = torch.tensor(nodes, dtype=torch.float32)
                    self.static_hex_features = t
                else:
                    self.static_hex_features = torch.empty((0, 5), dtype=torch.float32)
                    
                if edges:
                    import numpy as np
                    edge_array = np.array(edges, dtype=np.int64)
                    self.static_hex_adjacency_edges = edge_array
                else:
                    self.static_hex_adjacency_edges = None
                    
            print("Topology cached. Awaiting actual initial state...")
            payload = self._receive_payload()
            if payload is None:
                 raise ConnectionError("Server disconnected while waiting for STATE.")
                    
        data_graph, mask = self._parse_to_heterodata(payload)
        if device is not None and HAS_PYG:
            data_graph = data_graph.to(device)
            self.static_hex_features = self.static_hex_features.to(device)
            self.static_hex_adjacency_edges = self.static_hex_adjacency_edges.to(device)
            
        return data_graph, mask

    def step(self, action_dict, device=None):
        """
        Submit an action dict (e.g. {"selected_path_index": 0}) and await the next state.
        Optional device parameter dynamically shifts the next payload onto GPU natively.
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
        if device is not None and HAS_PYG and state is not None:
            state = state.to(device)
            
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
        mask = payload.get("mask", {})
        
        if not HAS_PYG:
            # Fallback for compilation testing before library installs
            return {"raw_state": raw_state, "context": context}, mask
            
        data = HeteroData()
        
        # 1. Global Phase Features ($p \rightarrow z_{context}$)
        phase_str = raw_state.get('phase_main', "UNKNOWN")
        turn = raw_state.get('turn_number', 0)
        
        # Dummy embedding for phase
        data.global_context = torch.tensor([turn, len(phase_str)], dtype=torch.float32)
        
        # 2. Dynamic Entity Nodes ($V_U$)
        raw_entities = raw_state.get("entities", [])
        if raw_entities:
            t = torch.tensor(raw_entities, dtype=torch.float32)
            pad = torch.zeros((t.size(0), 36 - t.size(1)), dtype=torch.float32)
            data['unit'].x = torch.cat([t, pad], dim=1)
        else:
            # Padded to 36 covariates based on ARCHITECTURE.md specifications
            data['unit'].x = torch.empty((0, 36), dtype=torch.float32)
        
        # 3. Static Hex/Topology Cache ($V_H$ and $E_{adj}$)
        if self.static_hex_features is not None:
            t = self.static_hex_features
            if t.size(0) > 0:
                pad = torch.zeros((t.size(0), 14 - t.size(1)), dtype=torch.float32)
                data['hex'].x = torch.cat([t, pad], dim=1)
            else:
                data['hex'].x = torch.empty((0, 14), dtype=torch.float32)
        
        if self.static_hex_adjacency_edges is not None:
            edge_arr = self.static_hex_adjacency_edges
            if len(edge_arr.shape) > 1 and edge_arr.shape[1] == 3:
                for dir_idx in range(6):
                    mask = edge_arr[:, 2] == dir_idx
                    if mask.any():
                        data['hex', f'hexAdj_{dir_idx}', 'hex'].edge_index = torch.tensor(edge_arr[mask, :2].T, dtype=torch.long)
                    else:
                        data['hex', f'hexAdj_{dir_idx}', 'hex'].edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                # Legacy, fallback to undirected edge mapping for index 0
                edge_t = torch.tensor(edge_arr.T, dtype=torch.long) if len(edge_arr.shape) > 1 else torch.empty((2,0), dtype=torch.long)
                data['hex', 'hexAdj_0', 'hex'].edge_index = edge_t
                for dir_idx in range(1, 6):
                    data['hex', f'hexAdj_{dir_idx}', 'hex'].edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            for dir_idx in range(6):
                data['hex', f'hexAdj_{dir_idx}', 'hex'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 3.5 Dummy Weapon Nodes ($V_W$)
        data['weapon'].x = torch.empty((0, 10), dtype=torch.float32)
        data['weapon', 'equips', 'unit'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 4. Ephemeral Edges ($E_{occ}$)
        width = raw_state.get("global_state", {}).get("board_width", 0)
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
                action_features.append([0.0, 0.0, 0.0])
                
                action_target_hex_idx.append(dest_idx if dest_idx != -1 else -1)
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
                    action_features.append([mp_used, facing, is_jump])
                    
                    # Target is still the hex to ground it spatially
                    action_target_hex_idx.append(true_dest_index)
                    action_source_unit_idx.append(active_entity_idx if active_entity_idx != -1 else -1)
                        
                    if raw_path_idx == true_selected_idx:
                        true_a1_idx = action_idx
            
            # Populate PyG object
            if action_features:
                data['action'].x = torch.tensor(action_features, dtype=torch.float32)
                data['action'].step_idx = torch.tensor(step_indices, dtype=torch.long)
                data['action'].target_hex_idx = torch.tensor(action_target_hex_idx, dtype=torch.long)
                data['action'].source_unit_idx = torch.tensor(action_source_unit_idx, dtype=torch.long)
                
                if true_a0_idx != -1 and true_a1_idx != -1:
                    data.y_sequence = torch.tensor([true_a0_idx, true_a1_idx], dtype=torch.long)
            else:
                data['action'].x = torch.empty((0, 3), dtype=torch.float32)
                data['action'].step_idx = torch.empty((0,), dtype=torch.long)
                data['action'].target_hex_idx = torch.empty((0,), dtype=torch.long)
                data['action'].source_unit_idx = torch.empty((0,), dtype=torch.long)
        else:
            data['action'].x = torch.empty((0, 3), dtype=torch.float32)
            data['action'].step_idx = torch.empty((0,), dtype=torch.long)
            data['action'].target_hex_idx = torch.empty((0,), dtype=torch.long)
            data['action'].source_unit_idx = torch.empty((0,), dtype=torch.long)

        return data, mask

    def close(self):
        if self.sock:
            self.sock.close()
            self._connected = False
            print("Environment socket closed.")
