import torch
from torch_geometric.data import HeteroData

class StateParser:
    """
    Converts dynamic dictionary state from MegaMek into PyTorch Geometric Graph representations.
    Utilizes Delta Caching for static terrain geometries to reduce parsing overhead.
    """
    # Unit/Entity Feature Indices
    UNIT_FEAT_IS_FRIENDLY = 0
    UNIT_FEAT_X = 1
    UNIT_FEAT_Y = 2
    UNIT_FEAT_SIN_FACING = 3
    UNIT_FEAT_COS_FACING = 4
    UNIT_FEAT_HEAT = 5
    UNIT_FEAT_MAX_HEAT = 6
    UNIT_FEAT_ARMOR = 7
    UNIT_FEAT_STRUCTURE = 8
    UNIT_FEAT_TMM = 9
    UNIT_FEAT_SPEED_MODE = 10
    UNIT_FEAT_GUNNERY = 11
    UNIT_FEAT_PILOTING = 12
    UNIT_FEAT_WALK_MP = 13
    UNIT_FEAT_RUN_MP = 14
    UNIT_FEAT_JUMP_MP = 15
    UNIT_FEAT_WEIGHT = 16
    UNIT_FEAT_IS_PRONE = 17
    UNIT_FEAT_IS_DESTROYED = 18
    UNIT_FEAT_IS_IMMOBILE = 19
    UNIT_FEAT_HEIGHT = 20
    
    UNIT_FEAT_LOCATIONS_ARMOR_OFFSET = 21
    UNIT_FEAT_LOCATIONS_REAR_ARMOR_OFFSET = 29
    UNIT_FEAT_LOCATIONS_INTERNAL_OFFSET = 37

    # Weapon Feature Indices
    WEAPON_FEAT_MIN_RANGE = 0
    WEAPON_FEAT_SHORT_RANGE = 1
    WEAPON_FEAT_MEDIUM_RANGE = 2
    WEAPON_FEAT_LONG_RANGE = 3
    WEAPON_FEAT_DAMAGE = 4
    WEAPON_FEAT_IS_OPERATIONAL = 5
    WEAPON_FEAT_IS_CLUSTER = 6
    WEAPON_FEAT_NUM_CLUSTERS = 7
    WEAPON_FEAT_SALVOS_REMAINING = 8
    WEAPON_FEAT_HEAT = 9

    def __init__(self, ablate_ephemeral=False):
        self.ablate_ephemeral = ablate_ephemeral
        self.static_hex_features = None
        self.static_hex_adjacency_edges = None
        self.board_width = 0
        self.feature_dims = {"hex": 14, "unit": 45, "weapon": 10}

    def handle_topology(self, payload):
        self.board_width = payload.get('width', 0)
        self.feature_dims = payload.get('feature_dims', self.feature_dims)
        nodes = payload.get("hex_nodes", [])
        edges = payload.get("hex_edges", [])
        if nodes:
            self.static_hex_features = torch.tensor(nodes, dtype=torch.float32)
        else:
            hex_dim = self.feature_dims.get("hex", 14)
            self.static_hex_features = torch.empty((0, hex_dim), dtype=torch.float32)
        if edges:
            self.static_hex_adjacency_edges = torch.tensor(edges, dtype=torch.long)
        else:
            self.static_hex_adjacency_edges = torch.empty((2, 0), dtype=torch.long)

    def _populate_action_nodes(self, data, action_features, step_indices, 
                               action_target_hex_idx, action_target_unit_idx, 
                               action_target_weapon_idx, action_source_unit_idx, 
                               action_path_idx, true_sequence_indices, target_action):
        """Helper to consolidate duplicated PyG mapping logic for Action Nodes."""
        if action_features:
            data['action'].x = torch.tensor(action_features, dtype=torch.float32)
            data['action'].step_idx = torch.tensor(step_indices, dtype=torch.long)
            data['action'].target_hex_idx = torch.tensor(action_target_hex_idx, dtype=torch.long)
            data['action'].target_unit_idx = torch.tensor(action_target_unit_idx, dtype=torch.long)
            data['action'].target_weapon_idx = torch.tensor(action_target_weapon_idx, dtype=torch.long)
            data['action'].source_unit_idx = torch.tensor(action_source_unit_idx, dtype=torch.long)
            data['action'].path_idx = torch.tensor(action_path_idx, dtype=torch.long)
            
            if true_sequence_indices and target_action:
                data.y_sequence = torch.tensor(true_sequence_indices, dtype=torch.long)
        else:
            data['action'].x = torch.empty((0, 8), dtype=torch.float32)
            data['action'].step_idx = torch.empty((0,), dtype=torch.long)
            data['action'].target_hex_idx = torch.empty((0,), dtype=torch.long)
            data['action'].target_unit_idx = torch.empty((0,), dtype=torch.long)
            data['action'].target_weapon_idx = torch.empty((0,), dtype=torch.long)
            data['action'].source_unit_idx = torch.empty((0,), dtype=torch.long)
            data['action'].path_idx = torch.empty((0,), dtype=torch.long)

    def parse_to_heterodata(self, payload):
        context = payload.get("context", "UNKNOWN")
        raw_state = payload.get("state", {})
        mask_raw = payload.get("mask", {})
        mask = mask_raw if isinstance(mask_raw, dict) else {}
        
        data = HeteroData()
        data.context = context
        data.action_mask = mask
        
        if context.startswith("MOVEMENT"):
            phase_type = 0.0
        elif context.startswith("WEAPON"):
            phase_type = 1.0
        elif context.startswith("PHYSICAL"):
            phase_type = 2.0
        else:
            phase_type = -1.0
        
        # 1. Global Phase Features
        phase_str = raw_state.get('phase_main', "UNKNOWN")
        turn = raw_state.get('turn_number', 0)
        data.global_context = torch.tensor([turn, len(phase_str)], dtype=torch.float32)
        
        # 2. Dynamic Entity Nodes
        raw_entities = raw_state.get("entities", [])
        unit_dim = self.feature_dims.get("unit", 45)
        if raw_entities:
            t = torch.tensor(raw_entities, dtype=torch.float32)
            pad = torch.zeros((t.size(0), unit_dim - t.size(1)), dtype=torch.float32)
            data['unit'].x = torch.cat([t, pad], dim=1)
        else:
            data['unit'].x = torch.empty((0, unit_dim), dtype=torch.float32)
        
        # 3. Static Hex/Topology Cache
        if self.static_hex_features is not None:
            data['hex'].x = self.static_hex_features
        else:
            hex_dim = self.feature_dims.get("hex", 14)
            data['hex'].x = torch.empty((0, hex_dim), dtype=torch.float32)
        
        if self.static_hex_adjacency_edges is not None:
            edge_arr = self.static_hex_adjacency_edges
            for dir_idx in range(6):
                dir_mask = edge_arr[:, 2] == dir_idx
                data['hex', f'hexAdj_{dir_idx}', 'hex'].edge_index = edge_arr[dir_mask, :2].T.contiguous()
        else:
            for dir_idx in range(6):
                data['hex', f'hexAdj_{dir_idx}', 'hex'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # 4. Weapon Nodes
        raw_weapons = raw_state.get("weapons", [])
        weapon_dim = self.feature_dims.get("weapon", 10)
        if raw_weapons:
            w_tensor = torch.tensor(raw_weapons, dtype=torch.float32)
            w_tensor[w_tensor < 0] = 0.0
            data['weapon'].x = w_tensor
        else:
            data['weapon'].x = torch.empty((0, weapon_dim), dtype=torch.float32)
            
        equips = raw_state.get("equips_edges", [])
        if equips:
            e_arr = torch.tensor(equips, dtype=torch.long).T
            data['weapon', 'equips', 'unit'].edge_index = e_arr
        else:
            data['weapon', 'equips', 'unit'].edge_index = torch.empty((2, 0), dtype=torch.long)
            
        declared_attacks = raw_state.get("declared_attack_edges", [])
        if declared_attacks:
            da_arr = torch.tensor(declared_attacks, dtype=torch.long).T
            data['weapon', 'targeted', 'unit'].edge_index = da_arr
        else:
            data['weapon', 'targeted', 'unit'].edge_index = torch.empty((2, 0), dtype=torch.long)
        
        # 5. Ephemeral Edges (Occupies)
        width = self.board_width
        mech_indices = []
        hex_indices = []
        
        for i, ent in enumerate(raw_entities):
            ex, ey = ent[StateParser.UNIT_FEAT_X], ent[StateParser.UNIT_FEAT_Y]
            if ex >= 0 and ey >= 0 and width > 0:
                hex_idx = int(ey * width + ex)
                mech_indices.append(i)
                hex_indices.append(hex_idx)
                
        if mech_indices:
            data['unit', 'occupies', 'hex'].edge_index = torch.tensor([mech_indices, hex_indices], dtype=torch.long)
        else:
            data['unit', 'occupies', 'hex'].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 6. Ephemeral Threat and LOS Edges
        def add_ephemeral_edges(key, src, dst, out_type):
            if self.ablate_ephemeral:
                data[src, out_type, dst].edge_index = torch.empty((2, 0), dtype=torch.long)
                return
            edges = raw_state.get(key, [])
            if edges:
                e_arr = torch.tensor(edges, dtype=torch.long).T
                data[src, out_type, dst].edge_index = e_arr
            else:
                data[src, out_type, dst].edge_index = torch.empty((2, 0), dtype=torch.long)
                
        add_ephemeral_edges("los_target_edges", "unit", "unit", "LOSTarget")
        add_ephemeral_edges("los_threat_edges", "unit", "hex", "LOSThreat")
        add_ephemeral_edges("partial_cover_edges", "unit", "hex", "partialCover")
        add_ephemeral_edges("movement_threat_edges", "unit", "hex", "movementThreat")
        for tmm in range(5):
            add_ephemeral_edges(f"move_type_tmm_{tmm}_edges", "unit", "hex", f"moveTypeTMM_{tmm}")

        # 7. Dynamic Action Nodes for Autoregressive Trees
        valid_paths = mask.get("valid_paths", [])
        active_entity_idx = mask.get("active_entity_index")
        if active_entity_idx is None:
            active_entity_idx = -1
        target_action = payload.get("target_action", {})
        
        action_features = []
        step_indices = []
        action_target_hex_idx = []
        action_target_unit_idx = []
        action_target_weapon_idx = []
        action_source_unit_idx = []
        action_path_idx = []
        true_sequence_indices = []
        
        if valid_paths and context.startswith("MOVEMENT"):
            true_selected_idx = target_action.get("selected_path_index", -1)
            
            paths_by_source = {}
            for i, p in enumerate(valid_paths):
                src_idx = p.get("source_entity_index", -1)
                if src_idx not in paths_by_source:
                    paths_by_source[src_idx] = []
                paths_by_source[src_idx].append((i, p))
            
            true_a0_idx = -1
            true_a1_idx = -1
            true_dest_index = -1
            
            if true_selected_idx != -1 and true_selected_idx < len(valid_paths):
                true_dest_index = valid_paths[true_selected_idx].get("dest_index", -1)
            
            for src_idx, entity_paths in paths_by_source.items():
                tier0_hexes = {}
                for idx_tuple in entity_paths:
                    i, p = idx_tuple
                    d_idx = p.get("dest_index", -1)
                    if d_idx not in tier0_hexes:
                        tier0_hexes[d_idx] = []
                    tier0_hexes[d_idx].append((i, p))
                    
                unique_hex_list = list(tier0_hexes.keys())
                
                # Tier 0 Candidates for this entity
                for c_idx, dest_idx in enumerate(unique_hex_list):
                    action_idx = len(action_features)
                    step_indices.append(0)
                    action_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, phase_type])
                    action_target_hex_idx.append(dest_idx)
                    action_target_unit_idx.append(-1)
                    action_target_weapon_idx.append(-1)
                    action_source_unit_idx.append(src_idx)
                    action_path_idx.append(-1)
                    if dest_idx == true_dest_index and true_a0_idx == -1 and true_selected_idx in [x[0] for x in tier0_hexes[dest_idx]]:
                        true_a0_idx = action_idx
                        
                # Tier 1 Candidates for this entity
                for dest_idx, valid_tier1_paths in tier0_hexes.items():
                    for raw_path_idx, p_dict in valid_tier1_paths:
                        action_idx = len(action_features)
                        step_indices.append(1)
                        
                        facing = float(p_dict.get("dest_facing", 0))
                        mp_used = float(p_dict.get("mp_used", 0))
                        is_walk = 1.0 if p_dict.get("is_walk", False) else 0.0
                        is_run = 1.0 if p_dict.get("is_run", False) else 0.0
                        is_jump = 1.0 if p_dict.get("is_jump", False) else 0.0
                        
                        has_masc = p_dict.get("has_masc", False)
                        has_sc = p_dict.get("has_supercharger", False)
                        if has_masc and has_sc:
                            booster_type = 3.0
                        elif has_sc:
                            booster_type = 2.0
                        elif has_masc:
                            booster_type = 1.0
                        else:
                            booster_type = 0.0
                            
                        action_features.append([mp_used, facing, is_walk, is_run, is_jump, booster_type, 1.0, phase_type])
                        action_target_hex_idx.append(dest_idx)
                        action_target_unit_idx.append(-1)
                        action_target_weapon_idx.append(-1)
                        action_source_unit_idx.append(src_idx)
                        action_path_idx.append(raw_path_idx)
                            
                        if raw_path_idx == true_selected_idx:
                            true_a1_idx = action_idx
                    

                        
            if true_a0_idx != -1 and true_a1_idx != -1:
                true_sequence_indices = [true_a0_idx, true_a1_idx]
                
            self._populate_action_nodes(data, action_features, step_indices, action_target_hex_idx, 
                                        action_target_unit_idx, action_target_weapon_idx, action_source_unit_idx,
                                        action_path_idx, true_sequence_indices, target_action)
                
        elif "valid_twists" in mask and context.startswith("WEAPON"):
            valid_twists = mask.get("valid_twists", [])
            chosen_attacks = target_action.get("attacks", [])
            chosen_twist = target_action.get("torso_twist", 0)
            chosen_entity_idx = target_action.get("selected_entity_index", -1)
            
            action_type_flags = []
            action_twist_context = []
            
            twists_by_source = {}
            for tm in valid_twists:
                src_idx = tm.get("source_entity_index", -1)
                if src_idx not in twists_by_source:
                    twists_by_source[src_idx] = []
                twists_by_source[src_idx].append(tm)
                
            for src_idx, entity_twists in twists_by_source.items():
                twist_values = [-1, 0, 1]
                twist_node_indices = {}
                for tv in twist_values:
                    idx = len(action_features)
                    twist_node_indices[tv] = idx
                    step_indices.append(0)
                    action_features.append([tv, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, phase_type])
                    action_target_hex_idx.append(-1)
                    action_target_unit_idx.append(-1)
                    action_target_weapon_idx.append(-1)
                    action_source_unit_idx.append(src_idx)
                    action_path_idx.append(-1)
                    action_type_flags.append(0)
                    action_twist_context.append(tv)
                
                for twist_mask in entity_twists:
                    tv = twist_mask.get("twist", 0)
                    twist_mask["node_idx"] = twist_node_indices.get(tv, -1)
                    valid_targets = twist_mask.get("valid_targets", [])
                    
                    for tm in valid_targets:
                        target_entity_index = tm.get("target_entity_index", -1)
                        idx = len(action_features)
                        tm["node_idx"] = idx
                        step_indices.append(0)
                        action_features.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, phase_type])
                        action_target_hex_idx.append(-1)
                        action_target_unit_idx.append(target_entity_index)
                        action_target_weapon_idx.append(-1)
                        action_source_unit_idx.append(src_idx)
                        action_path_idx.append(-1)
                        action_type_flags.append(1)
                        action_twist_context.append(tv)
                        
                        valid_weapons = tm.get("valid_weapons", [])
                        for wm in valid_weapons:
                            weapon_id = wm.get("weapon_id", -1)
                            to_hit = float(wm.get("to_hit", 0.0))
                            sec_to_hit = float(wm.get("secondary_to_hit", to_hit))
                            
                            idx = len(action_features)
                            wm["node_idx"] = idx
                            step_indices.append(0)
                            action_features.append([0.0, 1.0, to_hit, sec_to_hit, 0.0, 0.0, 2.0, phase_type])
                            action_target_hex_idx.append(-1)
                            action_target_unit_idx.append(-1)
                            action_target_weapon_idx.append(weapon_id)
                            action_source_unit_idx.append(src_idx)
                            action_path_idx.append(-1)
                            action_type_flags.append(2)
                            action_twist_context.append(tv)
                            
                end_node_idx = len(action_features)
                step_indices.append(0)
                action_features.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0, phase_type])
                action_target_hex_idx.append(-1)
                action_target_unit_idx.append(-1)
                action_target_weapon_idx.append(-1)
                action_source_unit_idx.append(src_idx)
                action_path_idx.append(-1)
                action_type_flags.append(3)
                action_twist_context.append(-99)
                
                if target_action and src_idx == chosen_entity_idx:
                    true_sequence_indices.append(twist_node_indices.get(chosen_twist, twist_node_indices[0]))
                    if chosen_attacks:
                        target_to_weapons = {}
                        for att in chosen_attacks:
                            t_idx = att.get("target_entity_index", -1)
                            w_id = att.get("weapon_id", -1)
                            if t_idx not in target_to_weapons:
                                target_to_weapons[t_idx] = []
                            target_to_weapons[t_idx].append(w_id)
                        
                        for t_idx, w_ids in target_to_weapons.items():
                            try:
                                t_node_idx = next(i for i, (type_flag, tgt_idx, ctx) in enumerate(zip(action_type_flags, action_target_unit_idx, action_twist_context)) 
                                                if type_flag == 1 and tgt_idx == t_idx and ctx == chosen_twist)
                                true_sequence_indices.append(t_node_idx)
                                for w_id in w_ids:
                                    w_node_idx = next(i for i, (type_flag, tgt_w_idx, ctx) in enumerate(zip(action_type_flags, action_target_weapon_idx, action_twist_context)) 
                                                    if type_flag == 2 and tgt_w_idx == w_id and ctx == chosen_twist)
                                    true_sequence_indices.append(w_node_idx)
                            except StopIteration:
                                continue
                    true_sequence_indices.append(end_node_idx)
                    

            
            self._populate_action_nodes(data, action_features, step_indices, action_target_hex_idx, 
                                        action_target_unit_idx, action_target_weapon_idx, action_source_unit_idx,
                                        action_path_idx, true_sequence_indices, target_action)

        elif "valid_targets" in mask and context.startswith("PHYSICAL"):
            valid_targets = mask.get("valid_targets", [])
            chosen_attacks = target_action.get("attacks", [])
            chosen_entity_idx = target_action.get("selected_entity_index", -1)
            
            action_type_flags = []
            action_type_context = []
            
            targets_by_source = {}
            for tm in valid_targets:
                src_idx = tm.get("source_entity_index", -1)
                if src_idx not in targets_by_source:
                    targets_by_source[src_idx] = []
                targets_by_source[src_idx].append(tm)
                
            for src_idx, entity_targets in targets_by_source.items():
                for tm in entity_targets:
                    target_entity_index = tm.get("target_entity_index", -1)
                    idx = len(action_features)
                    tm["node_idx"] = idx
                    step_indices.append(0)
                    action_features.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, phase_type])
                    action_target_hex_idx.append(-1)
                    action_target_unit_idx.append(target_entity_index)
                    action_target_weapon_idx.append(-1)
                    action_source_unit_idx.append(src_idx)
                    action_path_idx.append(-1)
                    action_type_flags.append(1)
                    action_type_context.append(-1)
                    
                    valid_attacks = tm.get("valid_attacks", [])
                    for am in valid_attacks:
                        action_type = am.get("action_type", -1)
                        to_hit = float(am.get("to_hit", 0.0))
                        idx = len(action_features)
                        am["node_idx"] = idx
                        step_indices.append(0)
                        action_features.append([0.0, 1.0, to_hit, float(action_type), 0.0, 0.0, 2.0, phase_type])
                        action_target_hex_idx.append(-1)
                        action_target_unit_idx.append(-1)
                        action_target_weapon_idx.append(-1)
                        action_source_unit_idx.append(src_idx)
                        action_path_idx.append(-1)
                        action_type_flags.append(2)
                        action_type_context.append(action_type)
                        
                end_node_idx = len(action_features)
                step_indices.append(0)
                action_features.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 3.0, phase_type])
                action_target_hex_idx.append(-1)
                action_target_unit_idx.append(-1)
                action_target_weapon_idx.append(-1)
                action_source_unit_idx.append(src_idx)
                action_path_idx.append(-1)
                action_type_flags.append(3)
                action_type_context.append(-1)
                
                if target_action and src_idx == chosen_entity_idx:
                    if chosen_attacks:
                        for att in chosen_attacks:
                            t_idx = att.get("target_entity_index", -1)
                            p_action = att.get("physical_action_type", -1)
                            try:
                                t_node_idx = next(i for i, (type_flag, tgt_idx) in enumerate(zip(action_type_flags, action_target_unit_idx)) 
                                                if type_flag == 1 and tgt_idx == t_idx)
                                true_sequence_indices.append(t_node_idx)
                                a_node_idx_refined = next(i for i in range(t_node_idx+1, len(action_type_flags))
                                                if action_type_flags[i] == 2 and action_type_context[i] == p_action)
                                true_sequence_indices.append(a_node_idx_refined)
                            except StopIteration:
                                continue
                    true_sequence_indices.append(end_node_idx)
            
            self._populate_action_nodes(data, action_features, step_indices, action_target_hex_idx, 
                                         action_target_unit_idx, action_target_weapon_idx, action_source_unit_idx,
                                         action_path_idx, true_sequence_indices, target_action)
 
        else:
            self._populate_action_nodes(data, [], [], [], [], [], [], [], [], {})

        return data, mask
