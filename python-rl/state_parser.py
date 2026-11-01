import math
import torch
from torch_geometric.data import HeteroData


class StateParser:
    """
    Converts dynamic dictionary state from MegaMek into PyTorch Geometric Graph representations.
    Utilizes Delta Caching for static terrain geometries and constructs a phase-agnostic
    Unified Hierarchical Action Tree on HeteroData['action'] with root SELECT_UNIT nodes.
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

    # Unified Sub-Action Types (8 one-hot classes + 4 normalized continuous features = 12 dims)
    ACTION_SELECT_UNIT = 0
    ACTION_MOVE_HEX = 1
    ACTION_MOVE_PATH = 2
    ACTION_TORSO_TWIST = 3
    ACTION_SELECT_TARGET = 4
    ACTION_FIRE_WEAPON = 5
    ACTION_PHYSICAL_ATTACK = 6
    ACTION_END_DECLARATION = 7
    ACTION_FEATURE_DIM = 12

    # Exact 2d6 cumulative probability table P(2d6 >= N) for target numbers 2..12
    _TWO_D6_PROBS = {
        2: 36.0 / 36.0,
        3: 35.0 / 36.0,
        4: 33.0 / 36.0,
        5: 30.0 / 36.0,
        6: 26.0 / 36.0,
        7: 21.0 / 36.0,
        8: 15.0 / 36.0,
        9: 10.0 / 36.0,
        10: 6.0 / 36.0,
        11: 3.0 / 36.0,
        12: 1.0 / 36.0,
    }

    def __init__(self, ablate_ephemeral=False):
        self.ablate_ephemeral = ablate_ephemeral
        self.static_hex_features = None
        self.static_hex_adjacency_edges = None
        self.board_width = 0
        self.feature_dims = {"hex": 14, "unit": 45, "weapon": 10}

    @classmethod
    def to_hit_probability(cls, target_roll: float) -> float:
        n = int(round(float(target_roll)))
        if n <= 2:
            return 1.0
        if n > 12:
            return 0.0
        return cls._TWO_D6_PROBS.get(n, 0.0)

    @classmethod
    def _make_action_feature(cls, action_type_id: int, f0: float = 0.0, f1: float = 0.0, f2: float = 0.0, f3: float = 0.0):
        vec = [0.0] * cls.ACTION_FEATURE_DIM
        if 0 <= action_type_id < 8:
            vec[action_type_id] = 1.0
        vec[8] = float(f0)
        vec[9] = float(f1)
        vec[10] = float(f2)
        vec[11] = float(f3)
        return vec

    def handle_topology(self, payload):
        self.board_width = payload.get("width", 0)
        self.feature_dims = payload.get("feature_dims", self.feature_dims)
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

    def _populate_action_nodes(
        self,
        data,
        action_features,
        step_indices,
        parent_indices,
        next_parent_indices,
        exclusivity_groups,
        action_type_ids,
        raw_metas,
        action_target_hex_idx,
        action_target_unit_idx,
        action_target_weapon_idx,
        action_source_unit_idx,
        action_path_idx,
        true_sequence_indices,
        target_action,
    ):
        """Consolidates PyG tensor construction for the Unified Hierarchical Action Tree."""
        if action_features:
            data["action"].x = torch.tensor(action_features, dtype=torch.float32)
            data["action"].step_idx = torch.tensor(step_indices, dtype=torch.long)
            data["action"].parent_idx = torch.tensor(parent_indices, dtype=torch.long)
            data["action"].next_parent_idx = torch.tensor(next_parent_indices, dtype=torch.long)
            data["action"].exclusivity_group = torch.tensor(exclusivity_groups, dtype=torch.long)
            data["action"].action_type_id = torch.tensor(action_type_ids, dtype=torch.long)
            data["action"].raw_meta = torch.tensor(raw_metas, dtype=torch.long)
            data["action"].target_hex_idx = torch.tensor(action_target_hex_idx, dtype=torch.long)
            data["action"].target_unit_idx = torch.tensor(action_target_unit_idx, dtype=torch.long)
            data["action"].target_weapon_idx = torch.tensor(action_target_weapon_idx, dtype=torch.long)
            data["action"].source_unit_idx = torch.tensor(action_source_unit_idx, dtype=torch.long)
            data["action"].path_idx = torch.tensor(action_path_idx, dtype=torch.long)

            if true_sequence_indices and target_action:
                data.y_sequence = torch.tensor(true_sequence_indices, dtype=torch.long)
            else:
                data.y_sequence = torch.empty((0,), dtype=torch.long)
            data.y_len = torch.tensor([data.y_sequence.size(0)], dtype=torch.long)
        else:
            data["action"].x = torch.empty((0, self.ACTION_FEATURE_DIM), dtype=torch.float32)
            data["action"].step_idx = torch.empty((0,), dtype=torch.long)
            data["action"].parent_idx = torch.empty((0,), dtype=torch.long)
            data["action"].next_parent_idx = torch.empty((0,), dtype=torch.long)
            data["action"].exclusivity_group = torch.empty((0,), dtype=torch.long)
            data["action"].action_type_id = torch.empty((0,), dtype=torch.long)
            data["action"].raw_meta = torch.empty((0, 4), dtype=torch.long)
            data["action"].target_hex_idx = torch.empty((0,), dtype=torch.long)
            data["action"].target_unit_idx = torch.empty((0,), dtype=torch.long)
            data["action"].target_weapon_idx = torch.empty((0,), dtype=torch.long)
            data["action"].source_unit_idx = torch.empty((0,), dtype=torch.long)
            data["action"].path_idx = torch.empty((0,), dtype=torch.long)
            data.y_sequence = torch.empty((0,), dtype=torch.long)
            data.y_len = torch.tensor([0], dtype=torch.long)

    @staticmethod
    def _collect_selectable_units(mask, items_with_source, raw_entities, fallback_entity_idx=-1):
        units = []
        for u in (mask.get("selectable_entities") or []):
            if u is not None and int(u) >= 0 and int(u) not in units:
                units.append(int(u))
        for item in (items_with_source or []):
            if isinstance(item, dict):
                u = item.get("source_entity_index", -1)
                if u is not None and int(u) >= 0 and int(u) not in units:
                    units.append(int(u))
        if fallback_entity_idx is not None and int(fallback_entity_idx) >= 0 and int(fallback_entity_idx) not in units:
            units.append(int(fallback_entity_idx))
        if not units and raw_entities:
            for idx, ent in enumerate(raw_entities):
                if len(ent) > StateParser.UNIT_FEAT_IS_DESTROYED:
                    if ent[StateParser.UNIT_FEAT_IS_FRIENDLY] > 0.5 and ent[StateParser.UNIT_FEAT_IS_DESTROYED] < 0.5:
                        units.append(idx)
        return units

    def parse_to_heterodata(self, payload):
        context = payload.get("context", "UNKNOWN")
        raw_state = payload.get("state") or {}
        mask_raw = payload.get("mask")
        mask = mask_raw if isinstance(mask_raw, dict) else {}

        data = HeteroData()
        data.context = context

        if context.startswith("MOVEMENT") or context.startswith("DEPLOYMENT"):
            phase_type = 0.0
        elif context.startswith("WEAPON"):
            phase_type = 1.0
        elif context.startswith("PHYSICAL"):
            phase_type = 2.0
        else:
            phase_type = -1.0

        # 1. Global Phase Features (normalized for context_mlp) + Raw Rewards metadata
        turn = float(raw_state.get("turn_number", 0))
        round_num = float(raw_state.get("round_number", 0))
        current_player_id = float(raw_state.get("current_player_id", 0))
        my_activations_left = float(raw_state.get("my_activations_left", 0))
        enemy_activations_left = float(raw_state.get("enemy_activations_left", 0))

        rewards_dict = (payload.get("rewards") or {}) if payload else {}
        bv1 = float(rewards_dict.get("bv1", 0.0))
        bv2 = float(rewards_dict.get("bv2", 0.0))
        init_bv1 = float(rewards_dict.get("initial_bv1", 0.0))
        init_bv2 = float(rewards_dict.get("initial_bv2", 0.0))
        vp1 = float(rewards_dict.get("vp1", 0.0))
        vp2 = float(rewards_dict.get("vp2", 0.0))
        tp1 = float(rewards_dict.get("tp1", 0.0))
        tp2 = float(rewards_dict.get("tp2", 0.0))

        total_init_bv = (init_bv1 + init_bv2) if (init_bv1 + init_bv2) > 0.0 else max(bv1 + bv2, 1.0)
        bv1_norm = bv1 / total_init_bv
        bv2_norm = bv2 / total_init_bv

        data.global_context = torch.tensor(
            [[
                turn / 50.0,
                round_num / 50.0,
                phase_type,
                current_player_id,
                my_activations_left / 12.0,
                enemy_activations_left / 12.0,
                bv1_norm,
                bv2_norm,
                vp1,
                vp2,
                tp1,
                tp2,
            ]],
            dtype=torch.float32,
        )
        data.raw_rewards = torch.tensor(
            [[bv1, bv2, init_bv1, init_bv2, vp1, vp2, tp1, tp2, current_player_id]],
            dtype=torch.float32,
        )

        # 2. Dynamic Entity Nodes
        raw_entities = raw_state.get("entities") or []
        unit_dim = self.feature_dims.get("unit", 45)
        if raw_entities:
            t = torch.tensor(raw_entities, dtype=torch.float32)
            if t.size(1) < unit_dim:
                pad = torch.zeros((t.size(0), unit_dim - t.size(1)), dtype=torch.float32)
                data["unit"].x = torch.cat([t, pad], dim=1)
            else:
                data["unit"].x = t[:, :unit_dim]
        else:
            data["unit"].x = torch.empty((0, unit_dim), dtype=torch.float32)

        # 3. Static Hex/Topology Cache
        if self.static_hex_features is not None:
            data["hex"].x = self.static_hex_features
        else:
            hex_dim = self.feature_dims.get("hex", 14)
            data["hex"].x = torch.empty((0, hex_dim), dtype=torch.float32)

        if self.static_hex_adjacency_edges is not None and self.static_hex_adjacency_edges.numel() > 0:
            edge_arr = self.static_hex_adjacency_edges
            for dir_idx in range(6):
                dir_mask = edge_arr[:, 2] == dir_idx
                data["hex", f"hexAdj_{dir_idx}", "hex"].edge_index = edge_arr[dir_mask, :2].T.contiguous()
        else:
            for dir_idx in range(6):
                data["hex", f"hexAdj_{dir_idx}", "hex"].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 4. Weapon Nodes & Bidirectional Equipment Edges
        raw_weapons = raw_state.get("weapons") or []
        weapon_dim = self.feature_dims.get("weapon", 10)
        if raw_weapons:
            w_tensor = torch.tensor(raw_weapons, dtype=torch.float32)
            w_tensor[w_tensor < 0] = 0.0
            data["weapon"].x = w_tensor
        else:
            data["weapon"].x = torch.empty((0, weapon_dim), dtype=torch.float32)

        equips = raw_state.get("equips_edges") or []
        if equips:
            e_arr = torch.tensor(equips, dtype=torch.long).T.contiguous()
            data["weapon", "equips", "unit"].edge_index = e_arr
            data["unit", "equipped_with", "weapon"].edge_index = e_arr.flip(0).contiguous()
        else:
            data["weapon", "equips", "unit"].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["unit", "equipped_with", "weapon"].edge_index = torch.empty((2, 0), dtype=torch.long)

        declared_attacks = raw_state.get("declared_attack_edges") or []
        if declared_attacks:
            da_arr = torch.tensor(declared_attacks, dtype=torch.long).T.contiguous()
            data["weapon", "targeted", "unit"].edge_index = da_arr
        else:
            data["weapon", "targeted", "unit"].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 5. Ephemeral Edges (Occupies <-> Occupied_by)
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
            occ_edges = torch.tensor([mech_indices, hex_indices], dtype=torch.long)
            data["unit", "occupies", "hex"].edge_index = occ_edges
            data["hex", "occupied_by", "unit"].edge_index = occ_edges.flip(0).contiguous()
        else:
            data["unit", "occupies", "hex"].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["hex", "occupied_by", "unit"].edge_index = torch.empty((2, 0), dtype=torch.long)

        # 6. Ephemeral Threat, LOS, Cover, and Reachability Edges (+ Reverse Edges)
        def add_ephemeral_edges(key, src, dst, out_type, rev_type=None):
            if self.ablate_ephemeral:
                data[src, out_type, dst].edge_index = torch.empty((2, 0), dtype=torch.long)
                if rev_type is not None:
                    data[dst, rev_type, src].edge_index = torch.empty((2, 0), dtype=torch.long)
                return
            edges = raw_state.get(key) or []
            if edges:
                e_arr = torch.tensor(edges, dtype=torch.long).T.contiguous()
                data[src, out_type, dst].edge_index = e_arr
                if rev_type is not None:
                    data[dst, rev_type, src].edge_index = e_arr.flip(0).contiguous()
            else:
                data[src, out_type, dst].edge_index = torch.empty((2, 0), dtype=torch.long)
                if rev_type is not None:
                    data[dst, rev_type, src].edge_index = torch.empty((2, 0), dtype=torch.long)

        add_ephemeral_edges("los_target_edges", "unit", "unit", "LOSTarget", None)
        add_ephemeral_edges("los_threat_edges", "unit", "hex", "LOSThreat", "threatenedByLOS")
        add_ephemeral_edges("partial_cover_edges", "unit", "hex", "partialCover", "providesCoverAgainst")
        add_ephemeral_edges("movement_threat_edges", "unit", "hex", "movementThreat", "threatenedByMove")
        for tmm in range(5):
            add_ephemeral_edges(f"move_type_tmm_{tmm}_edges", "unit", "hex", f"moveTypeTMM_{tmm}", f"reachableTMM_{tmm}")

        # 7. Unified Phase-Agnostic Action Tree (SELECT_UNIT at Step 0)
        target_action = (payload.get("target_action") or mask.get("target_action") or {})
        if not isinstance(target_action, dict):
            target_action = {}

        action_features = []
        step_indices = []
        parent_indices = []
        next_parent_indices = []
        exclusivity_groups = []
        action_type_ids = []
        raw_metas = []
        action_target_hex_idx = []
        action_target_unit_idx = []
        action_target_weapon_idx = []
        action_source_unit_idx = []
        action_path_idx = []
        true_sequence_indices = []

        def add_node(
            act_type,
            step_k,
            parent_id,
            next_parent_id,
            excl_group=-1,
            src_unit=-1,
            tgt_hex=-1,
            tgt_unit=-1,
            tgt_weapon=-1,
            path_id=-1,
            meta=(-1, 0, -1, -1),
            f0=0.0,
            f1=0.0,
            f2=0.0,
            f3=0.0,
        ):
            node_idx = len(action_features)
            resolved_next_parent = node_idx if next_parent_id == -999 else next_parent_id
            action_features.append(self._make_action_feature(act_type, f0, f1, f2, f3))
            step_indices.append(step_k)
            parent_indices.append(parent_id)
            next_parent_indices.append(resolved_next_parent)
            exclusivity_groups.append(excl_group)
            action_type_ids.append(act_type)
            raw_metas.append(list(meta))
            action_source_unit_idx.append(src_unit)
            action_target_hex_idx.append(tgt_hex)
            action_target_unit_idx.append(tgt_unit)
            action_target_weapon_idx.append(tgt_weapon)
            action_path_idx.append(path_id)
            return node_idx

        phase_norm = max(phase_type, 0.0) / 2.0

        if context.startswith("MOVEMENT"):
            valid_paths = mask.get("valid_paths") or []
            true_selected_idx = target_action.get("selected_path_index", -1)
            if true_selected_idx is None:
                true_selected_idx = -1

            paths_by_source = {}
            for i, p in enumerate(valid_paths):
                src_idx = p.get("source_entity_index", -1)
                if src_idx is None:
                    src_idx = -1
                paths_by_source.setdefault(int(src_idx), []).append((i, p))

            for src_idx, entity_paths in paths_by_source.items():
                if not entity_paths:
                    continue
                # Step 0: SELECT_UNIT
                u_node_idx = add_node(
                    self.ACTION_SELECT_UNIT,
                    step_k=0,
                    parent_id=-1,
                    next_parent_id=-999,
                    src_unit=src_idx,
                    tgt_unit=src_idx,
                    meta=(-1, 0, -1, -1),
                    f0=phase_norm,
                )

                tier1_hexes = {}
                for raw_path_idx, p_dict in entity_paths:
                    d_idx = int(p_dict.get("dest_index", -1))
                    tier1_hexes.setdefault(d_idx, []).append((raw_path_idx, p_dict))

                # Step 1: MOVE_HEX
                hex_node_map = {}
                for dest_idx in tier1_hexes.keys():
                    hex_node_idx = add_node(
                        self.ACTION_MOVE_HEX,
                        step_k=1,
                        parent_id=u_node_idx,
                        next_parent_id=-999,
                        src_unit=src_idx,
                        tgt_hex=dest_idx,
                        meta=(-1, 0, dest_idx, -1),
                        f0=phase_norm,
                    )
                    hex_node_map[dest_idx] = hex_node_idx

                # Step 2: MOVE_PATH (terminal leaves: next_parent_id = -2)
                for dest_idx, hex_paths in tier1_hexes.items():
                    hex_node_idx = hex_node_map[dest_idx]
                    for raw_path_idx, p_dict in hex_paths:
                        facing = float(p_dict.get("dest_facing", 0))
                        mp_used = float(p_dict.get("mp_used", 0))
                        is_walk = bool(p_dict.get("is_walk", False))
                        is_run = bool(p_dict.get("is_run", False))
                        is_jump = bool(p_dict.get("is_jump", False))
                        has_masc = bool(p_dict.get("has_masc", False))
                        has_sc = bool(p_dict.get("has_supercharger", False))

                        if has_masc or has_sc:
                            mode_val = 1.0
                        elif is_jump:
                            mode_val = 0.75
                        elif is_run:
                            mode_val = 0.50
                        elif is_walk:
                            mode_val = 0.25
                        else:
                            mode_val = 0.0

                        sin_f = math.sin(facing * math.pi / 3.0)
                        cos_f = math.cos(facing * math.pi / 3.0)
                        mp_norm = min(max(mp_used, 0.0) / 20.0, 1.0)

                        path_node_idx = add_node(
                            self.ACTION_MOVE_PATH,
                            step_k=2,
                            parent_id=hex_node_idx,
                            next_parent_id=-2,
                            src_unit=src_idx,
                            tgt_hex=dest_idx,
                            path_id=raw_path_idx,
                            meta=(raw_path_idx, int(facing), dest_idx, -1),
                            f0=mp_norm,
                            f1=sin_f,
                            f2=cos_f,
                            f3=mode_val,
                        )
                        p_dict["node_idx"] = path_node_idx
                        p_dict["tier0_node_idx"] = hex_node_idx
                        p_dict["unit_node_idx"] = u_node_idx

                        if raw_path_idx == true_selected_idx and not true_sequence_indices:
                            true_sequence_indices = [u_node_idx, hex_node_idx, path_node_idx]

            self._populate_action_nodes(
                data,
                action_features,
                step_indices,
                parent_indices,
                next_parent_indices,
                exclusivity_groups,
                action_type_ids,
                raw_metas,
                action_target_hex_idx,
                action_target_unit_idx,
                action_target_weapon_idx,
                action_source_unit_idx,
                action_path_idx,
                true_sequence_indices,
                target_action,
            )

        elif context.startswith("WEAPON"):
            valid_twists = mask.get("valid_twists") or []
            chosen_entity_idx = target_action.get(
                "selected_entity_index", target_action.get("selected_entity_id", -1)
            )
            if chosen_entity_idx is None:
                chosen_entity_idx = -1
            chosen_twist = target_action.get("torso_twist", target_action.get("twist", 0))
            if chosen_twist is None:
                chosen_twist = 0
            chosen_attacks = target_action.get("attacks") or []

            selectable_units = self._collect_selectable_units(
                mask, valid_twists, raw_entities, fallback_entity_idx=chosen_entity_idx
            )
            if not selectable_units and valid_twists:
                selectable_units = [-1]

            twists_by_source = {}
            for tm in valid_twists:
                src_idx = tm.get("source_entity_index", -1)
                if src_idx is None:
                    src_idx = -1
                twists_by_source.setdefault(int(src_idx), []).append(tm)

            for src_idx in selectable_units:
                # Step 0: SELECT_UNIT
                u_node_idx = add_node(
                    self.ACTION_SELECT_UNIT,
                    step_k=0,
                    parent_id=-1,
                    next_parent_id=-999,
                    src_unit=src_idx,
                    tgt_unit=src_idx,
                    meta=(-1, 0, -1, -1),
                    f0=phase_norm,
                )

                entity_twists = twists_by_source.get(src_idx, [])
                # Ensure twist == 0 always exists so every selectable unit can at least twist=0 -> END
                seen_twists = {}
                for tm in entity_twists:
                    tv = int(tm.get("twist", 0))
                    if tv not in seen_twists:
                        seen_twists[tv] = tm
                if 0 not in seen_twists:
                    seen_twists[0] = {"source_entity_index": src_idx, "twist": 0, "valid_targets": []}
                if src_idx == chosen_entity_idx and int(chosen_twist) not in seen_twists:
                    seen_twists[int(chosen_twist)] = {
                        "source_entity_index": src_idx,
                        "twist": int(chosen_twist),
                        "valid_targets": [],
                    }

                twist_lookup = {}
                for tv in sorted(seen_twists.keys()):
                    twist_mask = seen_twists[tv]
                    twist_node_idx = add_node(
                        self.ACTION_TORSO_TWIST,
                        step_k=1,
                        parent_id=u_node_idx,
                        next_parent_id=-999,
                        src_unit=src_idx,
                        tgt_unit=src_idx,
                        meta=(-1, tv, -1, -1),
                        f0=float(tv) / 3.0,
                    )
                    twist_mask["node_idx"] = twist_node_idx

                    target_lookup = {}
                    for tm in (twist_mask.get("valid_targets") or []):
                        target_entity_index = int(tm.get("target_entity_index", -1))
                        valid_weapons = tm.get("valid_weapons") or []
                        if not valid_weapons:
                            continue
                        target_node_idx = add_node(
                            self.ACTION_SELECT_TARGET,
                            step_k=2,
                            parent_id=twist_node_idx,
                            next_parent_id=-999,
                            src_unit=src_idx,
                            tgt_unit=target_entity_index,
                            meta=(-1, tv, target_entity_index, -1),
                            f0=phase_norm,
                        )
                        tm["node_idx"] = target_node_idx

                        weapon_lookup = {}
                        for wm in valid_weapons:
                            weapon_id = int(wm.get("weapon_id", -1))
                            to_hit_prob = self.to_hit_probability(wm.get("to_hit", 13.0))
                            sec_to_hit_prob = self.to_hit_probability(wm.get("secondary_to_hit", wm.get("to_hit", 13.0)))
                            w_node_idx = add_node(
                                self.ACTION_FIRE_WEAPON,
                                step_k=3,
                                parent_id=target_node_idx,
                                next_parent_id=twist_node_idx,  # Return to TORSO_TWIST to select next target or END
                                excl_group=weapon_id,
                                src_unit=src_idx,
                                tgt_unit=target_entity_index,
                                tgt_weapon=weapon_id,
                                meta=(-1, tv, target_entity_index, weapon_id),
                                f0=to_hit_prob,
                                f1=sec_to_hit_prob,
                            )
                            wm["node_idx"] = w_node_idx
                            weapon_lookup[weapon_id] = w_node_idx
                        target_lookup[target_entity_index] = (target_node_idx, weapon_lookup)

                    # Terminal END_DECLARATION node under this TORSO_TWIST
                    end_node_idx = add_node(
                        self.ACTION_END_DECLARATION,
                        step_k=2,
                        parent_id=twist_node_idx,
                        next_parent_id=-2,
                        src_unit=src_idx,
                        tgt_unit=src_idx,
                        meta=(-1, tv, -1, -1),
                        f0=phase_norm,
                    )
                    twist_lookup[tv] = (twist_node_idx, target_lookup, end_node_idx)

                # Reconstruct teacher-forcing trajectory y_sequence if target_action matches this unit
                if target_action and (src_idx == chosen_entity_idx or (chosen_entity_idx == -1 and len(selectable_units) == 1)):
                    tv_key = int(chosen_twist) if int(chosen_twist) in twist_lookup else 0
                    twist_node_idx, target_lookup, end_node_idx = twist_lookup[tv_key]
                    seq = [u_node_idx, twist_node_idx]
                    used_w = set()
                    for att in chosen_attacks:
                        if not isinstance(att, dict):
                            continue
                        t_idx = int(att.get("target_entity_index", att.get("target_id", -1)))
                        w_id = int(att.get("weapon_id", -1))
                        if w_id in used_w:
                            continue
                        if t_idx in target_lookup:
                            t_node_idx, weapon_lookup = target_lookup[t_idx]
                            if w_id in weapon_lookup:
                                seq.append(t_node_idx)
                                seq.append(weapon_lookup[w_id])
                                used_w.add(w_id)
                    seq.append(end_node_idx)
                    true_sequence_indices = seq

            self._populate_action_nodes(
                data,
                action_features,
                step_indices,
                parent_indices,
                next_parent_indices,
                exclusivity_groups,
                action_type_ids,
                raw_metas,
                action_target_hex_idx,
                action_target_unit_idx,
                action_target_weapon_idx,
                action_source_unit_idx,
                action_path_idx,
                true_sequence_indices,
                target_action,
            )

        elif context.startswith("PHYSICAL"):
            valid_targets = mask.get("valid_targets") or []
            chosen_entity_idx = target_action.get(
                "selected_entity_index", target_action.get("selected_entity_id", -1)
            )
            if chosen_entity_idx is None:
                chosen_entity_idx = -1

            # Extract chosen physical attack from either BC ("attacks": [...]) or RL ("attack": {...})
            chosen_target_id = -1
            chosen_phys_type = -1
            if isinstance(target_action.get("attack"), dict):
                att = target_action["attack"]
                chosen_target_id = int(att.get("target_id", att.get("target_entity_index", -1)))
                chosen_phys_type = int(att.get("action_type", att.get("physical_action_type", -1)))
            elif isinstance(target_action.get("attacks"), list) and len(target_action["attacks"]) > 0:
                first_att = target_action["attacks"][0]
                if isinstance(first_att, dict):
                    chosen_target_id = int(first_att.get("target_entity_index", first_att.get("target_id", -1)))
                    chosen_phys_type = int(first_att.get("physical_action_type", first_att.get("action_type", -1)))

            selectable_units = self._collect_selectable_units(
                mask, valid_targets, raw_entities, fallback_entity_idx=chosen_entity_idx
            )
            if not selectable_units and valid_targets:
                selectable_units = [-1]

            targets_by_source = {}
            for tm in valid_targets:
                src_idx = tm.get("source_entity_index", -1)
                if src_idx is None:
                    src_idx = -1
                targets_by_source.setdefault(int(src_idx), []).append(tm)

            for src_idx in selectable_units:
                # Step 0: SELECT_UNIT
                u_node_idx = add_node(
                    self.ACTION_SELECT_UNIT,
                    step_k=0,
                    parent_id=-1,
                    next_parent_id=-999,
                    src_unit=src_idx,
                    tgt_unit=src_idx,
                    meta=(-1, 0, -1, -1),
                    f0=phase_norm,
                )

                target_lookup = {}
                for tm in targets_by_source.get(src_idx, []):
                    target_entity_index = int(tm.get("target_entity_index", -1))
                    valid_attacks = tm.get("valid_attacks") or []
                    if not valid_attacks:
                        continue
                    target_node_idx = add_node(
                        self.ACTION_SELECT_TARGET,
                        step_k=1,
                        parent_id=u_node_idx,
                        next_parent_id=-999,
                        src_unit=src_idx,
                        tgt_unit=target_entity_index,
                        meta=(-1, 0, target_entity_index, -1),
                        f0=phase_norm,
                    )
                    tm["node_idx"] = target_node_idx

                    phys_lookup = {}
                    for am in valid_attacks:
                        action_type = int(am.get("action_type", -1))
                        to_hit_prob = self.to_hit_probability(am.get("to_hit", 13.0))
                        phys_node_idx = add_node(
                            self.ACTION_PHYSICAL_ATTACK,
                            step_k=2,
                            parent_id=target_node_idx,
                            next_parent_id=-2,  # Terminal leaf
                            src_unit=src_idx,
                            tgt_unit=target_entity_index,
                            meta=(-1, 0, target_entity_index, action_type),
                            f0=to_hit_prob,
                            f1=float(action_type) / 13.0,
                        )
                        am["node_idx"] = phys_node_idx
                        phys_lookup[action_type] = phys_node_idx
                    target_lookup[target_entity_index] = (target_node_idx, phys_lookup)

                # Terminal END_DECLARATION node under SELECT_UNIT (for passing physical attack)
                end_node_idx = add_node(
                    self.ACTION_END_DECLARATION,
                    step_k=1,
                    parent_id=u_node_idx,
                    next_parent_id=-2,
                    src_unit=src_idx,
                    tgt_unit=src_idx,
                    meta=(-1, 0, -1, -1),
                    f0=phase_norm,
                )

                if target_action and (src_idx == chosen_entity_idx or (chosen_entity_idx == -1 and len(selectable_units) == 1)):
                    if chosen_target_id in target_lookup and chosen_phys_type in target_lookup[chosen_target_id][1]:
                        t_node_idx, phys_lookup = target_lookup[chosen_target_id]
                        true_sequence_indices = [u_node_idx, t_node_idx, phys_lookup[chosen_phys_type]]
                    else:
                        true_sequence_indices = [u_node_idx, end_node_idx]

            self._populate_action_nodes(
                data,
                action_features,
                step_indices,
                parent_indices,
                next_parent_indices,
                exclusivity_groups,
                action_type_ids,
                raw_metas,
                action_target_hex_idx,
                action_target_unit_idx,
                action_target_weapon_idx,
                action_source_unit_idx,
                action_path_idx,
                true_sequence_indices,
                target_action,
            )

        else:
            self._populate_action_nodes(
                data, [], [], [], [], [], [], [], [], [], [], [], [], [], {}
            )

        return data, mask
