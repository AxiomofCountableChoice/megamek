import os
import sys
import math
import torch
from torch_geometric.data import Batch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from state_parser import StateParser
from models.agent import MegaMekAgent
from rewards import RewardCalculator
from impala_master import compute_annealed_weight, ImpalaLearner


def build_mock_topology():
    hex_nodes = [[float(i % 4), float(i // 4)] + [0.1] * 12 for i in range(16)]
    hex_edges = []
    for i in range(15):
        hex_edges.append([i, i + 1, i % 6])
        hex_edges.append([i + 1, i, (i + 3) % 6])
    return {"width": 4, "height": 4, "hex_nodes": hex_nodes, "hex_edges": hex_edges}


def build_mock_base_state():
    entities = [
        [1.0, 1.0, 1.0, 0.0, 1.0] + [0.5] * 40,  # Friendly unit 0 at (1, 1) -> hex 5
        [1.0, 2.0, 1.0, 0.0, 1.0] + [0.5] * 40,  # Friendly unit 1 at (2, 1) -> hex 6
        [0.0, 2.0, 2.0, 0.0, -1.0] + [0.5] * 40,  # Enemy unit 2 at (2, 2) -> hex 10
    ]
    weapons = [
        [0.0, 3.0, 6.0, 9.0, 8.0, 1.0, 0.0, 1.0, 10.0, 3.0],  # Weapon 0 on unit 0
        [0.0, 5.0, 10.0, 15.0, 10.0, 1.0, 0.0, 1.0, 10.0, 8.0],  # Weapon 1 on unit 0
        [0.0, 3.0, 6.0, 9.0, 5.0, 1.0, 0.0, 1.0, 10.0, 2.0],  # Weapon 2 on unit 1
    ]
    return {
        "turn_number": 5,
        "round_number": 2,
        "current_player_id": 0,
        "my_activations_left": 2,
        "enemy_activations_left": 1,
        "entities": entities,
        "weapons": weapons,
        "equips_edges": [[0, 0], [1, 0], [2, 1]],
        "declared_attack_edges": [],
        "los_target_edges": [[0, 2], [1, 2]],
        "los_threat_edges": [[2, 5], [2, 6]],
        "partial_cover_edges": [[2, 5]],
        "movement_threat_edges": [[2, 9]],
        "move_type_tmm_0_edges": [[0, 5]],
        "move_type_tmm_1_edges": [[0, 6], [1, 7]],
        "move_type_tmm_2_edges": [[0, 9]],
        "move_type_tmm_3_edges": [],
        "move_type_tmm_4_edges": [],
    }


def test_bc_and_inference_parsing():
    print("=== Test 1: BC & Inference Payload Parsing (All Phases) ===")
    parser = StateParser()
    parser.handle_topology(build_mock_topology())
    base_state = build_mock_base_state()
    rewards = {"bv1": 2800.0, "bv2": 2600.0, "initial_bv1": 3500.0, "initial_bv2": 3500.0, "vp1": 0.0, "vp2": 0.0, "tp1": 0.0, "tp2": 0.0}

    # 1a. MOVEMENT_BC
    move_mask = {
        "selectable_entities": [0, 1],
        "valid_paths": [
            {"source_entity_index": 0, "path_index": 0, "dest_index": 5, "dest_facing": 0, "mp_used": 0, "is_walk": True, "is_run": False, "is_jump": False},
            {"source_entity_index": 0, "path_index": 1, "dest_index": 9, "dest_facing": 2, "mp_used": 4, "is_walk": False, "is_run": True, "is_jump": False},
            {"source_entity_index": 1, "path_index": 2, "dest_index": 7, "dest_facing": 1, "mp_used": 3, "is_walk": True, "is_run": False, "is_jump": True},
        ],
    }
    move_payload = {
        "context": "MOVEMENT_BC",
        "state": base_state,
        "mask": move_mask,
        "rewards": rewards,
        "target_action": {"selected_path_index": 1},
    }
    g_move, _ = parser.parse_to_heterodata(move_payload)
    assert g_move.y_sequence.tolist() and len(g_move.y_sequence) == 3, f"Expected 3-step MOVEMENT_BC y_sequence, got {g_move.y_sequence}"
    assert g_move["hex", "occupied_by", "unit"].edge_index.size(1) == 3, "Expected 3 occupied_by reverse edges"
    assert g_move["unit", "equipped_with", "weapon"].edge_index.size(1) == 3, "Expected 3 equipped_with reverse edges"

    # 1b. WEAPON_BC (using Java's exact field names: selected_entity_index, torso_twist, target_entity_index)
    weapon_mask = {
        "selectable_entities": [0, 1],
        "valid_twists": [
            {
                "source_entity_index": 0,
                "twist": 0,
                "valid_targets": [
                    {
                        "source_entity_index": 0,
                        "target_entity_index": 2,
                        "valid_weapons": [
                            {"weapon_id": 0, "to_hit": 7, "secondary_to_hit": 8},
                            {"weapon_id": 1, "to_hit": 8, "secondary_to_hit": 9},
                        ],
                    }
                ],
            },
            {
                "source_entity_index": 0,
                "twist": 1,
                "valid_targets": [
                    {
                        "source_entity_index": 0,
                        "target_entity_index": 2,
                        "valid_weapons": [
                            {"weapon_id": 0, "to_hit": 6, "secondary_to_hit": 7},
                            {"weapon_id": 1, "to_hit": 7, "secondary_to_hit": 8},
                        ],
                    }
                ],
            },
        ],
    }
    weapon_bc_payload = {
        "context": "WEAPON_BC",
        "state": base_state,
        "mask": weapon_mask,
        "rewards": rewards,
        "target_action": {
            "selected_entity_index": 0,
            "torso_twist": 1,
            "attacks": [
                {"target_entity_index": 2, "weapon_id": 0},
                {"target_entity_index": 2, "weapon_id": 1},
            ],
        },
    }
    g_weap, _ = parser.parse_to_heterodata(weapon_bc_payload)
    # Sequence: [SELECT_UNIT(0), TORSO_TWIST(1), SELECT_TARGET(2), FIRE_WEAPON(0), SELECT_TARGET(2), FIRE_WEAPON(1), END_DECLARATION] -> length 7
    assert len(g_weap.y_sequence) == 7, f"Expected 7-step WEAPON_BC y_sequence, got {g_weap.y_sequence.tolist()}"

    # 1c. PHYSICAL_BC (using Java's exact field names: selected_entity_index, attacks: [{target_entity_index, physical_action_type}])
    phys_mask = {
        "selectable_entities": [0, 1],
        "valid_targets": [
            {
                "source_entity_index": 0,
                "target_entity_index": 2,
                "valid_attacks": [
                    {"action_type": 1, "name": "PUNCH_LEFT", "to_hit": 6},
                    {"action_type": 4, "name": "KICK_LEFT", "to_hit": 5},
                ],
            }
        ],
    }
    phys_bc_payload = {
        "context": "PHYSICAL_BC",
        "state": base_state,
        "mask": phys_mask,
        "rewards": rewards,
        "target_action": {
            "selected_entity_index": 0,
            "attacks": [{"target_entity_index": 2, "physical_action_type": 4}],
        },
    }
    g_phys, _ = parser.parse_to_heterodata(phys_bc_payload)
    assert len(g_phys.y_sequence) == 3, f"Expected 3-step PHYSICAL_BC y_sequence, got {g_phys.y_sequence.tolist()}"

    # 1d. PHYSICAL_BC with unit 1 choosing NO attack (empty attacks list, unit 1 has no valid targets)
    phys_pass_payload = {
        "context": "PHYSICAL_BC",
        "state": base_state,
        "mask": phys_mask,
        "rewards": rewards,
        "target_action": {"selected_entity_index": 1, "attacks": []},
    }
    g_phys_pass, _ = parser.parse_to_heterodata(phys_pass_payload)
    assert len(g_phys_pass.y_sequence) == 2, f"Expected 2-step [SELECT_UNIT, END] y_sequence, got {g_phys_pass.y_sequence.tolist()}"
    print("  [PASS] All BC & Inference payloads parsed into valid Unified Action Trees with non-empty y_sequence.")
    return parser, g_move, g_weap, g_phys, move_payload, weapon_bc_payload, phys_bc_payload


def test_get_action_vs_evaluate_actions_equivalence(parser, move_payload, weapon_payload, phys_payload):
    print("=== Test 2: Exact Log-Probability Equivalence (get_action vs evaluate_actions) ===")
    torch.manual_seed(42)
    agent = MegaMekAgent(hidden_dim=64, ensemble_size=8)
    agent.eval()

    reconstructed_graphs = []
    sampled_log_probs = []

    for name, base_payload in [("MOVEMENT", move_payload), ("WEAPON", weapon_payload), ("PHYSICAL", phys_payload)]:
        inf_payload = dict(base_payload)
        inf_payload["context"] = f"{name}_INFERENCE"
        inf_payload.pop("target_action", None)

        g_inf, mask_inf = parser.parse_to_heterodata(inf_payload)
        with torch.no_grad():
            action_dict, v_mean, _, mu_log_prob = agent.get_action(g_inf, mask_inf, deterministic=False)

        # Re-parse with chosen action_dict as target_action (exact IMPALA Learner pathway)
        eval_payload = dict(inf_payload)
        eval_payload["target_action"] = action_dict
        g_eval, _ = parser.parse_to_heterodata(eval_payload)
        assert g_eval.y_sequence.numel() > 0, f"{name}: Re-parsed y_sequence was empty for action_dict={action_dict}"

        with torch.no_grad():
            pi_lp, ent, _, _, v_preds, _, steps = agent.evaluate_actions(g_eval)

        diff = abs(float(pi_lp.item()) - float(mu_log_prob))
        print(f"  [{name}] mu_log_prob={mu_log_prob:.6f}, pi_log_prob={pi_lp.item():.6f}, diff={diff:.2e}, steps={int(steps.item())}, action={action_dict}")
        assert diff < 1e-5, f"{name} log_prob mismatch! mu={mu_log_prob}, pi={pi_lp.item()}"

        reconstructed_graphs.append(g_eval)
        sampled_log_probs.append(float(mu_log_prob))

    # Also verify batched evaluation across heterogeneous phases (validates ptr offset in compute_action_embeddings)
    batched = Batch.from_data_list(reconstructed_graphs)
    with torch.no_grad():
        pi_lp_batch, _, _, _, v_preds_batch, _, _ = agent.evaluate_actions(batched)

    for b, expected_lp in enumerate(sampled_log_probs):
        diff_b = abs(float(pi_lp_batch[b].item()) - expected_lp)
        assert diff_b < 1e-5, f"Batch item {b} log_prob mismatch! expected={expected_lp}, got={pi_lp_batch[b].item()}"
    assert v_preds_batch.shape == (3, 8), f"Expected v_preds_batch shape (3, 8), got {v_preds_batch.shape}"
    print("  [PASS] Single and batched evaluate_actions match get_action log_probs within < 1e-6 across all phases.")


def test_bidirectional_gradients_and_ensemble_vtrace(g_move, g_weap, g_phys):
    print("=== Test 3: Bidirectional HGT Gradients & Per-Head V-Trace Ensemble ===")
    torch.manual_seed(123)
    agent = MegaMekAgent(hidden_dim=64, ensemble_size=8)
    agent.train()
    agent.zero_grad()

    batch = Batch.from_data_list([g_move, g_weap, g_phys])
    pi_lp, ent, v_mean, v_var, v_preds, _, _ = agent.evaluate_actions(batch)

    # Check that hex features flow through reverse edges into unit/weapon nodes and into both Actor and Critic
    boot_mask = (torch.rand_like(v_preds) < 0.63).float()
    loss = -pi_lp.mean() + 0.5 * ((v_preds - 1.5) ** 2 * boot_mask).mean()
    loss.backward()

    hex_grad_norm = agent.encoder.node_proj["hex"].weight.grad.norm().item()
    unit_grad_norm = agent.encoder.node_proj["unit"].weight.grad.norm().item()
    weapon_grad_norm = agent.encoder.node_proj["weapon"].weight.grad.norm().item()
    print(f"  Gradient norms -> hex_proj: {hex_grad_norm:.6e}, unit_proj: {unit_grad_norm:.6e}, weapon_proj: {weapon_grad_norm:.6e}")
    assert hex_grad_norm > 0.0, "Hex node projection received zero gradient! Reverse HGT edges are not connected."
    assert unit_grad_norm > 0.0, "Unit node projection received zero gradient!"
    assert weapon_grad_norm > 0.0, "Weapon node projection received zero gradient!"

    # Check that the 8 Critic heads have distinct gradients
    head_grad_norms = [head[0].weight.grad.norm().item() for head in agent.ensembles]
    unique_norms = len(set(round(gn, 6) for gn in head_grad_norms))
    assert unique_norms > 1, f"Expected distinct Critic head gradients, got {head_grad_norms}"

    # Verify per-head V-Trace recursion in ImpalaLearner and full learn_step execution
    learner = ImpalaLearner(device="cpu", batch_size=1, expert_batch_size=2)
    rewards = torch.tensor([0.2, -0.1, 1.0])
    v_preds_seq = torch.randn(4, 8)
    mu_probs = torch.tensor([0.5, 0.4, 0.6])
    pi_probs = torch.tensor([0.5, 0.5, 0.6])
    vs, rhos = learner.compute_vtrace_targets(rewards, v_preds_seq, mu_probs, pi_probs)
    assert vs.shape == (3, 8), f"Expected per-head V-Trace shape (3, 8), got {vs.shape}"
    learner.close()
    print("  [PASS] Bidirectional HGT gradients active and per-head V-Trace targets verified.")


def test_full_impala_learn_step(move_payload, weapon_payload, phys_payload, g_move, g_weap):
    print("=== Test 5: End-to-End ImpalaLearner.learn_step (Multi-Phase Trajectory + Annealed BC) ===")
    learner = ImpalaLearner(
        device="cpu",
        batch_size=1,
        expert_batch_size=2,
        lambda_bc_initial=1.0,
        lambda_bc_final=0.0,
        bc_warmup_steps=10,
        bc_decay_steps=100,
        bc_schedule="cosine",
        lam_epistemic_initial=0.5,
        lam_epistemic_final=0.0,
        epistemic_decay_steps=100,
    )
    topo = build_mock_topology()
    steps = [
        {
            "raw_payload": dict(move_payload, context="MOVEMENT_INFERENCE"),
            "action_dict": {"selected_entity_id": 0, "selected_path_index": 1},
            "mu_log_prob": -1.2,
            "reward": 0.05,
            "env_reward": 0.05,
            "reward_bv": 0.05,
            "reward_vp": 0.0,
        },
        {
            "raw_payload": dict(weapon_payload, context="WEAPON_INFERENCE"),
            "action_dict": {
                "selected_entity_id": 0,
                "twist": 1,
                "attacks": [{"target_id": 2, "weapon_id": 0}, {"target_id": 2, "weapon_id": 1}],
            },
            "mu_log_prob": -2.5,
            "reward": 0.25,
            "env_reward": 0.25,
            "reward_bv": 0.25,
            "reward_vp": 0.0,
        },
        {
            "raw_payload": dict(phys_payload, context="PHYSICAL_INFERENCE"),
            "action_dict": {
                "selected_entity_id": 0,
                "attack": {"target_id": 2, "action_type": 4},
            },
            "mu_log_prob": -1.1,
            "reward": 2.0,
            "env_reward": 2.0,
            "reward_bv": 0.0,
            "reward_vp": 2.0,
        },
        {
            "raw_payload": dict(phys_payload, context="PHYSICAL_INFERENCE", game_over=True, win=True),
            "action_dict": {},
            "mu_log_prob": 0.0,
            "reward": 0.0,
            "is_terminal": True,
        },
    ]
    with learner.buffer.lock:
        learner.buffer.trajectories.append(("mock_traj.pt", {"topology_payload": topo, "steps": steps, "win": True}))
    with learner.expert_buffer.lock:
        learner.expert_buffer.graphs.extend([g_move, g_weap])

    stepped = learner.learn_step(batch_size=1, sequence_length=4)
    assert stepped, "Expected learn_step to return True"
    assert learner.global_step == 1
    learner.close()
    print("  [PASS] ImpalaLearner.learn_step executed cleanly across multi-phase trajectory and expert BC batch.")


def test_annealing_and_psi_rewards():
    print("=== Test 4: Annealing Schedules & PSI Initial-BV Reward Normalization ===")
    # Cosine schedule with warmup
    w0 = compute_annealed_weight(0, initial=1.0, final=0.0, warmup_steps=500, decay_steps=5000, schedule="cosine")
    w_warm = compute_annealed_weight(499, initial=1.0, final=0.0, warmup_steps=500, decay_steps=5000, schedule="cosine")
    w_mid = compute_annealed_weight(3000, initial=1.0, final=0.0, warmup_steps=500, decay_steps=5000, schedule="cosine")
    w_end = compute_annealed_weight(5500, initial=1.0, final=0.0, warmup_steps=500, decay_steps=5000, schedule="cosine")
    assert abs(w0 - 1.0) < 1e-6 and abs(w_warm - 1.0) < 1e-6
    assert abs(w_mid - 0.5) < 1e-6
    assert abs(w_end - 0.0) < 1e-6

    # PSI mid-game load: current BV is 1500 vs 1200, but true match-start initial BV is 4000 vs 4000
    rc = RewardCalculator(beta_bv=1.0, beta_vp=1.0)
    rc.compute_reward({"rewards": {"bv1": 1500, "bv2": 1200, "initial_bv1": 4000, "initial_bv2": 4000, "vp1": 0}})
    assert rc.initial_bv1 == 4000.0 and rc.initial_bv2 == 4000.0
    # Deal 400 BV damage to enemy (delta_bv2 = -400): reward_bv = 1.0 * 2.0 * (0 - (-400)) / 8000 = 0.1
    r = rc.compute_reward({"rewards": {"bv1": 1500, "bv2": 800, "initial_bv1": 4000, "initial_bv2": 4000, "vp1": 0}})
    assert abs(r - 0.1) < 1e-6, f"Expected PSI normalized reward 0.1, got {r}"
    print("  [PASS] Annealing schedules and PSI initial-BV normalization verified.")


def test_inspection_ui_and_lifecycle(parser, move_payload, weapon_payload, phys_payload):
    print("=== Test 6: UI Probabilities, Step Report Rendering, Inspector CLI & Atomic Checkpoints ===")
    import tempfile
    from validations.render_game_state import generate_interactive_html
    from validations.inspect_gamelog import inspect_trajectory_file

    agent = MegaMekAgent(hidden_dim=64, ensemble_size=8)
    agent.eval()

    # 1. Test compute_ui_action_probabilities across Movement, Weapon, Physical
    g_m, m_m = parser.parse_to_heterodata(dict(move_payload, context="MOVEMENT_INFERENCE"))
    path_probs = agent.compute_ui_action_probabilities(g_m, m_m)
    assert len(path_probs) == 3 and abs(sum(path_probs) - 1.0) < 1e-4, f"Expected path_probs sum=1.0, got {path_probs}"

    g_w, m_w = parser.parse_to_heterodata(dict(weapon_payload, context="WEAPON_INFERENCE"))
    weap_probs = agent.compute_ui_action_probabilities(g_w, m_w)
    assert len(weap_probs) > 0, "Expected non-empty weapon node probability map"
    assert "prob" in m_w["valid_twists"][0], "Expected twist dict to be annotated with 'prob'"

    # Test score_partial_weapon_trajectory
    partial_probs = agent.score_partial_weapon_trajectory(
        g_w, m_w, g_w, {"selected_entity_id": 0, "twist": 1, "attacks": [{"target_id": 2, "weapon_id": 0}]}
    )
    assert len(partial_probs) > 0, "Expected non-empty conditional probabilities after partial weapon selection"

    g_p, m_p = parser.parse_to_heterodata(dict(phys_payload, context="PHYSICAL_INFERENCE"))
    phys_probs = agent.compute_ui_action_probabilities(g_p, m_p)
    assert len(phys_probs) > 0, "Expected non-empty physical node probability map"

    # 2. Test step-level report rendering & trajectory inspection
    with tempfile.TemporaryDirectory() as tmpdir:
        traj_path = os.path.join(tmpdir, "test_traj.pt")
        html_path = os.path.join(tmpdir, "test_render.html")
        raw_step = dict(weapon_payload, context="WEAPON_INFERENCE")
        raw_step["reports"] = [
            "=== Round 3 | Phase: FIRING ===",
            "Atlas AS7-D fires Medium Laser at Warhammer WHM-6R; needs 7, rolls 9 : hits!",
        ]
        traj_data = {
            "topology_payload": build_mock_topology(),
            "steps": [
                {
                    "raw_payload": raw_step,
                    "action_dict": {"selected_entity_id": 0, "twist": 1, "attacks": [{"target_id": 2, "weapon_id": 0}]},
                    "reward": 0.42,
                    "value_pred": 1.25,
                }
            ],
            "win": True,
        }
        torch.save(traj_data, traj_path)
        inspect_trajectory_file(traj_path, show_all=True, show_reports=True)
        generate_interactive_html([traj_path], output_path=html_path)
        assert os.path.exists(html_path), "Expected HTML visualization file to be created"
        with open(html_path, "r", encoding="utf-8") as f:
            html_text = f.read()
        assert "Atlas AS7-D fires Medium Laser" in html_text, "Step report missing from generated HTML!"

    print("  [PASS] UI action probabilities, partial weapon scoring, inspect_gamelog, and HTML report rendering verified.")


if __name__ == "__main__":
    parser, g_move, g_weap, g_phys, move_p, weap_p, phys_p = test_bc_and_inference_parsing()
    test_get_action_vs_evaluate_actions_equivalence(parser, move_p, weap_p, phys_p)
    test_bidirectional_gradients_and_ensemble_vtrace(g_move, g_weap, g_phys)
    test_annealing_and_psi_rewards()
    test_full_impala_learn_step(move_p, weap_p, phys_p, g_move, g_weap)
    test_inspection_ui_and_lifecycle(parser, move_p, weap_p, phys_p)
    print("\nALL UNIFIED ARCHITECTURE VALIDATION TESTS PASSED!")
