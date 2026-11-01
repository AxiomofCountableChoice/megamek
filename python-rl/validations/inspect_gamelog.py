import argparse
import glob
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def find_latest_trajectories(limit=1):
    search_patterns = [
        "data/rl_princess_trajectories/**/*.pt",
        "data/rl_selfplay_trajectories/**/*.pt",
        "data/bc_trajectories/*.pt",
    ]
    all_files = []
    for pat in search_patterns:
        all_files.extend(glob.glob(pat, recursive=True))
    all_files = [f for f in all_files if not f.endswith(".consumed") and not f.endswith(".tmp")]
    all_files.sort(key=os.path.getmtime, reverse=True)
    return all_files[:limit]


def format_action_summary(context, action_dict, entities_meta):
    if not isinstance(action_dict, dict):
        return str(action_dict)

    def ent_label(idx):
        if idx is None or idx < 0:
            return f"Unit({idx})"
        if isinstance(entities_meta, list) and 0 <= idx < len(entities_meta):
            name = entities_meta[idx].get("short_name") or entities_meta[idx].get("name") or f"Unit#{idx}"
            return f"{idx}:{name}"
        return f"Unit#{idx}"

    u_idx = action_dict.get("selected_entity_id", action_dict.get("selected_entity_index", -1))
    if context.startswith("MOVEMENT"):
        p_idx = action_dict.get("selected_path_index", -1)
        return f"{ent_label(u_idx)} -> Path#{p_idx}"
    elif context.startswith("WEAPON"):
        twist = action_dict.get("twist", action_dict.get("torso_twist", 0))
        attacks = action_dict.get("attacks", [])
        return f"{ent_label(u_idx)} | Twist:{twist} | Weapons:{len(attacks)}"
    elif context.startswith("PHYSICAL"):
        att = action_dict.get("attack")
        if att:
            return f"{ent_label(u_idx)} -> Target#{att.get('target_id')} (Type {att.get('action_type')})"
        return f"{ent_label(u_idx)} | No Physical Attack"
    return str(action_dict)


def inspect_trajectory_file(traj_path, show_all=False, target_step=None, show_reports=True):
    if not os.path.exists(traj_path):
        print(f"[Error] File not found: {traj_path}")
        return

    data = torch.load(traj_path, weights_only=False, map_location="cpu")
    steps = data.get("steps") or data.get("trajectory") or []
    topology = data.get("topology_payload") or data.get("topology") or {}
    board_w = topology.get("width", "?")
    board_h = topology.get("height", "?")

    print("=" * 95)
    print(f"Trajectory File : {traj_path}")
    print(f"Total Steps     : {len(steps)} | Board Size: {board_w}x{board_h}")
    print("=" * 95)

    total_reward = 0.0
    total_reports = 0

    for i, step in enumerate(steps):
        if isinstance(step, dict) and "raw_payload" in step:
            raw = step.get("raw_payload", {})
            reward = float(step.get("reward", 0.0))
            val_pred = float(step.get("value_pred", 0.0))
            action_dict = step.get("action_dict", {})
        else:
            raw = step if isinstance(step, dict) else {}
            rew_field = raw.get("rewards", 0.0)
            reward = float(sum(rew_field.values())) if isinstance(rew_field, dict) else float(raw.get("reward", 0.0))
            val_pred = 0.0
            action_dict = raw.get("target_action") or raw.get("mask", {}).get("target_action") or {}

        total_reward += reward
        state = raw.get("state", {})
        context = raw.get("context", "UNKNOWN")
        phase = state.get("phase_main", raw.get("phase", "UNKNOWN"))
        rnd = state.get("round_number", -1)
        turn = state.get("turn_number", raw.get("turn_number", -1))
        entities = state.get("entities", raw.get("entities", []))
        entities_meta = state.get("entities_meta", [])
        reports = raw.get("reports", [])
        total_reports += len(reports)

        should_print = (
            show_all
            or (target_step is not None and i == target_step)
            or (target_step is None and (i < 10 or i >= len(steps) - 10))
        )

        if should_print:
            act_str = format_action_summary(context, action_dict, entities_meta)
            print(
                f"Step {i:3d} | Rnd {rnd:2d} Turn {turn:2d} | {context:18s} | Units:{len(entities):2d} "
                f"| Rew:{reward:+7.3f} (Sum:{total_reward:+7.3f}) | V:{val_pred:+6.3f} | {act_str}"
            )
            if show_reports and reports:
                for r_line in reports:
                    print(f"         [Report] {r_line}")
        elif target_step is None and i == 10:
            print(f"   ... ({len(steps) - 20} intermediate steps omitted; pass --all to view every step) ...")

    print("-" * 95)
    print(f"Cumulative Return: {total_reward:+.4f} | Total Captured Report Lines: {total_reports}")
    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Inspect MegaMek RL / BC Trajectory (.pt) files and combat reports.")
    parser.add_argument("paths", nargs="*", help="Path(s) to .pt trajectory file(s) or directory. Defaults to latest.")
    parser.add_argument("--all", action="store_true", help="Print all steps instead of first/last 10.")
    parser.add_argument("--step", type=int, default=None, help="Inspect a specific step index in detail.")
    parser.add_argument("--no-reports", action="store_true", help="Hide step-by-step MegaMek game reports.")
    parser.add_argument("--render", type=str, default=None, help="Also render interactive HTML to the specified file (e.g. render.html).")
    args = parser.parse_args()

    target_files = []
    if not args.paths:
        target_files = find_latest_trajectories(limit=1)
        if not target_files:
            print("No trajectory (.pt) files found in data/. Provide a file path explicitly.")
            return
    else:
        for p in args.paths:
            if os.path.isdir(p):
                found = glob.glob(os.path.join(p, "**", "*.pt"), recursive=True)
                found = [f for f in found if not f.endswith(".consumed") and not f.endswith(".tmp")]
                found.sort(key=os.path.getctime)
                target_files.extend(found)
            else:
                target_files.append(p)

    for tf in target_files:
        inspect_trajectory_file(
            tf,
            show_all=args.all,
            target_step=args.step,
            show_reports=not args.no_reports,
        )

    if args.render and target_files:
        from validations.render_game_state import generate_interactive_html

        generate_interactive_html(target_files, output_path=args.render)


if __name__ == "__main__":
    main()
