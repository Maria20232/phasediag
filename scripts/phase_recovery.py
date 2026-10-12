#!/usr/bin/env python3

import argparse
import glob
import os
import re

import h5py
import numpy as np
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from eval_bc import BCPolicy, flatten_obs


import os as _os
DATASET = _os.path.expanduser("~/robosuite_work/demos_lift/low_dim.hdf5")

# Default checkpoint glob patterns. Each match is expected to contain
# "seed<N>" in its filename, e.g. bc_50_seed42.pt, bc_500_seed43.pt
BC50_GLOB = _os.path.expanduser("~/robosuite_work/demos_lift/bc_50_seed*.pt")
BC500_GLOB = _os.path.expanduser("~/robosuite_work/demos_lift/bc_500_seed*.pt")

CONTROL_FREQ = 20
HORIZON = 400

PREGRASP_DISTANCE = 0.05
EARLY_LIFT_DELTA = 0.01
SUCCESS_HEIGHT = 0.04

STABLE_GRASP_FRAMES = 3


def create_env():
    controller_config = load_composite_controller_config(
        controller=None,
        robot="Panda",
    )
    return suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=CONTROL_FREQ,
        horizon=HORIZON,
        ignore_done=True,
    )


def load_policy(path):
    policy = BCPolicy()
    policy.load_state_dict(
        torch.load(
            path,
            map_location="cpu",
        )
    )
    policy.eval()
    return policy


def get_state_info(env, state):
    env.sim.set_state_from_flattened(state)
    env.sim.forward()

    distance = env._gripper_to_target(
        gripper=env.robots[0].gripper,
        target=env.cube.root_body,
        target_type="body",
        return_distance=True,
    )
    grasped = env._check_grasp(
        gripper=env.robots[0].gripper,
        object_geoms=env.cube,
    )
    cube_z = float(env.sim.data.body_xpos[env.cube_body_id][2])
    table_z = float(env.model.mujoco_arena.table_offset[2])
    successful = cube_z > (table_z + SUCCESS_HEIGHT)

    return {
        "distance": float(distance),
        "grasped": bool(grasped),
        "cube_z": cube_z,
        "successful": bool(successful),
    }


def first_stable_grasp(infos):
    n = len(infos)
    for i in range(n - STABLE_GRASP_FRAMES + 1):
        stable = all(infos[j]["grasped"] for j in range(i, i + STABLE_GRASP_FRAMES))
        if stable:
            return i
    return None


def find_phase_indices(env, states):
    infos = []
    for state in states:
        infos.append(get_state_info(env, state))

    n = len(states)

    first_near = None
    for i, info in enumerate(infos):
        if info["distance"] <= PREGRASP_DISTANCE:
            first_near = i
            break

    stable_grasp = first_stable_grasp(infos)

    first_success = None
    for i, info in enumerate(infos):
        if info["successful"]:
            first_success = i
            break

    if first_near is not None:
        approach = max(0, first_near // 2)
    elif stable_grasp is not None:
        approach = max(0, stable_grasp // 2)
    else:
        approach = n // 4

    if stable_grasp is not None and stable_grasp > 0:
        pregrasp = stable_grasp - 1
    elif first_near is not None:
        pregrasp = first_near
    else:
        pregrasp = None

    grasp = stable_grasp

    post_grasp = None
    if (stable_grasp is not None and first_success is not None
            and first_success > stable_grasp + 1):
        post_grasp = stable_grasp + (first_success - stable_grasp) // 2

    if first_success is not None and first_success > 0:
        pre_success = first_success - 1
    else:
        pre_success = None

    phases = {
        "Approach": approach,
        "Pre-grasp": pregrasp,
        "Grasp": grasp,
        "Post-grasp": post_grasp,
        "Pre-success": pre_success,
    }
    return phases, infos


def restore_state(env, state):
    env.reset()
    env.sim.set_state_from_flattened(state)
    env.sim.forward()

    try:
        env.robots[0].controller.update(force=True)
    except AttributeError:
        env.robots[0].composite_controller.part_controllers["right"].update(force=True)

    obs = env._get_observations(force_update=True)
    return obs


def rollout_policy(env, policy, state):
    obs = restore_state(env, state)

    initially_successful = bool(env._check_success())
    if initially_successful:
        return {
            "valid": False,
            "success": False,
            "steps": 0,
            "reason": "initial_state_already_successful",
        }

    for step in range(HORIZON):
        state_vec = flatten_obs(obs)
        with torch.no_grad():
            action = policy(torch.FloatTensor(state_vec)).detach().numpy()

        obs, reward, done, info = env.step(action)

        if env._check_success():
            return {"valid": True, "success": True, "steps": step + 1, "reason": "policy_success"}

        if (not np.all(np.isfinite(env.sim.data.qpos))
                or not np.all(np.isfinite(env.sim.data.qvel))):
            return {"valid": True, "success": False, "steps": step + 1, "reason": "simulation_unstable"}

    return {"valid": True, "success": False, "steps": HORIZON, "reason": "horizon"}


def evaluate_policy(env, policy, policy_name, demos):
    phase_names = ["Approach", "Pre-grasp", "Grasp", "Post-grasp", "Pre-success"]
    results = {
        phase: {"successes": 0, "attempts": 0, "invalid": 0, "success_steps": []}
        for phase in phase_names
    }

    print()
    print("=" * 70)
    print(f"EVALUATING {policy_name}")
    print("=" * 70)

    for demo_name, demo_data in demos.items():
        states = demo_data["states"]
        phases = demo_data["phases"]

        print(f"\n{demo_name}")

        for phase in phase_names:
            idx = phases.get(phase)
            if idx is None:
                print(f"  {phase:<12}: SKIPPED - phase unavailable")
                continue

            result = rollout_policy(env, policy, states[idx])

            if not result["valid"]:
                results[phase]["invalid"] += 1
                print(f"  {phase:<12}: INVALID ({result['reason']})")
                continue

            results[phase]["attempts"] += 1

            if result["success"]:
                results[phase]["successes"] += 1
                results[phase]["success_steps"].append(result["steps"])
                print(f"  {phase:<12}: SUCCESS ({result['steps']} steps)")
            else:
                print(f"  {phase:<12}: FAIL ({result['reason']})")

    return results


def print_policy_results(policy_name, results):
    print()
    print("=" * 70)
    print(f"{policy_name} PHASE RESULTS")
    print("=" * 70)

    for phase, data in results.items():
        attempts = data["attempts"]
        successes = data["successes"]
        rate = successes / attempts if attempts > 0 else float("nan")
        mean_steps = np.mean(data["success_steps"]) if data["success_steps"] else float("nan")
        print(f"{phase:<12} | {successes:2d}/{attempts:2d} | {rate*100:6.1f}% | "
              f"mean success steps={mean_steps:.1f}")


def print_comparison(results_50, results_500):
    phase_names = ["Approach", "Pre-grasp", "Grasp", "Post-grasp", "Pre-success"]
    print()
    print("=" * 70)
    print("SEMANTIC PHASE CONTINUATION RESULTS")
    print("=" * 70)
    print(f"{'Phase':<14}{'BC-50':<18}{'BC-500':<18}")
    print("-" * 50)

    for phase in phase_names:
        d50 = results_50[phase]
        d500 = results_500[phase]

        if d50["attempts"] > 0:
            rate50 = d50["successes"] / d50["attempts"]
            text50 = f"{d50['successes']}/{d50['attempts']} ({rate50*100:.1f}%)"
        else:
            text50 = "N/A"

        if d500["attempts"] > 0:
            rate500 = d500["successes"] / d500["attempts"]
            text500 = f"{d500['successes']}/{d500['attempts']} ({rate500*100:.1f}%)"
        else:
            text500 = "N/A"

        print(f"{phase:<14}{text50:<18}{text500:<18}")

    print("=" * 70)


def find_seeded_checkpoints(pattern):
    """Return {seed:int -> path} for every checkpoint matching pattern
    whose filename contains 'seed<N>'."""
    found = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"seed(\d+)", os.path.basename(path))
        if m:
            found[int(m.group(1))] = path
    return found


def phase_success_rate(results, phase):
    d = results[phase]
    if d["attempts"] == 0:
        return float("nan")
    return 100.0 * d["successes"] / d["attempts"]


def print_multi_seed_summary(label, per_seed_results, phase_names):
    """Prints a Table-IV-style summary: mean success rate per phase plus
    the individual per-seed rates, matching the paper's reporting format."""
    print()
    print("=" * 78)
    print(f"{label} PHASE-CONDITIONED SUCCESS ACROSS SEEDS")
    print("=" * 78)
    seeds = sorted(per_seed_results.keys())
    print(f"Seeds used: {seeds}")
    print(f"{'Phase':<14}{'Mean %':<10}Per-seed (%)")
    print("-" * 78)
    for phase in phase_names:
        rates = [phase_success_rate(per_seed_results[s], phase) for s in seeds]
        valid_rates = [r for r in rates if not np.isnan(r)]
        mean_rate = np.mean(valid_rates) if valid_rates else float("nan")
        per_seed_str = ", ".join(
            f"{r:.1f}" if not np.isnan(r) else "N/A" for r in rates
        )
        print(f"{phase:<14}{mean_rate:<10.1f}({per_seed_str})")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bc50_glob", type=str, default=BC50_GLOB,
        help="Glob pattern for BC-50 checkpoints, each containing 'seed<N>' in its filename.",
    )
    parser.add_argument(
        "--bc500_glob", type=str, default=BC500_GLOB,
        help="Glob pattern for BC-500 checkpoints, each containing 'seed<N>' in its filename.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="*", default=None,
        help="Optional subset of seeds to evaluate (default: evaluate every matched checkpoint).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("DiagVLA SEMANTIC PHASE CONTINUATION ANALYSIS")
    print("=" * 70)

    bc50_ckpts = find_seeded_checkpoints(args.bc50_glob)
    bc500_ckpts = find_seeded_checkpoints(args.bc500_glob)

    if args.seeds is not None:
        bc50_ckpts = {s: p for s, p in bc50_ckpts.items() if s in args.seeds}
        bc500_ckpts = {s: p for s, p in bc500_ckpts.items() if s in args.seeds}

    if not bc50_ckpts:
        raise FileNotFoundError(f"No BC-50 checkpoints matched {args.bc50_glob!r}")
    if not bc500_ckpts:
        raise FileNotFoundError(f"No BC-500 checkpoints matched {args.bc500_glob!r}")

    print(f"\nFound BC-50 seeds:  {sorted(bc50_ckpts.keys())}")
    print(f"Found BC-500 seeds: {sorted(bc500_ckpts.keys())}")

    print("\nCreating Lift environment...")
    env = create_env()
    env.reset()

    demos = {}
    print("\nDetecting semantic phases...")
    with h5py.File(DATASET, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        for demo_name in demo_keys:
            states = f[f"data/{demo_name}/states"][:]
            phases, infos = find_phase_indices(env, states)
            demos[demo_name] = {"states": states, "phases": phases}
            phase_text = ", ".join(f"{name}={idx}" for name, idx in phases.items())
            print(f"  {demo_name}: {phase_text}")

    phase_names = ["Approach", "Pre-grasp", "Grasp", "Post-grasp", "Pre-success"]

    results_50_by_seed = {}
    for seed, path in sorted(bc50_ckpts.items()):
        policy = load_policy(path)
        results_50_by_seed[seed] = evaluate_policy(env, policy, f"BC-50 (seed {seed})", demos)
        print_policy_results(f"BC-50 (seed {seed})", results_50_by_seed[seed])

    results_500_by_seed = {}
    for seed, path in sorted(bc500_ckpts.items()):
        policy = load_policy(path)
        results_500_by_seed[seed] = evaluate_policy(env, policy, f"BC-500 (seed {seed})", demos)
        print_policy_results(f"BC-500 (seed {seed})", results_500_by_seed[seed])

    print_multi_seed_summary("BC-50", results_50_by_seed, phase_names)
    print_multi_seed_summary("BC-500", results_500_by_seed, phase_names)

    # Legacy single-seed-pair comparison view, using the lowest available seed
    # from each policy, kept for continuity with earlier console output.
    lowest_50 = min(results_50_by_seed)
    lowest_500 = min(results_500_by_seed)
    print_comparison(results_50_by_seed[lowest_50], results_500_by_seed[lowest_500])

    env.close()


if __name__ == "__main__":
    main()
