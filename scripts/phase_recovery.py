#!/usr/bin/env python3

import h5py
import numpy as np
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

from eval_bc import BCPolicy, flatten_obs


DATASET = "/home/maria/robosuite_work/demos_lift/low_dim.hdf5"
BC50_PATH = "/home/maria/robosuite_work/demos_lift/bc_50.pt"
BC500_PATH = "/home/maria/robosuite_work/demos_lift/bc_500.pt"

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


def main():
    print("=" * 70)
    print("DiagVLA SEMANTIC PHASE CONTINUATION ANALYSIS")
    print("=" * 70)

    print("\nLoading policies...")
    bc50 = load_policy(BC50_PATH)
    bc500 = load_policy(BC500_PATH)
    print("Policies loaded.")

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

    results_50 = evaluate_policy(env, bc50, "BC-50", demos)
    results_500 = evaluate_policy(env, bc500, "BC-500", demos)

    print_policy_results("BC-50", results_50)
    print_policy_results("BC-500", results_500)
    print_comparison(results_50, results_500)

    env.close()


if __name__ == "__main__":
    main()
