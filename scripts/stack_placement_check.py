#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

LIFT_HEIGHT_THRESH = 0.04
PLACEMENT_XY_THRESH = 0.05

def create_env():
    controller_config = load_composite_controller_config(controller=None, robot="Panda")
    return suite.make(
        env_name="Stack", robots="Panda", controller_configs=controller_config,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        reward_shaping=True, control_freq=20, ignore_done=True,
    )

def set_robot_state(env, state):
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    try:
        env.robots[0].controller.update(force=True)
    except AttributeError:
        env.robots[0].composite_controller.part_controllers["right"].update(force=True)
    return env._get_observations(force_update=True)

def find_stack_phase_indices(env, states):
    T = states.shape[0]
    table_height = env.table_offset[2]
    grasp_idx = None
    for t in range(T):
        obs = set_robot_state(env, states[t])
        is_grasping = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA)
        cubeA_height = obs["cubeA_pos"][2]
        cubeA_lifted = cubeA_height > table_height + LIFT_HEIGHT_THRESH
        if is_grasping and cubeA_lifted:
            grasp_idx = t
            break
    if grasp_idx is None:
        return None, None
    for t in range(grasp_idx, T):
        obs = set_robot_state(env, states[t])
        cubeA_xy = obs["cubeA_pos"][:2]
        cubeB_xy = obs["cubeB_pos"][:2]
        horiz_dist = np.linalg.norm(cubeA_xy - cubeB_xy)
        still_grasping = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA)
        already_stacked = env._check_success()
        if horiz_dist < PLACEMENT_XY_THRESH and still_grasping and not already_stacked:
            return grasp_idx, t
    return grasp_idx, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="../data/demos_stack/1786611107_4237583/demo.hdf5")
    parser.add_argument("--max_demos", type=int, default=15)
    args = parser.parse_args()

    env = create_env()
    env.reset()

    with h5py.File(args.dataset, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]),
        )[: args.max_demos]

        print(f"{'demo':<10} {'already_success':<16} {'xy_dist':<10} {'z_gap':<10} {'grasping':<10} {'expert_steps_left':<18}")
        print("-" * 80)

        n_already_success = 0
        for demo_name in demo_keys:
            states = f[f"data/{demo_name}/states"][:]
            T = states.shape[0]
            grasp_idx, placement_idx = find_stack_phase_indices(env, states)
            if placement_idx is None:
                print(f"{demo_name:<10} SKIPPED (no near-placement state found)")
                continue

            obs = set_robot_state(env, states[placement_idx])
            already_success = env._check_success()
            cubeA_xy = obs["cubeA_pos"][:2]
            cubeB_xy = obs["cubeB_pos"][:2]
            xy_dist = np.linalg.norm(cubeA_xy - cubeB_xy)
            z_gap = obs["cubeA_pos"][2] - obs["cubeB_pos"][2]
            still_grasping = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA)

            expert_steps_left = None
            for t in range(placement_idx, T):
                obs_t = set_robot_state(env, states[t])
                if env._check_success():
                    expert_steps_left = t - placement_idx
                    break

            if already_success:
                n_already_success += 1

            print(f"{demo_name:<10} {str(already_success):<16} {xy_dist:<10.4f} {z_gap:<10.4f} {str(still_grasping):<10} {str(expert_steps_left):<18}")

    env.close()
    print("-" * 80)
    print(f"\n{n_already_success}/{len(demo_keys)} Near-placement states were ALREADY at _check_success()==True before any policy action.")
    if n_already_success > 0:
        print("WARNING: some restored states were trivially pre-successful.")
    else:
        print("None pre-successful -- genuine post-restoration competence.")

if __name__ == "__main__":
    main()
