#!/usr/bin/env python3
import argparse
import numpy as np
import h5py
import torch
import torch.nn as nn
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object-state"]
LIFT_HEIGHT_THRESH = 0.04
PLACEMENT_XY_THRESH = 0.05

class BCPolicy(nn.Module):
    def __init__(self, state_dim=32, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    def forward(self, x):
        return self.net(x)

def flatten_obs(obs_dict, obs_keys=OBS_KEYS):
    return np.concatenate([np.asarray(obs_dict[k]).flatten() for k in obs_keys], axis=-1)

def create_env():
    controller_config = load_composite_controller_config(controller=None, robot="Panda")
    return suite.make(
        env_name="Stack", robots="Panda", controller_configs=controller_config,
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        reward_shaping=True, control_freq=20, ignore_done=True,
    )

def load_policy(path):
    policy = BCPolicy()
    policy.load_state_dict(torch.load(path, map_location="cpu"))
    policy.eval()
    return policy

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
    phases = {"Approach": int(0.05 * T)}
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

    if grasp_idx is not None:
        phases["Grasp-transport"] = grasp_idx
        placement_idx = None
        for t in range(grasp_idx, T):
            obs = set_robot_state(env, states[t])
            cubeA_xy = obs["cubeA_pos"][:2]
            cubeB_xy = obs["cubeB_pos"][:2]
            horiz_dist = np.linalg.norm(cubeA_xy - cubeB_xy)
            still_grasping = env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cubeA)
            already_stacked = env._check_success()
            if horiz_dist < PLACEMENT_XY_THRESH and still_grasping and not already_stacked:
                placement_idx = t
                break
        if placement_idx is not None:
            phases["Near-placement"] = placement_idx

    return phases

def rollout_from_state(env, policy, state, max_steps=500):
    obs = set_robot_state(env, state)
    for step in range(max_steps):
        state_vec = flatten_obs(obs)
        with torch.no_grad():
            action = policy(torch.FloatTensor(state_vec).unsqueeze(0)).squeeze(0).numpy()
        action = np.clip(action, -1.0, 1.0)
        obs, reward, done, info = env.step(action)
        if env._check_success():
            return True, step + 1
    return False, max_steps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/maria/robosuite_work/demos_stack/bc_stack_500.pt")
    parser.add_argument("--dataset", type=str, default="/home/maria/robosuite_work/demos_stack/1786611107_4237583/demo.hdf5")
    parser.add_argument("--max_demos", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    print("Loading policy...")
    policy = load_policy(args.model)
    print("Creating Stack environment...")
    env = create_env()
    env.reset()

    phase_names = ["Approach", "Grasp-transport", "Near-placement"]
    results = {p: {"successes": 0, "attempts": 0, "skipped": 0} for p in phase_names}

    print(f"\nLoading demos from {args.dataset} ...")
    with h5py.File(args.dataset, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]),
        )[: args.max_demos]

        for demo_name in demo_keys:
            states = f[f"data/{demo_name}/states"][:]
            print(f"\n{demo_name}: detecting phases...")
            phases = find_stack_phase_indices(env, states)
            print(f"  found: {phases}")
            for phase_name in phase_names:
                idx = phases.get(phase_name)
                if idx is None:
                    results[phase_name]["skipped"] += 1
                    print(f"  {phase_name:<16}: SKIPPED - phase unavailable")
                    continue
                success, steps = rollout_from_state(env, policy, states[idx], args.max_steps)
                results[phase_name]["attempts"] += 1
                if success:
                    results[phase_name]["successes"] += 1
                print(f"  {phase_name:<16}: {'SUCCESS' if success else 'FAIL'} ({steps} steps)")

    env.close()
    print("\n" + "=" * 60)
    print("STACK PHASE-CONDITIONED RESULTS (compact, single model)")
    print("=" * 60)
    for phase_name in phase_names:
        r = results[phase_name]
        if r["attempts"] > 0:
            rate = r["successes"] / r["attempts"] * 100
            print(f"{phase_name:<16} | {r['successes']}/{r['attempts']} | {rate:.1f}% (skipped: {r['skipped']})")
        else:
            print(f"{phase_name:<16} | no valid attempts (skipped: {r['skipped']})")

if __name__ == "__main__":
    main()
