#!/usr/bin/env python3
"""
Expert-state recovery analysis for BC policies on the Lift task.
Samples expert states from 5 trajectory-progress bins (0-20%, 20-40%, ...,
80-100%), restores the simulator to each state, and measures whether the
policy can recover and complete the task from there.
"""
import argparse
import random
import numpy as np
import h5py
import torch
import torch.nn as nn
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object-state"]


class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def flatten_obs(obs_dict, obs_keys=OBS_KEYS):
    return np.concatenate([np.asarray(obs_dict[k]).flatten() for k in obs_keys], axis=-1)


def set_robot_state(env, state):
    """Teleport the simulator to a saved expert state and force the OSC
    controller to resync its cached kinematics (ee_pos, jacobian, etc.),
    avoiding stale-reference actions on the very next env.step()."""
    env.sim.set_state_from_flattened(state)
    env.sim.forward()
    try:
        env.robots[0].controller.update(force=True)
    except AttributeError:
        # composite controller structure differs across robosuite versions;
        # try the nested "right" arm controller as a fallback
        env.robots[0].composite_controller.part_controllers["right"].update(force=True)
    return env._get_observations(force_update=True)


def load_expert_states(dataset_path, num_bins=5, samples_per_bin=15, seed=0):
    """Sample raw expert states from 5 equally-sized trajectory-progress bins,
    pooling across all demos in the dataset."""
    rng = random.Random(seed)
    bins = [[] for _ in range(num_bins)]

    with h5py.File(dataset_path, "r") as f:
        demo_names = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        for demo_name in demo_names:
            states = f["data"][demo_name]["states"][:]
            T = states.shape[0]
            for b in range(num_bins):
                lo = int(b / num_bins * T)
                hi = int((b + 1) / num_bins * T)
                if hi <= lo:
                    continue
                idx = rng.randint(lo, hi - 1)
                bins[b].append(states[idx])

    # subsample down to samples_per_bin per bin if we have more than needed
    for b in range(num_bins):
        if len(bins[b]) > samples_per_bin:
            bins[b] = rng.sample(bins[b], samples_per_bin)

    return bins


def run_recovery_analysis(model_path, dataset_path, num_bins=5, samples_per_bin=15,
                           max_steps=400, seed=0):
    controller_config = load_composite_controller_config(controller=None, robot="Panda")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        ignore_done=True,
    )

    policy = BCPolicy(state_dim=19, action_dim=7)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    print(f"Sampling expert states from {dataset_path} ...")
    bins = load_expert_states(dataset_path, num_bins=num_bins,
                               samples_per_bin=samples_per_bin, seed=seed)

    bin_labels = [f"{int(100*b/num_bins)}-{int(100*(b+1)/num_bins)}%" for b in range(num_bins)]
    results = {}

    for b, label in enumerate(bin_labels):
        states = bins[b]
        n_eligible = len(states)
        n_recovered = 0
        print(f"\n=== Bin {label} ({n_eligible} sampled states) ===")

        for i, state in enumerate(states):
            env.reset()
            obs = set_robot_state(env, state)
            success = False
            for step in range(max_steps):
                state_vec = flatten_obs(obs)
                state_t = torch.FloatTensor(state_vec).unsqueeze(0)
                with torch.no_grad():
                    action = policy(state_t).squeeze(0).numpy()
                action = np.clip(action, -1.0, 1.0)
                obs, reward, done, info = env.step(action)
                if env._check_success():
                    success = True
                    break
            n_recovered += int(success)
            print(f"  state {i+1}/{n_eligible}: {'RECOVERED' if success else 'failed'}")

        rate = n_recovered / n_eligible if n_eligible > 0 else float("nan")
        results[label] = (n_recovered, n_eligible, rate)
        print(f"  -> Bin {label}: {n_recovered}/{n_eligible} = {rate*100:.1f}%")

    env.close()

    print(f"\n{'='*50}")
    print(f"RECOVERY SUMMARY for {model_path}")
    print(f"{'='*50}")
    for label, (n_rec, n_elig, rate) in results.items():
        print(f"  {label}: {n_rec}/{n_elig} = {rate*100:.1f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str,
                         default="/home/maria/robosuite_work/demos_lift/demo_consolidated.hdf5")
    parser.add_argument("--samples_per_bin", type=int, default=15)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_recovery_analysis(args.model, args.dataset,
                           samples_per_bin=args.samples_per_bin,
                           max_steps=args.max_steps, seed=args.seed)
