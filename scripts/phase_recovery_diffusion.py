#!/usr/bin/env python3
"""
Phase-conditioned evaluation for the Diffusion Policy on Lift, mirroring the
protocol used for BC-50/BC-500 (Section III-H / phase_recovery.py) so the
results are directly comparable across architectures.

Reuses phase_recovery.py's environment setup, phase-detection, state
restoration, and rollout logic unchanged -- the only new piece is loading a
DiffusionBCPolicy checkpoint instead of a BCPolicy one. Because
DiffusionBCPolicy.forward(state) returns a single sampled action just like
BCPolicy.forward(state) does, phase_recovery.py's rollout_policy() and
evaluate_policy() work on it without modification.

Note: each action from the Diffusion Policy requires a full reverse-diffusion
sample (num_timesteps forward passes, default 20), so rollouts are slower
per-step than the plain MLP. Expect longer wall-clock time than the
equivalent BC-50/BC-500 run.
"""

import argparse
import glob
import os
import re

import h5py
import torch

from diffusion_policy import DiffusionBCPolicy
from phase_recovery import (
    DATASET,
    create_env,
    evaluate_policy,
    find_phase_indices,
    print_policy_results,
)

DIFFUSION_GLOB = os.path.expanduser("~/robosuite_work/demos_lift/diffusion_*_seed*.pt")


def load_diffusion_policy(path):
    checkpoint = torch.load(path, map_location="cpu")
    policy = DiffusionBCPolicy(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        num_timesteps=checkpoint["num_timesteps"],
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def find_seeded_checkpoints(pattern):
    found = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"seed(\d+)", os.path.basename(path))
        if m:
            found[int(m.group(1))] = path
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", type=str, default=DIFFUSION_GLOB,
                         help="Glob pattern for Diffusion Policy checkpoints.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    ckpts = find_seeded_checkpoints(args.glob)
    if args.seeds is not None:
        ckpts = {s: p for s, p in ckpts.items() if s in args.seeds}
    if not ckpts:
        raise FileNotFoundError(f"No Diffusion Policy checkpoints matched {args.glob!r}")

    print(f"Found Diffusion Policy seeds: {sorted(ckpts.keys())}")

    print("\nCreating Lift environment...")
    env = create_env()
    env.reset()

    demos = {}
    print("\nDetecting semantic phases (same protocol as BC-50/BC-500)...")
    with h5py.File(DATASET, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        for demo_name in demo_keys:
            states = f[f"data/{demo_name}/states"][:]
            phases, _ = find_phase_indices(env, states)
            demos[demo_name] = {"states": states, "phases": phases}
            phase_text = ", ".join(f"{name}={idx}" for name, idx in phases.items())
            print(f"  {demo_name}: {phase_text}")

    for seed, path in sorted(ckpts.items()):
        print(f"\nLoading Diffusion Policy (seed {seed}) from {path}...")
        policy = load_diffusion_policy(path)
        results = evaluate_policy(env, policy, f"Diffusion Policy (seed {seed})", demos)
        print_policy_results(f"Diffusion Policy (seed {seed})", results)

    env.close()


if __name__ == "__main__":
    main()
