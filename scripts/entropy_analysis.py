#!/usr/bin/env python3
"""
Action-space entropy analysis for PhaseDiag (Lift task).

Addresses the reviewer suggestion: "look at the action-space entropy at the
points where the policy fails. Does the policy get 'confused' (high entropy)
at the specific phases where success rates drop?"

BCPolicy is a deterministic MLP -- it has no explicit output distribution to
compute entropy from directly. We instead use MC-Dropout: the policy's
Dropout(p=0.1) layers are normally disabled at evaluation time (required for
deterministic rollouts, see paper Section III-D / III-I). Here we
deliberately re-enable them and draw N stochastic forward passes per input.
The spread of predicted actions across those passes is used as an
approximate-uncertainty signal: treating each action dimension as
independently Gaussian, we estimate a per-dimension differential entropy
from the sample variance and sum across dimensions.

This is a diagnostic proxy for "how confused is the policy here", not a
claim that BCPolicy is a genuinely calibrated probabilistic model.

Usage:
    python entropy_analysis.py
    python entropy_analysis.py --seeds 42 43 44
    python entropy_analysis.py --n_samples 50
"""

import argparse

import h5py
import numpy as np
import torch

from eval_bc import flatten_obs
from phase_recovery import (
    BC50_GLOB,
    BC500_GLOB,
    DATASET,
    create_env,
    find_phase_indices,
    find_seeded_checkpoints,
    load_policy,
    restore_state,
)

PHASE_NAMES = ["Approach", "Pre-grasp", "Grasp", "Post-grasp", "Pre-success"]


def mc_dropout_entropy(policy, obs_vec, n_samples=30):
    """Estimate action-space differential entropy via MC-Dropout.

    Returns (total_entropy, per_dim_std) where total_entropy sums
    per-dimension estimated differential entropy assuming a diagonal
    Gaussian fit to the MC samples, and per_dim_std is the raw per-dimension
    standard deviation across samples (useful for sanity-checking which
    action dimensions actually drive the effect, e.g. gripper vs. position).
    """
    policy.train()  # re-enable dropout for stochastic forward passes
    obs_tensor = torch.FloatTensor(obs_vec)
    with torch.no_grad():
        samples = torch.stack([policy(obs_tensor) for _ in range(n_samples)])
    policy.eval()  # restore deterministic mode for any other use of policy

    samples_np = samples.numpy()  # shape (n_samples, action_dim)
    variance = samples_np.var(axis=0) + 1e-8  # avoid log(0)
    std = np.sqrt(variance)

    # Differential entropy of an independent Gaussian per action dimension:
    # H = 0.5 * log(2 * pi * e * sigma^2), summed across dimensions.
    per_dim_entropy = 0.5 * np.log(2 * np.pi * np.e * variance)
    total_entropy = float(per_dim_entropy.sum())
    return total_entropy, std


def collect_phase_entropies(env, policy, demos, n_samples):
    """Returns {phase_name: [entropy_value, ...]} across all demos where
    that phase was identified."""
    results = {phase: [] for phase in PHASE_NAMES}

    for demo_name, demo_data in demos.items():
        states = demo_data["states"]
        phases = demo_data["phases"]

        for phase in PHASE_NAMES:
            idx = phases.get(phase)
            if idx is None:
                continue

            obs = restore_state(env, states[idx])
            obs_vec = flatten_obs(obs)
            entropy, _ = mc_dropout_entropy(policy, obs_vec, n_samples=n_samples)
            results[phase].append(entropy)

    return results


def print_entropy_vs_success(label, entropy_by_phase, success_rate_by_phase=None):
    print()
    print("=" * 78)
    print(f"{label}: MEAN ACTION-SPACE ENTROPY BY PHASE (MC-Dropout proxy)")
    print("=" * 78)
    header = f"{'Phase':<14}{'Mean Entropy':<16}{'Std':<10}{'N':<6}"
    if success_rate_by_phase is not None:
        header += "Success %"
    print(header)
    print("-" * 78)
    for phase in PHASE_NAMES:
        vals = entropy_by_phase.get(phase, [])
        if not vals:
            print(f"{phase:<14}{'N/A':<16}{'N/A':<10}{0:<6}")
            continue
        mean_e = np.mean(vals)
        std_e = np.std(vals)
        row = f"{phase:<14}{mean_e:<16.3f}{std_e:<10.3f}{len(vals):<6}"
        if success_rate_by_phase is not None:
            row += f"{success_rate_by_phase.get(phase, float('nan')):.1f}"
        print(row)
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc50_glob", type=str, default=BC50_GLOB)
    parser.add_argument("--bc500_glob", type=str, default=BC500_GLOB)
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                         help="Optional subset of seeds to analyze.")
    parser.add_argument("--n_samples", type=int, default=30,
                         help="Number of MC-Dropout forward passes per state.")
    args = parser.parse_args()

    bc50_ckpts = find_seeded_checkpoints(args.bc50_glob)
    bc500_ckpts = find_seeded_checkpoints(args.bc500_glob)
    if args.seeds is not None:
        bc50_ckpts = {s: p for s, p in bc50_ckpts.items() if s in args.seeds}
        bc500_ckpts = {s: p for s, p in bc500_ckpts.items() if s in args.seeds}

    if not bc50_ckpts or not bc500_ckpts:
        raise FileNotFoundError("No matching checkpoints found for one or both policies.")

    print(f"BC-50 seeds:  {sorted(bc50_ckpts.keys())}")
    print(f"BC-500 seeds: {sorted(bc500_ckpts.keys())}")
    print(f"MC-Dropout samples per state: {args.n_samples}")

    print("\nCreating Lift environment...")
    env = create_env()
    env.reset()

    demos = {}
    print("Detecting semantic phases (reusing phase_recovery logic)...")
    with h5py.File(DATASET, "r") as f:
        demo_keys = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda x: int(x.split("_")[-1]),
        )
        for demo_name in demo_keys:
            states = f[f"data/{demo_name}/states"][:]
            phases, _ = find_phase_indices(env, states)
            demos[demo_name] = {"states": states, "phases": phases}

    # BC-50: aggregate entropy across all seeds
    all_bc50_entropy = {phase: [] for phase in PHASE_NAMES}
    for seed, path in sorted(bc50_ckpts.items()):
        print(f"\nComputing entropy for BC-50 seed {seed}...")
        policy = load_policy(path)
        per_seed = collect_phase_entropies(env, policy, demos, args.n_samples)
        for phase in PHASE_NAMES:
            all_bc50_entropy[phase].extend(per_seed[phase])

    all_bc500_entropy = {phase: [] for phase in PHASE_NAMES}
    for seed, path in sorted(bc500_ckpts.items()):
        print(f"Computing entropy for BC-500 seed {seed}...")
        policy = load_policy(path)
        per_seed = collect_phase_entropies(env, policy, demos, args.n_samples)
        for phase in PHASE_NAMES:
            all_bc500_entropy[phase].extend(per_seed[phase])

    print_entropy_vs_success("BC-50", all_bc50_entropy)
    print_entropy_vs_success("BC-500", all_bc500_entropy)

    print("\nNOTE: cross-reference these mean-entropy-per-phase numbers against")
    print("phase_recovery.py's success-rate-per-phase output (Table IV) by eye,")
    print("or pass the printed success rates into --success_rates if you extend")
    print("this script. A rising entropy trend as phase success drops would")
    print("support the 'confusion' hypothesis; a flat or inverted trend would not.")

    env.close()


if __name__ == "__main__":
    main()
