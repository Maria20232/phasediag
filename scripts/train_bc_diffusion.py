#!/usr/bin/env python3
"""
Train a Diffusion Policy on the same Lift demonstrations used for BC-50/BC-500,
as a secondary architecture to test whether PhaseDiag's diagnostic findings
generalize beyond the feedforward MLP policy (reviewer-requested experiment).

Mirrors train_bc.py's data loading and CLI conventions exactly, so results
are directly comparable: same demonstrations, same train/seed handling, same
checkpoint-naming pattern (seed baked into filename).
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from diffusion_policy import DiffusionBCPolicy
from train_bc import load_demonstrations, OBS_KEYS  # reuse exact data pipeline


def train_diffusion(demo_path, epochs=100, batch_size=64, lr=1e-3,
                     save_path=None, seed=None, num_timesteps=20):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Random seed: {seed}")

    states, actions = load_demonstrations(demo_path, obs_keys=OBS_KEYS)
    print(f"Loaded {len(states)} state-action pairs")
    print(f"State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")

    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    dataloader = DataLoader(TensorDataset(states, actions), batch_size=batch_size, shuffle=True)

    policy = DiffusionBCPolicy(
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        num_timesteps=num_timesteps,
    )
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    print(f"\nTraining Diffusion Policy for {epochs} epochs "
          f"(num_timesteps={num_timesteps})...")
    for epoch in range(epochs):
        total_loss, num_batches = 0, 0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            loss = policy.diffusion_loss(batch_states, batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {total_loss / num_batches:.6f}")

    if save_path is None:
        seed_tag = f"_seed{seed}" if seed is not None else ""
        save_path = os.path.join(
            os.path.dirname(demo_path), f"diffusion_ep{epochs}{seed_tag}.pt"
        )
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "state_dim": states.shape[1],
            "action_dim": actions.shape[1],
            "num_timesteps": num_timesteps,
        },
        save_path,
    )
    print(f"\nModel saved to: {save_path}")
    return policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100,
                         help="Kept modest since this is a supplementary "
                              "architecture-generalization check, not the "
                              "main result (CPU-only training budget).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_timesteps", type=int, default=20,
                         help="Number of diffusion steps; kept small for "
                              "CPU-feasible rollout speed during evaluation.")
    parser.add_argument("--demo_path", type=str, default=None,
                         help="Defaults to the same Lift demo file used for BC-50/BC-500.")
    args = parser.parse_args()

    demo_path = args.demo_path or os.path.expanduser(
        "~/robosuite_work/demos_lift/low_dim.hdf5"
    )
    seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    save_path = os.path.expanduser(
        f"~/robosuite_work/demos_lift/diffusion_{args.epochs}{seed_tag}.pt"
    )
    train_diffusion(
        demo_path,
        epochs=args.epochs,
        save_path=save_path,
        seed=args.seed,
        num_timesteps=args.num_timesteps,
    )
