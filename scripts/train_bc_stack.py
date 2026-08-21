#!/usr/bin/env python3
import argparse
import random
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]

class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    def forward(self, x):
        return self.net(x)

def load_demonstrations(demo_path, obs_keys=OBS_KEYS):
    states, actions = [], []
    with h5py.File(demo_path, "r") as f:
        demo_names = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        for demo_name in demo_names:
            demo = f["data"][demo_name]
            obs = demo["obs"]
            flat_obs = np.concatenate([obs[k][:] for k in obs_keys], axis=-1)
            states.append(flat_obs)
            actions.append(demo["actions"][:])
    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)

def train_bc(demo_path, epochs=500, batch_size=64, lr=1e-3, save_path=None, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Random seed: {seed}")

    states, actions = load_demonstrations(demo_path)
    print(f"Loaded {len(states)} state-action pairs")
    print(f"State dim: {states.shape[1]}, Action dim: {actions.shape[1]}")

    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)
    dataloader = DataLoader(TensorDataset(states, actions), batch_size=batch_size, shuffle=True)

    policy = BCPolicy(state_dim=states.shape[1], action_dim=actions.shape[1])
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\nTraining BC policy for {epochs} epochs...")
    for epoch in range(epochs):
        total_loss, num_batches = 0, 0
        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()
            loss = criterion(policy(batch_states), batch_actions)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}: Loss = {total_loss/num_batches:.6f}")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(demo_path), f"bc_stack_ep{epochs}.pt")
    torch.save(policy.state_dict(), save_path)
    print(f"\nModel saved to: {save_path}")
    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    demo_path = os.path.expanduser("~/robosuite_work/demos_stack/1786611107_4237583/low_dim.hdf5")
    save_path = os.path.expanduser(f"~/robosuite_work/demos_stack/bc_stack_{args.epochs}.pt")
    train_bc(demo_path, epochs=args.epochs, save_path=save_path, seed=args.seed)
