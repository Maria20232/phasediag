#!/usr/bin/env python3

import argparse
import os
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]


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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Random seed: {seed}")


def load_demonstrations(demo_path, obs_keys=OBS_KEYS):
    states = []
    actions = []

    with h5py.File(demo_path, "r") as f:
        demo_names = sorted(
            [
                k
                for k in f["data"].keys()
                if k.startswith("demo_")
            ],
            key=lambda name: int(
                name.split("_")[-1]
            ),
        )

        for demo_name in demo_names:
            demo = f["data"][demo_name]
            obs = demo["obs"]

            flat_obs = np.concatenate(
                [
                    obs[k][:]
                    for k in obs_keys
                ],
                axis=-1,
            )

            states.append(flat_obs)
            actions.append(
                demo["actions"][:]
            )

    return (
        np.concatenate(states, axis=0),
        np.concatenate(actions, axis=0),
    )


def train_bc(
    demo_path,
    epochs,
    seed,
    batch_size=64,
    lr=1e-3,
    save_path=None,
):

    set_seed(seed)

    print(
        f"Loading demonstrations from {demo_path}"
    )

    states, actions = load_demonstrations(
        demo_path
    )

    print(
        f"Loaded {len(states)} state-action pairs"
    )

    print(
        f"State dim: {states.shape[1]}, "
        f"Action dim: {actions.shape[1]}"
    )

    states = torch.FloatTensor(states)
    actions = torch.FloatTensor(actions)

    dataset = TensorDataset(
        states,
        actions,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    policy = BCPolicy(
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
    )

    optimizer = optim.Adam(
        policy.parameters(),
        lr=lr,
    )

    criterion = nn.MSELoss()

    print(
        f"\nTraining BC policy "
        f"for {epochs} epochs..."
    )

    final_loss = None

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch_states, batch_actions in dataloader:
            optimizer.zero_grad()

            pred_actions = policy(
                batch_states
            )

            loss = criterion(
                pred_actions,
                batch_actions,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        final_loss = (
            total_loss / num_batches
        )

        if (
            epoch % 50 == 0
            or epoch == epochs - 1
        ):
            print(
                f"Epoch {epoch}: "
                f"Loss = {final_loss:.6f}"
            )

    if save_path is None:
        save_path = os.path.join(
            os.path.dirname(demo_path),
            f"bc_{epochs}_seed{seed}.pt",
        )

    torch.save(
        policy.state_dict(),
        save_path,
    )

    print(
        f"\nModel saved to: {save_path}"
    )

    print(
        f"Final loss: {final_loss:.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    args = parser.parse_args()

    demo_path = os.path.expanduser(
        "~/robosuite_work/"
        "demos_lift/"
        "low_dim.hdf5"
    )

    train_bc(
        demo_path=demo_path,
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
        lr=args.lr,
    )
