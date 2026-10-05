import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_bc import BCPolicy, OBS_KEYS, set_seed

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=("lift", "stack"), required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--dry_run", action="store_true")
args = parser.parse_args()

dataset_path = Path(f"data/demos_{args.task}/low_dim.hdf5")
output_dir = Path(f"data/revision/train_split/{args.task}")
checkpoints = {
    epoch: output_dir / f"bc_{epoch}_seed{args.seed}.pt"
    for epoch in (50, 500)
}
metadata_path = output_dir / f"training_seed{args.seed}.json"

with h5py.File(dataset_path, "r") as f:
    def names(split):
        return sorted(
            (n.decode() if isinstance(n, bytes) else str(n)
             for n in f[f"mask/{split}"][:]),
            key=lambda n: int(n.split("_")[-1]),
        )

    train_names, valid_names = names("train"), names("valid")
    all_names = {n for n in f["data"] if n.startswith("demo_")}
    assert train_names and valid_names
    assert len(train_names) == len(set(train_names))
    assert len(valid_names) == len(set(valid_names))
    assert not set(train_names) & set(valid_names)
    assert set(train_names) | set(valid_names) == all_names

    states, actions = [], []
    for name in train_names:
        demo = f["data"][name]
        obs = np.concatenate(
            [demo["obs"][key][:] for key in OBS_KEYS], axis=-1
        )
        ac = demo["actions"][:]
        assert len(obs) == len(ac)
        states.append(obs)
        actions.append(ac)

states = np.concatenate(states).astype(np.float32)
actions = np.concatenate(actions).astype(np.float32)
assert np.isfinite(states).all() and np.isfinite(actions).all()

print(f"Task: {args.task}", flush=True)
print(f"Training: {len(train_names)} demos, {len(states)} transitions")
print(f"Held out: {valid_names}")
print(f"Dimensions: observations={states.shape[1]}, actions={actions.shape[1]}")
print(f"Output directory: {output_dir}", flush=True)

if args.dry_run:
    raise SystemExit("Data check passed. No training started.")

if any(p.exists() for p in (*checkpoints.values(), metadata_path)):
    raise SystemExit("An output already exists; refusing to overwrite.")

output_dir.mkdir(parents=True, exist_ok=True)
set_seed(args.seed)
generator = torch.Generator().manual_seed(args.seed)
loader = DataLoader(
    TensorDataset(torch.from_numpy(states), torch.from_numpy(actions)),
    batch_size=64, shuffle=True, generator=generator,
)
policy = BCPolicy(states.shape[1], actions.shape[1])
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

digest = hashlib.sha256()
with dataset_path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)

metadata = {
    "task": args.task, "seed": args.seed,
    "dataset_sha256": digest.hexdigest(),
    "train_demos": train_names, "valid_demos": valid_names,
    "training_transitions": len(states),
    "obs_keys": OBS_KEYS, "batch_size": 64, "lr": 1e-3,
    "checkpoint_epochs": [50, 500],
    "torch_version": str(torch.__version__),
    "numpy_version": np.__version__,
}
metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

for epoch in range(1, 501):
    policy.train()
    total, count = 0.0, 0
    for obs, action in loader:
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(policy(obs), action)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss.")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(obs)
        count += len(obs)

    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch {epoch}: train MSE={total/count:.6f}", flush=True)
    if epoch in checkpoints:
        torch.save(policy.state_dict(), checkpoints[epoch])
        print(f"Saved: {checkpoints[epoch]}", flush=True)
