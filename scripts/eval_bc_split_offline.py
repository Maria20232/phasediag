import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from train_bc import BCPolicy, OBS_KEYS

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
results = []

for task in ("lift", "stack"):
    with h5py.File(f"data/demos_{task}/low_dim.hdf5", "r") as f:
        for epochs in (50, 500):
            model = Path(
                f"data/revision/train_split/{task}/bc_{epochs}_seed{args.seed}.pt"
            )
            state_dict = torch.load(
                model, map_location="cpu", weights_only=True
            )
            policy = BCPolicy(state_dict["net.0.weight"].shape[1])
            policy.load_state_dict(state_dict)
            policy.eval()

            for split in ("train", "valid"):
                names = [
                    n.decode() if isinstance(n, bytes) else str(n)
                    for n in f[f"mask/{split}"][:]
                ]
                demo_results = []
                squared_error, elements = 0.0, 0

                for name in names:
                    demo = f["data"][name]
                    obs = np.concatenate(
                        [demo["obs"][k][:] for k in OBS_KEYS], axis=-1
                    )
                    actions = demo["actions"][:]
                    with torch.no_grad():
                        predicted = policy(torch.as_tensor(
                            obs, dtype=torch.float32
                        )).numpy()

                    error = predicted.astype(np.float64) - actions
                    assert np.isfinite(error).all()
                    total = float(np.sum(error ** 2))
                    squared_error += total
                    elements += error.size
                    demo_results.append({
                        "demo": name,
                        "transitions": len(actions),
                        "mse": total / error.size,
                    })

                record = {
                    "task": task, "epochs": epochs, "seed": args.seed,
                    "model": str(model), "split": split,
                    "transition_weighted_mse": squared_error / elements,
                    "demo_mean_mse": float(np.mean(
                        [r["mse"] for r in demo_results]
                    )),
                    "per_demo": demo_results,
                }
                results.append(record)
                print(
                    f"{task} BC-{epochs} {split}: "
                    f"MSE={record['transition_weighted_mse']:.6f} | "
                    f"demo mean={record['demo_mean_mse']:.6f}",
                    flush=True,
                )

output = Path(f"results/revision/offline_split_seed{args.seed}.json")
if output.exists():
    raise SystemExit(f"Already exists: {output}")
output.write_text(json.dumps(results, indent=2) + "\n")
print(f"Saved: {output}")
