import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from phase_stack import create_env, load_policy, flatten_obs
from phase_initialization import initialize_from_expert_prefix

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

dataset = Path("data/demos_stack/low_dim.hdf5")
manifest = Path("results/revision/stack_initialization_validation.json")
output = Path(args.output)
if output.exists():
    raise SystemExit(f"Output already exists: {output}")

entries = json.loads(manifest.read_text())
if len(entries) != 45 or any(
    entry["status"] != "expert_success" for entry in entries
):
    raise SystemExit("Expected 45 successfully validated phase states.")

policy = load_policy(args.model)
env = create_env()
records = []

try:
    with h5py.File(dataset, "r") as f:
        for entry in entries:
            name, phase, index = entry["demo"], entry["phase"], entry["index"]
            demo = f["data"][name]
            obs, checks = initialize_from_expert_prefix(
                env, demo.attrs["model_file"],
                demo["states"][:], demo["actions"][:], index,
            )
            if checks["restored_state_error"] != 0:
                raise RuntimeError(f"Restoration mismatch: {name}, {phase}")
            if checks["initially_successful"]:
                raise RuntimeError(f"Initially successful: {name}, {phase}")

            success_step = None
            reason = "horizon"
            for step in range(1, 1001):
                with torch.no_grad():
                    action = policy(torch.as_tensor(
                        flatten_obs(obs), dtype=torch.float32
                    )).numpy()
                if not np.all(np.isfinite(action)):
                    reason = "nonfinite_action"
                    break
                obs, _, _, _ = env.step(np.clip(action, -1, 1))
                if not (
                    np.all(np.isfinite(env.sim.data.qpos))
                    and np.all(np.isfinite(env.sim.data.qvel))
                ):
                    reason = "nonfinite_state"
                    break
                if env._check_success():
                    success_step = step
                    reason = "success"
                    break

            record = {
                "demo": name, "phase": phase, "index": index,
                "success_step": success_step, "reason": reason,
                "steps_executed": step, **checks,
            }
            records.append(record)
            print(json.dumps(record), flush=True)

    result = {
        "model": str(Path(args.model).resolve()),
        "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "continuation_budget": 1000,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))

    print("\nSUMMARY")
    for phase in ("Approach", "Grasp-transport", "Near-placement"):
        rows = [r for r in records if r["phase"] == phase]
        for budget in (500, 1000):
            successes = sum(
                r["success_step"] is not None and r["success_step"] <= budget
                for r in rows
            )
            print(f"{phase} | budget={budget} | {successes}/{len(rows)}")
finally:
    env.close()
