import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from phase_recovery import create_env, load_policy, flatten_obs
from phase_initialization import initialize_from_expert_prefix

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

model = Path(args.model).expanduser().resolve()
output = Path(args.output)
dataset = Path("data/demos_lift/low_dim.hdf5")
manifest = Path("results/revision/lift_initialization_validation.json")

if output.exists():
    raise SystemExit(f"Already exists: {output}")

entries = json.loads(manifest.read_text())
phases = ("Approach", "Pre-grasp", "Grasp", "Post-grasp", "Pre-success")
if len(entries) != 125:
    raise SystemExit("Expected 125 phase entries.")
if any(r["status"] not in ("expert_success", "missing_phase") for r in entries):
    raise SystemExit("Manifest contains an unresolved validation result.")
if len({(r["demo"], r["phase"]) for r in entries}) != 125:
    raise SystemExit("Manifest contains duplicate entries.")

policy = load_policy(str(model))
policy.eval()
env = create_env()
records = []

try:
    with h5py.File(dataset, "r") as f:
        for entry in entries:
            name, phase, index = entry["demo"], entry["phase"], entry["index"]
            if entry["status"] == "missing_phase":
                record = {
                    "demo": name, "phase": phase, "index": index,
                    "reason": "missing_phase",
                }
            else:
                demo = f["data"][name]
                obs, checks = initialize_from_expert_prefix(
                    env, demo.attrs["model_file"],
                    demo["states"][:], demo["actions"][:], index,
                )
                if checks["restored_state_error"] != 0:
                    raise RuntimeError(f"Restoration failed: {name}, {phase}")
                if checks["initially_successful"]:
                    raise RuntimeError(f"Initially successful: {name}, {phase}")

                success_step = None
                reason = "horizon"
                for step in range(1, 1601):
                    with torch.no_grad():
                        action = policy(torch.as_tensor(
                            flatten_obs(obs), dtype=torch.float32
                        )).detach().cpu().numpy().reshape(-1)

                    if not np.all(np.isfinite(action)):
                        reason = "nonfinite_action"
                        break

                    obs, _, _, _ = env.step(np.clip(action, -1, 1))
                    if (not np.all(np.isfinite(env.sim.data.qpos))
                            or not np.all(np.isfinite(env.sim.data.qvel))):
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
        "model": str(model),
        "model_sha256": sha256(model),
        "dataset_sha256": sha256(dataset),
        "manifest_sha256": sha256(manifest),
        "continuation_budget": 1600,
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    print("\nSUMMARY")
    for phase in phases:
        rows = [
            r for r in records
            if r["phase"] == phase and r["reason"] != "missing_phase"
        ]
        for budget in (400, 1600):
            successes = sum(
                r["success_step"] is not None and r["success_step"] <= budget
                for r in rows
            )
            print(f"{phase} | budget={budget} | {successes}/{len(rows)}")
    print("Numerical failures:", sum(
        r["reason"] in ("nonfinite_action", "nonfinite_state") for r in records
    ))
finally:
    env.close()
