import hashlib
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

from eval_bc import flatten_obs
from phase_recovery import create_env, load_policy
from phase_initialization import initialize_from_expert_prefix

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/revision"
DATASET = ROOT / "data/demos_lift/low_dim.hdf5"
MANIFEST = RESULTS / "lift_initialization_validation.json"
OUTPUT = RESULTS / "entropy_lift_validated.json"
PHASES = ["Approach", "Pre-grasp", "Grasp", "Post-grasp", "Pre-success"]
SAMPLES = 30
RNG_SEED = 20261010

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if OUTPUT.exists():
    raise SystemExit(f"Output already exists; refusing to overwrite: {OUTPUT}")

entries = json.loads(MANIFEST.read_text())
if len(entries) != 125:
    raise RuntimeError("Expected 125 manifest entries.")
if len({(r["demo"], r["phase"]) for r in entries}) != 125:
    raise RuntimeError("Duplicate manifest entries.")
if any(r["status"] not in ("expert_success", "missing_phase") for r in entries):
    raise RuntimeError("Manifest contains unresolved validation results.")

valid = [r for r in entries if r["status"] == "expert_success"]
counts = Counter(r["phase"] for r in valid)
expected = dict(zip(PHASES, [25, 25, 23, 23, 25]))
if dict(counts) != expected:
    raise RuntimeError(f"Unexpected phase counts: {counts}")

dataset_hash = sha256(DATASET)
manifest_hash = sha256(MANIFEST)
models = []

# Match the models to the previously saved continuation evaluations.
for epochs in (50, 500):
    for seed in range(42, 49):
        previous_path = RESULTS / f"lift_bc{epochs}_seed{seed}_validated.json"
        previous = json.loads(previous_path.read_text())
        if previous["dataset_sha256"] != dataset_hash:
            raise RuntimeError(f"Dataset mismatch: {previous_path}")
        if previous["manifest_sha256"] != manifest_hash:
            raise RuntimeError(f"Manifest mismatch: {previous_path}")

        model_path = Path(previous["model"]).expanduser()
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        if sha256(model_path) != previous["model_sha256"]:
            raise RuntimeError(f"Model mismatch: {model_path}")

        models.append({
            "epochs": epochs,
            "training_seed": seed,
            "path": str(model_path),
            "sha256": previous["model_sha256"],
            "policy": load_policy(str(model_path)),
        })

print("Verified all 14 models, dataset, and manifest.", flush=True)
print("Validated states per phase:", dict(counts), flush=True)

# Enable only dropout; leave other modules in evaluation mode.
for model in models:
    model["policy"].eval()
    for module in model["policy"].modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
torch.use_deterministic_algorithms(True)
records = []
env = create_env()

try:
    with h5py.File(DATASET, "r") as f:
        for number, entry in enumerate(valid):
            demo = f["data"][entry["demo"]]
            obs, checks = initialize_from_expert_prefix(
                env,
                demo.attrs["model_file"],
                demo["states"][:],
                demo["actions"][:],
                entry["index"],
            )
            if checks["restored_state_error"] != 0.0:
                raise RuntimeError(f"State restoration failed: {entry}")
            if checks["initially_successful"]:
                raise RuntimeError(f"Already successful: {entry}")

            vector = np.asarray(flatten_obs(obs), dtype=np.float32)
            if not np.isfinite(vector).all():
                raise RuntimeError(f"Nonfinite observation: {entry}")
            x = torch.from_numpy(vector)

            for model in models:
                dropout_seed = (
                    RNG_SEED + number * 10000
                    + model["epochs"] * 10 + model["training_seed"]
                )
                torch.manual_seed(dropout_seed)
                with torch.no_grad():
                    samples = torch.stack([
                        model["policy"](x) for _ in range(SAMPLES)
                    ]).cpu().numpy()
                if not np.isfinite(samples).all():
                    raise RuntimeError("Nonfinite action predictions.")

                variance = samples.var(axis=0, ddof=0)
                entropy = float(
                    0.5 * np.log(2 * np.pi * np.e * (variance + 1e-8)).sum()
                )
                records.append({
                    "epochs": model["epochs"],
                    "training_seed": model["training_seed"],
                    "demo": entry["demo"],
                    "phase": entry["phase"],
                    "index": entry["index"],
                    "dropout_seed": dropout_seed,
                    "entropy_nats": entropy,
                    "action_variance": variance.tolist(),
                    "observation": vector.tolist(),
                    **checks,
                })

            print(
                f"{number + 1}/{len(valid)}: "
                f"{entry['demo']} {entry['phase']}",
                flush=True,
            )
finally:
    env.close()

summary = {}
for epochs in (50, 500):
    label = f"BC-{epochs}"
    summary[label] = {}
    print(f"\n{label}: validated entropy results")
    print(f"{'Phase':<15} {'Mean':>12} {'Std':>12} {'N':>6}")
    for phase in PHASES:
        values = [
            r["entropy_nats"] for r in records
            if r["epochs"] == epochs and r["phase"] == phase
        ]
        stats = {
            "mean": float(np.mean(values)),
            "std_population": float(np.std(values, ddof=0)),
            "n": len(values),
        }
        summary[label][phase] = stats
        print(
            f"{phase:<15} {stats['mean']:>12.6f} "
            f"{stats['std_population']:>12.6f} {stats['n']:>6}"
        )

result = {
    "dataset_sha256": dataset_hash,
    "manifest_sha256": manifest_hash,
    "script_sha256": sha256(__file__),
    "initialization_script_sha256": sha256(
        ROOT / "scripts/phase_initialization.py"
    ),
    "samples_per_state": SAMPLES,
    "rng_seed": RNG_SEED,
    "variance_ddof": 0,
    "variance_epsilon": 1e-8,
    "actions_clipped": False,
    "torch_version": str(torch.__version__),
    "numpy_version": np.__version__,
    "models": [
        {k: v for k, v in m.items() if k != "policy"} for m in models
    ],
    "summary": summary,
    "records": records,
}
with OUTPUT.open("x") as f:
    json.dump(result, f, indent=2, allow_nan=False)
    f.write("\n")
print(f"\nSaved: {OUTPUT}", flush=True)
