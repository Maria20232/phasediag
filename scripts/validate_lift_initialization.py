import json
from pathlib import Path

import h5py
import numpy as np

from phase_recovery import create_env, find_phase_indices
from phase_initialization import initialize_from_expert_prefix

output = Path("results/revision/lift_initialization_validation.json")
if output.exists():
    raise SystemExit(f"Already exists: {output}")

env = create_env()
records = []

try:
    with h5py.File("data/demos_lift/low_dim.hdf5", "r") as f:
        names = sorted(
            (n for n in f["data"] if n.startswith("demo_")),
            key=lambda n: int(n.split("_")[-1]),
        )

        for name in names:
            demo = f["data"][name]
            states = demo["states"][:]
            actions = demo["actions"][:]
            xml = demo.attrs["model_file"]

            env.reset()
            phases, _ = find_phase_indices(env, states)

            for phase, index in phases.items():
                record = {"demo": name, "phase": phase, "index": index}

                if index is None:
                    record["status"] = "missing_phase"
                else:
                    _, checks = initialize_from_expert_prefix(
                        env, xml, states, actions, index
                    )
                    record.update(checks)
                    success_step = None

                    if checks["initially_successful"]:
                        record["status"] = "initially_successful"
                    else:
                        record["status"] = "expert_failure"
                        for step, action in enumerate(actions[index:], start=1):
                            env.step(np.clip(action, -1, 1))
                            if env._check_success():
                                success_step = step
                                record["status"] = "expert_success"
                                break

                    record["expert_success_step"] = success_step

                records.append(record)
                print(json.dumps(record), flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2) + "\n")

    print("\nSUMMARY", flush=True)
    for status in sorted({r["status"] for r in records}):
        print(status, sum(r["status"] == status for r in records))
    print("Phase entries:", len(records))
    print("Maximum restored-state error:", max(
        (r["restored_state_error"] for r in records
         if "restored_state_error" in r), default=None
    ))
finally:
    env.close()
