import json
from pathlib import Path
import h5py
import numpy as np
from phase_stack import create_env, find_stack_phase_indices
from phase_initialization import initialize_from_expert_prefix

if Path("results/revision/stack_heldout_initialization.json").exists():
    raise SystemExit("Held-out manifest already exists.")
env = create_env()
records = []

try:
    with h5py.File("data/demos_stack/low_dim.hdf5", "r") as f:
        names = sorted(
            (n.decode() if isinstance(n, bytes) else str(n)
             for n in f["mask/valid"][:]),
            key=lambda n: int(n.split("_")[-1]),
        )

        for name in names:
            demo = f["data"][name]
            states = demo["states"][:]
            actions = demo["actions"][:]
            xml = demo.attrs["model_file"]

            env.reset()
            phases = find_stack_phase_indices(env, states)

            for phase in ("Approach", "Grasp-transport", "Near-placement"):
                index = phases.get(phase)
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
                        for step, action in enumerate(actions[index:], start=1):
                            env.step(np.clip(action, -1, 1))
                            if env._check_success():
                                success_step = step
                                break
                        record["status"] = (
                            "expert_success" if success_step is not None
                            else "expert_failure"
                        )
                    record["expert_success_step"] = success_step

                records.append(record)
                print(json.dumps(record), flush=True)

    output = Path("results/revision/stack_heldout_initialization.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2))

    print("\nSUMMARY")
    for status in (
        "expert_success", "expert_failure",
        "initially_successful", "missing_phase",
    ):
        print(status, sum(r["status"] == status for r in records))
    print("Maximum restored-state error:", max(
        (r["restored_state_error"] for r in records
         if "restored_state_error" in r), default=None
    ))
finally:
    env.close()
