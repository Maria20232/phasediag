import h5py
import numpy as np
from phase_stack import create_env, find_stack_phase_indices

env = create_env()
phase_errors = []
completed = 0

try:
    with h5py.File("data/demos_stack/low_dim.hdf5", "r") as f:
        names = ["demo_13", "demo_14", "demo_15"]

        for name in names:
            demo = f["data"][name]
            states = demo["states"][:]
            actions = demo["actions"][:]
            xml = demo.attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode()

            env.reset()
            phases = find_stack_phase_indices(env, states)

            # Reproduce the collection wrapper's initialization.
            env.reset()
            env.set_ep_meta(env.get_ep_meta())
            env.reset_from_xml_string(xml)
            env.sim.reset()
            env.sim.set_state_from_flattened(states[0])
            env.sim.forward()

            first_success = None
            for index, action in enumerate(actions):
                # Inspect BEFORE action[index]: actions[:index] were replayed.
                for phase, target in phases.items():
                    if target == index:
                        error = float(np.max(np.abs(
                            env.sim.get_state().flatten() - states[index]
                        )))
                        actual = env.sim.get_state()
                        nq = len(actual.qpos)
                        expected = states[index]
                        print(
                            f"{name} | {phase}"
                            f" | qpos_error={np.max(np.abs(actual.qpos - expected[1:1+nq])):.8g}"
                            f" | qvel_error={np.max(np.abs(actual.qvel - expected[1+nq:])):.8g}",
                            flush=True,
                        )
                        phase_errors.append(error)
                        print(
                            f"{name} | {phase} | index={index}"
                            f" | state_error={error:.8g}"
                            f" | initial_success={bool(env._check_success())}",
                            flush=True,
                        )

                env.step(np.clip(action, -1, 1))
                if first_success is None and env._check_success():
                    first_success = index + 1

            completed += int(first_success is not None)
            print(f"{name} | expert_success_step={first_success}", flush=True)

    print("\nSUMMARY")
    print("Expert replay successes:", completed, "/", len(names))
    print("Phase states checked:", len(phase_errors))
    print("Largest phase-state error:",
          max(phase_errors) if phase_errors else None)
finally:
    env.close()
