import numpy as np

def initialize_from_expert_prefix(env, xml, states, actions, index):
    """Reconstruct controller history, then restore recorded physical state."""
    if isinstance(xml, bytes):
        xml = xml.decode()
    if not 0 <= index < len(states) or index > len(actions):
        raise ValueError("Invalid phase index.")

    env.reset()
    env.set_ep_meta(env.get_ep_meta())
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    for action in actions[:index]:
        env.step(np.clip(action, -1, 1))

    prefix_error = float(np.max(np.abs(
        env.sim.get_state().flatten() - states[index]
    )))

    env.sim.set_state_from_flattened(states[index])
    env.sim.forward()
    controller = env.robots[0].composite_controller.part_controllers["right"]
    controller.update(force=True)
    obs = env._get_observations(force_update=True)

    restored_error = float(np.max(np.abs(
        env.sim.get_state().flatten() - states[index]
    )))
    return obs, {
        "prefix_state_error": prefix_error,
        "restored_state_error": restored_error,
        "initially_successful": bool(env._check_success()),
    }
