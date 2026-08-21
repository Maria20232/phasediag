#!/usr/bin/env python3
import h5py
import numpy as np
import torch
from phase_recovery import create_env, load_policy, find_phase_indices, restore_state, DATASET
from eval_bc import flatten_obs

MODEL = "../data/demos_lift/bc_500_seed42.pt"

policy = load_policy(MODEL)
env = create_env()
env.reset()

with h5py.File(DATASET, "r") as f:
    states = f["data/demo_2/states"][:]

phases, infos = find_phase_indices(env, states)
idx = phases["Pre-grasp"]
state = states[idx]
print(f"Using demo_2, Pre-grasp phase, state index {idx}")

obs_a = restore_state(env, state)
qpos_a = env.sim.data.qpos.copy()
qvel_a = env.sim.data.qvel.copy()

obs_b = restore_state(env, state)
qpos_b = env.sim.data.qpos.copy()
qvel_b = env.sim.data.qvel.copy()

qpos_match = np.array_equal(qpos_a, qpos_b)
qvel_match = np.array_equal(qvel_a, qvel_b)
print(f"\nTEST 1 -- restored qpos identical: {qpos_match}")
print(f"TEST 1 -- restored qvel identical: {qvel_match}")
if not qpos_match:
    diff = np.abs(qpos_a - qpos_b)
    print(f"  max qpos diff: {diff.max()}")

state_vec_a = flatten_obs(obs_a)
state_vec_b = flatten_obs(obs_b)
obs_vec_match = np.array_equal(state_vec_a, state_vec_b)
print(f"\nTEST 2 -- flattened obs vector identical: {obs_vec_match}")

with torch.no_grad():
    action_a = policy(torch.FloatTensor(state_vec_a)).detach().numpy()
    action_b = policy(torch.FloatTensor(state_vec_b)).detach().numpy()

action_match = np.array_equal(action_a, action_b)
print(f"TEST 2 -- policy first action identical: {action_match}")
if not action_match:
    print(f"  action_a: {action_a}")
    print(f"  action_b: {action_b}")
    print(f"  max diff: {np.abs(action_a - action_b).max()}")

restore_state(env, state)
env.step(action_a)
qpos_after_a = env.sim.data.qpos.copy()

restore_state(env, state)
env.step(action_a)
qpos_after_b = env.sim.data.qpos.copy()

step_match = np.array_equal(qpos_after_a, qpos_after_b)
print(f"\nTEST 3 -- env.step() with identical state+action gives identical result: {step_match}")
if not step_match:
    diff = np.abs(qpos_after_a - qpos_after_b)
    print(f"  max qpos diff after 1 step: {diff.max()}")

print("\nTEST 4 -- tracking divergence over a full rollout...")

def rollout_track(env, policy, state, max_steps=60):
    obs = restore_state(env, state)
    qpos_trace = []
    for step in range(max_steps):
        state_vec = flatten_obs(obs)
        with torch.no_grad():
            action = policy(torch.FloatTensor(state_vec)).detach().numpy()
        obs, reward, done, info = env.step(action)
        qpos_trace.append(env.sim.data.qpos.copy())
        if env._check_success():
            break
    return qpos_trace

trace_a = rollout_track(env, policy, state)
trace_b = rollout_track(env, policy, state)

first_divergence = None
for i in range(min(len(trace_a), len(trace_b))):
    if not np.array_equal(trace_a[i], trace_b[i]):
        first_divergence = i
        break

if first_divergence is None:
    print(f"  No divergence detected in {min(len(trace_a), len(trace_b))} steps -- rollouts identical!")
else:
    diff = np.abs(trace_a[first_divergence] - trace_b[first_divergence])
    print(f"  First divergence at step {first_divergence}, max qpos diff: {diff.max()}")

env.close()
