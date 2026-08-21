#!/usr/bin/env python3
import numpy as np
import h5py
import torch
import torch.nn as nn

OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"]

class BCPolicy(nn.Module):
    def __init__(self, state_dim, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    def forward(self, x):
        return self.net(x)

def load_states_actions(demo_path, obs_keys=OBS_KEYS):
    states, actions = [], []
    with h5py.File(demo_path, "r") as f:
        demo_names = sorted(
            [k for k in f["data"].keys() if k.startswith("demo_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        for demo_name in demo_names:
            demo = f["data"][demo_name]
            obs = demo["obs"]
            flat_obs = np.concatenate([obs[k][:] for k in obs_keys], axis=-1)
            states.append(flat_obs)
            actions.append(demo["actions"][:])
    return np.concatenate(states, axis=0), np.concatenate(actions, axis=0)

def compute_metrics(pred_actions, expert_actions):
    diff = pred_actions - expert_actions
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff ** 2))
    pred_norm = np.linalg.norm(pred_actions, axis=1)
    expert_norm = np.linalg.norm(expert_actions, axis=1)
    valid = (pred_norm > 1e-8) & (expert_norm > 1e-8)
    cos_sim = np.sum(pred_actions[valid] * expert_actions[valid], axis=1) / (
        pred_norm[valid] * expert_norm[valid]
    )
    gripper_acc = np.mean(np.sign(pred_actions[:, 6]) == np.sign(expert_actions[:, 6]))
    return {"MAE": mae, "RMSE": rmse, "Cosine": np.mean(cos_sim), "GripperAcc": gripper_acc}

def evaluate_offline(model_path, dataset_path):
    states, expert_actions = load_states_actions(dataset_path)
    print(f"Loaded {len(states)} state-action pairs from {dataset_path}")
    policy = BCPolicy(state_dim=states.shape[1], action_dim=expert_actions.shape[1])
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()
    with torch.no_grad():
        pred_actions = policy(torch.FloatTensor(states)).numpy()
    metrics = compute_metrics(pred_actions, expert_actions)
    print(f"\nModel: {model_path}")
    print(f"  RMSE:       {metrics['RMSE']:.4f}")
    print(f"  MAE:        {metrics['MAE']:.4f}")
    print(f"  Cosine:     {metrics['Cosine']:.4f}")
    print(f"  GripperAcc: {metrics['GripperAcc']*100:.1f}%")
    return metrics

if __name__ == "__main__":
    dataset = "../data/demos_stack/1786611107_4237583/low_dim.hdf5"
    models = [
        "../data/demos_stack/bc_stack_50.pt",
        "../data/demos_stack/bc_stack_500.pt",
    ]
    print("=" * 60)
    print("OFFLINE IMITATION QUALITY - STACK")
    print("=" * 60)
    results = {}
    for m in models:
        results[m] = evaluate_offline(m, dataset)
    print("\nSUMMARY")
    print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'Cosine':>8} {'GripperAcc':>12}")
    for m, r in results.items():
        print(f"{m.split('/')[-1]:<20} {r['RMSE']:>8.4f} {r['MAE']:>8.4f} {r['Cosine']:>8.4f} {r['GripperAcc']*100:>11.1f}%")
