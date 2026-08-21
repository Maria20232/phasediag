#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object-state"]

class BCPolicy(nn.Module):
    def __init__(self, state_dim=19, action_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )
    def forward(self, x):
        return self.net(x)

def flatten_obs(obs_dict, obs_keys=OBS_KEYS):
    return np.concatenate([np.asarray(obs_dict[k]).flatten() for k in obs_keys], axis=-1)

def evaluate_policy(model_path, num_episodes=20, max_steps=400, render=False):
    controller_config = load_composite_controller_config(controller=None, robot="Panda")
    env = suite.make(
        env_name="Lift", robots="Panda", controller_configs=controller_config,
        has_renderer=render, has_offscreen_renderer=False, use_camera_obs=False,
        reward_shaping=True, control_freq=20, ignore_done=True,
    )
    policy = BCPolicy()
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    successes = 0
    episode_lengths = []
    for ep in range(num_episodes):
        obs = env.reset()
        success = False
        for step in range(max_steps):
            state = flatten_obs(obs)
            with torch.no_grad():
                action = policy(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
            action = np.clip(action, -1.0, 1.0)
            obs, reward, done, info = env.step(action)
            if render:
                env.render()
            if env._check_success():
                success = True
                break
        successes += int(success)
        episode_lengths.append(step + 1)
        print(f"  Episode {ep+1}/{num_episodes}: {'SUCCESS' if success else 'fail'} ({step+1} steps)")

    env.close()
    success_rate = successes / num_episodes
    print(f"\nModel: {model_path}")
    print(f"Success rate: {successes}/{num_episodes} = {success_rate*100:.1f}%")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} steps")
    return success_rate, episode_lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    evaluate_policy(args.model, args.episodes, args.max_steps, args.render)
