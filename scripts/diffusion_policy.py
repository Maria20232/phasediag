#!/usr/bin/env python3
"""
Compact state-conditioned Diffusion Policy for low-dimensional BC.

Motivated by the reviewer's suggestion to test PhaseDiag on "a small
Vision-Transformer or a Diffusion Policy" to show the framework generalizes
across architectures. Since observations here are low-dimensional state
vectors (19-dim Lift / 32-dim Stack), not images, a ViT has no natural patch
structure to exploit -- a Diffusion Policy operating directly on state
vectors is the architecturally appropriate choice (the original Diffusion
Policy paper itself evaluates a state-based, non-image variant).

This implements a small DDPM: a noise-prediction network conditioned on
(state, noisy_action, diffusion_timestep), trained to predict the noise
added to expert actions, with ancestral sampling at inference time.

The DiffusionBCPolicy class exposes the SAME calling convention as the
existing BCPolicy (policy(state_tensor) -> action_tensor), so it plugs
directly into phase_recovery.py / phase_stack.py / eval_bc.py's rollout code
without modification, aside from the checkpoint-loading line.
"""

import math

import torch
import torch.nn as nn


def sinusoidal_embedding(timesteps, dim):
    """Standard sinusoidal timestep embedding (as in DDPM / Transformer PE)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class NoisePredictor(nn.Module):
    """Predicts the noise added to an action, conditioned on state and timestep."""

    def __init__(self, state_dim, action_dim=7, hidden=128, time_emb_dim=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + time_emb_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state, noisy_action, t):
        t_emb = sinusoidal_embedding(t, self.time_emb_dim)
        x = torch.cat([state, noisy_action, t_emb], dim=-1)
        return self.net(x)


class DiffusionBCPolicy(nn.Module):
    """DDPM-based behavior cloning policy over a low-dimensional action space.

    Calling convention matches BCPolicy: policy(state_tensor) runs full
    reverse diffusion sampling and returns a single predicted action tensor,
    so it is a drop-in replacement inside existing rollout code.
    """

    def __init__(self, state_dim, action_dim=7, hidden=128, time_emb_dim=32,
                 num_timesteps=20, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.action_dim = action_dim
        self.num_timesteps = num_timesteps

        self.noise_pred = NoisePredictor(state_dim, action_dim, hidden, time_emb_dim)

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def diffusion_loss(self, state, action):
        """DDPM training loss: predict the noise added to a batch of expert actions."""
        batch_size = state.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=state.device)
        noise = torch.randn_like(action)

        sqrt_ac = self.alphas_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_ac = (1 - self.alphas_cumprod[t]).sqrt().unsqueeze(-1)
        noisy_action = sqrt_ac * action + sqrt_one_minus_ac * noise

        predicted_noise = self.noise_pred(state, noisy_action, t)
        return nn.functional.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, state):
        """Ancestral (DDPM) sampling: returns one action for a single state.

        state: 1D tensor of shape (state_dim,)
        returns: 1D tensor of shape (action_dim,)
        """
        device = state.device
        state_batch = state.unsqueeze(0)
        x = torch.randn(1, self.action_dim, device=device)

        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.full((1,), t_idx, dtype=torch.long, device=device)
            predicted_noise = self.noise_pred(state_batch, x, t)

            alpha_t = self.alphas[t_idx]
            alpha_cumprod_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]

            mean = (1.0 / alpha_t.sqrt()) * (
                x - (beta_t / (1 - alpha_cumprod_t).sqrt()) * predicted_noise
            )

            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean

        return x.squeeze(0).clamp(-1.0, 1.0)

    def forward(self, state):
        """Matches BCPolicy's calling convention: single state -> single action."""
        return self.sample(state)
