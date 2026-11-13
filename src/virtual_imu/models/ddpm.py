"""Conditional Transformer DDPM with the full ancestral reverse chain.

Equations follow Ho et al. (2020), https://arxiv.org/abs/2006.11239.
The clean variable is the normalized high-minus-low IMU correction.
Both epsilon prediction and clean-sample (x0) prediction are supported.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from virtual_imu.models.transformer import PositionalEncoding


class DiffusionTransformer(nn.Module):
    """Denoise six IMU channels, conditioned on low IMU and diffusion time."""

    def __init__(
        self,
        condition_channels: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_length: int = 4096,
    ):
        super().__init__()
        if d_model < 4 or d_model % 2 or d_model % nhead:
            raise ValueError("d_model must be even and divisible by nhead")
        self.d_model = d_model
        self.max_length = max_length
        self.input = nn.Linear(6 + condition_channels, d_model)
        self.position = PositionalEncoding(d_model, max_len=max_length)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Disable nested inference path to keep CUDA/AMP and train behavior consistent.
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 6))

    def forward(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        if noisy.shape[-1] > self.max_length:
            raise ValueError("Window exceeds configured max_length")
        half = self.d_model // 2
        frequencies = torch.exp(
            torch.arange(half, device=noisy.device, dtype=torch.float32)
            * (-math.log(10000) / (half - 1))
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        embedding = self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=-1))
        tokens = self.input(torch.cat((noisy, condition), dim=1).transpose(1, 2))
        tokens = self.position(tokens) + embedding[:, None]
        return self.head(self.encoder(tokens)).transpose(1, 2)


class ConditionalDDPM(nn.Module):
    """Forward diffusion, learning objective, posterior and exact T-step sampler."""

    def __init__(
        self,
        denoiser: nn.Module,
        timesteps: int = 100,
        prediction_type: str = "x0",
        schedule: str = "cosine",
        clip_x0: float | None = None,
    ):
        super().__init__()
        if timesteps < 2 or prediction_type not in {"epsilon", "x0"}:
            raise ValueError("Need at least two timesteps and prediction_type epsilon or x0")
        self.denoiser = denoiser
        self.timesteps = int(timesteps)
        self.prediction_type = prediction_type
        self.clip_x0 = clip_x0
        if schedule == "cosine":
            t = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64) / timesteps
            abar = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
            abar = abar / abar[0]
            betas = (1 - abar[1:] / abar[:-1]).clamp(1e-6, 0.999).float()
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        alphas = 1 - betas
        abar = alphas.cumprod(0)
        if abar[-1] > 0.001:
            raise ValueError(
                "Terminal alpha_bar must be <= 0.001 for Gaussian initialization; "
                "increase timesteps or use the cosine schedule"
            )
        prev = torch.cat((torch.ones(1), abar[:-1]))
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bar", abar)
        self.register_buffer("posterior_variance", betas * (1 - prev) / (1 - abar))
        self.register_buffer("posterior_coef_x0", betas * prev.sqrt() / (1 - abar))
        self.register_buffer("posterior_coef_xt", (1 - prev) * alphas.sqrt() / (1 - abar))

    @staticmethod
    def extract(values: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return values[t, None, None]

    def q_sample(self, clean: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a = self.extract(self.alpha_bar, t)
        return a.sqrt() * clean + (1 - a).sqrt() * noise

    def predict_x0(
        self,
        noisy: torch.Tensor,
        t: torch.Tensor,
        output: torch.Tensor,
        prediction_type: str | None = None,
    ) -> torch.Tensor:
        kind = prediction_type or self.prediction_type
        a = self.extract(self.alpha_bar, t)
        clean = output if kind == "x0" else (noisy - (1 - a).sqrt() * output) / a.sqrt()
        return clean if self.clip_x0 is None else clean.clamp(-self.clip_x0, self.clip_x0)

    def q_posterior(
        self, clean: torch.Tensor, noisy: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.extract(self.posterior_coef_x0, t) * clean
        mean = mean + self.extract(self.posterior_coef_xt, t) * noisy
        return mean, self.extract(self.posterior_variance, t)

    def training_loss(
        self, clean: torch.Tensor, condition: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        t = torch.randint(self.timesteps, (len(clean),), device=clean.device, generator=generator)
        noise = torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )
        noisy = self.q_sample(clean, t, noise)
        output = self.denoiser(noisy, t, condition)
        target = noise if self.prediction_type == "epsilon" else clean
        return (output.float() - target.float()).square().mean()

    @torch.no_grad()
    def sample(
        self, condition: torch.Tensor, *, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        shape = (len(condition), 6, condition.shape[-1])
        noisy = torch.randn(shape, device=condition.device, generator=generator)
        for step in reversed(range(self.timesteps)):
            t = torch.full((len(condition),), step, device=condition.device, dtype=torch.long)
            output = self.denoiser(noisy, t, condition)
            clean = self.predict_x0(noisy, t, output.float())
            mean, variance = self.q_posterior(clean, noisy, t)
            if step:
                noise = torch.randn(shape, device=condition.device, generator=generator)
                noisy = mean + variance.sqrt() * noise
            else:
                noisy = mean
        return noisy


def build_ddpm(config: dict) -> ConditionalDDPM:
    return ConditionalDDPM(DiffusionTransformer(**config["model"]), **config["diffusion"])
