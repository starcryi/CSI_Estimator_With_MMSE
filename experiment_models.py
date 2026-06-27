import math

import torch
import torch.nn as nn


def noise_feature(h_ls, noise_variance):
    log_noise = torch.log10(torch.clamp(noise_variance, min=1e-8))
    return log_noise[:, None, None, None, None].expand(-1, 1, *h_ls.shape[2:])


def lmmse_feature(h_ls, noise_variance, channel_variance=1.0):
    shrinkage = channel_variance / (channel_variance + noise_variance)
    return h_ls * shrinkage[:, None, None, None, None]


def tap_project_feature(value, num_taps):
    complex_value = torch.complex(value[:, 0], value[:, 1])
    taps = torch.fft.ifft(complex_value, dim=-1)
    mask = torch.zeros_like(taps)
    mask[..., :num_taps] = 1.0
    projected = torch.fft.fft(taps * mask, dim=-1)
    return torch.stack((projected.real, projected.imag), dim=1).to(value.dtype)


class ResidualCNN(nn.Module):
    def __init__(self, hidden_size=32, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv3d(3, hidden_size, 3, padding=1),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout3d(dropout))
        layers.extend(
            [
                nn.Conv3d(hidden_size, hidden_size, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(hidden_size, 2, 1),
            ]
        )
        self.network = nn.Sequential(*layers)

    def forward(self, h_ls, noise_variance):
        features = torch.cat((h_ls, noise_feature(h_ls, noise_variance)), dim=1)
        return h_ls + self.network(features)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = min(8, channels)
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class EnhancedResidualCNN(nn.Module):
    """Deeper residual CNN using LS, scalar LMMSE, and noise-level features."""

    def __init__(self, hidden_size=48, num_blocks=4):
        super().__init__()
        self.input = nn.Sequential(
            nn.Conv3d(5, hidden_size, 3, padding=1),
            nn.GroupNorm(min(8, hidden_size), hidden_size),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock3D(hidden_size) for _ in range(num_blocks)]
        )
        self.output = nn.Conv3d(hidden_size, 2, 1)

    def forward(self, h_ls, noise_variance):
        h_lmmse = lmmse_feature(h_ls, noise_variance)
        features = torch.cat(
            (h_ls, h_lmmse, noise_feature(h_ls, noise_variance)), dim=1
        )
        residual = self.output(self.blocks(self.input(features)))
        return h_lmmse + residual


class PhysicsGuidedCNN(nn.Module):
    """CNN constrained by a known maximum channel delay spread."""

    def __init__(self, hidden_size=32, num_blocks=3, num_taps=6):
        super().__init__()
        self.num_taps = num_taps
        self.input = nn.Sequential(
            nn.Conv3d(7, hidden_size, 3, padding=1),
            nn.GroupNorm(min(8, hidden_size), hidden_size),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock3D(hidden_size) for _ in range(num_blocks)]
        )
        self.output = nn.Conv3d(hidden_size, 2, 1)

    def forward(self, h_ls, noise_variance):
        h_lmmse = lmmse_feature(h_ls, noise_variance)
        h_tap = tap_project_feature(h_ls, self.num_taps)
        features = torch.cat(
            (h_ls, h_lmmse, h_tap, noise_feature(h_ls, noise_variance)), dim=1
        )
        refined = h_tap + self.output(self.blocks(self.input(features)))
        return tap_project_feature(refined, self.num_taps)


class ResidualLSTM(nn.Module):
    def __init__(self, num_rx=4, num_tx=4, hidden_size=128, num_layers=2):
        super().__init__()
        self.num_rx = num_rx
        self.num_tx = num_tx
        input_size = 2 * num_rx * num_tx + 1
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0,
        )
        self.output = nn.Linear(hidden_size, 2 * num_rx * num_tx)

    def forward(self, h_ls, noise_variance):
        batch, _, _, _, subcarriers = h_ls.shape
        sequence = h_ls.permute(0, 4, 1, 2, 3).reshape(batch, subcarriers, -1)
        log_noise = torch.log10(torch.clamp(noise_variance, min=1e-8))
        noise = log_noise[:, None, None].expand(-1, subcarriers, 1)
        residual, _ = self.lstm(torch.cat((sequence, noise), dim=-1))
        residual = self.output(residual)
        residual = residual.reshape(
            batch, subcarriers, 2, self.num_rx, self.num_tx
        ).permute(0, 2, 3, 4, 1)
        return h_ls + residual


class PositionalEncoding(nn.Module):
    def __init__(self, dimension, max_length=512):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, value):
        return value + self.encoding[:, : value.shape[1]]


class ResidualTransformer(nn.Module):
    def __init__(
        self,
        num_rx=4,
        num_tx=4,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
    ):
        super().__init__()
        self.num_rx = num_rx
        self.num_tx = num_tx
        input_size = 2 * num_rx * num_tx + 1
        self.embedding = nn.Linear(input_size, hidden_size)
        self.position = PositionalEncoding(hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.output = nn.Linear(hidden_size, 2 * num_rx * num_tx)

    def forward(self, h_ls, noise_variance):
        batch, _, _, _, subcarriers = h_ls.shape
        sequence = h_ls.permute(0, 4, 1, 2, 3).reshape(batch, subcarriers, -1)
        log_noise = torch.log10(torch.clamp(noise_variance, min=1e-8))
        noise = log_noise[:, None, None].expand(-1, subcarriers, 1)
        features = self.position(self.embedding(torch.cat((sequence, noise), dim=-1)))
        residual = self.output(self.encoder(features))
        residual = residual.reshape(
            batch, subcarriers, 2, self.num_rx, self.num_tx
        ).permute(0, 2, 3, 4, 1)
        return h_ls + residual


def build_model(name, num_rx=4, num_tx=4, hidden_size=None, dropout=0.0):
    name = name.lower()
    if name == "cnn":
        return ResidualCNN(hidden_size or 32, dropout=dropout)
    if name in ("enhanced_cnn", "rescnn"):
        return EnhancedResidualCNN(hidden_size or 48)
    if name in ("physics_cnn", "pdp_cnn"):
        return PhysicsGuidedCNN(hidden_size or 32)
    if name == "lstm":
        return ResidualLSTM(num_rx, num_tx, hidden_size or 128)
    if name == "transformer":
        return ResidualTransformer(num_rx, num_tx, hidden_size or 128)
    raise ValueError(f"Unknown model: {name}")
