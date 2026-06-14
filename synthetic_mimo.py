import math

import numpy as np
import torch
from torch.utils.data import Dataset


def complex_to_channels(value):
    return np.stack((value.real, value.imag), axis=1).astype(np.float32)


def channels_to_complex(value):
    return value[:, 0] + 1j * value[:, 1]


def _spatial_correlation(size, coefficient):
    indices = np.arange(size)
    return coefficient ** np.abs(indices[:, None] - indices[None, :])


def _unitary_dft(size):
    rows = np.arange(size)[:, None]
    columns = np.arange(size)[None, :]
    return np.exp(-2j * np.pi * rows * columns / size) / np.sqrt(size)


class SyntheticMIMODataset(Dataset):
    """Frequency-selective MIMO channels observed through orthogonal pilots."""

    def __init__(
        self,
        num_samples,
        num_rx=4,
        num_tx=4,
        num_subcarriers=32,
        num_taps=6,
        snr_min_db=0.0,
        snr_max_db=30.0,
        spatial_correlation=0.6,
        seed=0,
    ):
        super().__init__()
        self.num_rx = num_rx
        self.num_tx = num_tx
        self.num_subcarriers = num_subcarriers

        rng = np.random.default_rng(seed)
        pilot = _unitary_dft(num_tx).astype(np.complex64)
        pilot_h = pilot.conj().T

        rx_cholesky = np.linalg.cholesky(
            _spatial_correlation(num_rx, spatial_correlation)
        )
        tx_cholesky = np.linalg.cholesky(
            _spatial_correlation(num_tx, spatial_correlation)
        )

        tap_power = np.exp(-np.arange(num_taps, dtype=np.float32))
        tap_power /= tap_power.sum()
        tap_scale = np.sqrt(tap_power / 2.0)[None, None, None, :]

        raw_taps = (
            rng.standard_normal((num_samples, num_rx, num_tx, num_taps))
            + 1j * rng.standard_normal((num_samples, num_rx, num_tx, num_taps))
        )
        raw_taps *= tap_scale
        correlated_taps = np.einsum(
            "ra,natl,bt->nrbl",
            rx_cholesky,
            raw_taps,
            tx_cholesky.conj(),
            optimize=True,
        )
        channel = np.fft.fft(
            correlated_taps, n=num_subcarriers, axis=-1
        ).astype(np.complex64)

        clean_received = np.einsum(
            "nrtk,tp->nrpk", channel, pilot, optimize=True
        )
        snr_db = rng.uniform(snr_min_db, snr_max_db, size=num_samples).astype(
            np.float32
        )
        signal_power = np.mean(
            np.abs(clean_received) ** 2, axis=(1, 2, 3)
        ).astype(np.float32)
        noise_variance = signal_power / np.power(10.0, snr_db / 10.0)
        noise = (
            rng.standard_normal(clean_received.shape)
            + 1j * rng.standard_normal(clean_received.shape)
        ) * np.sqrt(noise_variance[:, None, None, None] / 2.0)
        received = clean_received + noise

        h_ls = np.einsum(
            "nrpk,pt->nrtk", received, pilot_h, optimize=True
        ).astype(np.complex64)

        self.h_true = torch.from_numpy(complex_to_channels(channel))
        self.h_ls = torch.from_numpy(complex_to_channels(h_ls))
        self.snr_db = torch.from_numpy(snr_db)
        self.noise_variance = torch.from_numpy(noise_variance)

    def __len__(self):
        return self.h_true.shape[0]

    def __getitem__(self, index):
        return (
            self.h_ls[index],
            self.h_true[index],
            self.snr_db[index],
            self.noise_variance[index],
        )


def lmmse_from_ls(h_ls, noise_variance, channel_variance=1.0):
    shrinkage = channel_variance / (channel_variance + noise_variance)
    return h_ls * shrinkage[:, None, None, None, None]


def mse_per_sample(prediction, target):
    dimensions = tuple(range(1, prediction.ndim))
    return torch.mean((prediction - target) ** 2, dim=dimensions)


def nmse_per_sample(prediction, target, eps=1e-12):
    dimensions = tuple(range(1, prediction.ndim))
    error = torch.sum((prediction - target) ** 2, dim=dimensions)
    power = torch.sum(target**2, dim=dimensions)
    return error / torch.clamp(power, min=eps)


def nmse_db(nmse):
    return 10.0 * torch.log10(torch.clamp(nmse, min=1e-12))


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
