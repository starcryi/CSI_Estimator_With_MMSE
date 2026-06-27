import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from synthetic_mimo import (
    SyntheticMIMODataset,
    count_parameters,
    lmmse_from_ls,
    nmse_db,
    nmse_per_sample,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA PyTorch is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def make_dataset(config, num_samples, seed, **overrides):
    values = {
        "num_samples": num_samples,
        "num_rx": config.num_rx,
        "num_tx": config.num_tx,
        "num_subcarriers": config.num_subcarriers,
        "num_taps": config.num_taps,
        "snr_min_db": config.snr_min_db,
        "snr_max_db": config.snr_max_db,
        "spatial_correlation": config.spatial_correlation,
        "channel_type": getattr(config, "channel_type", "multipath"),
        "iq_imbalance": getattr(config, "iq_imbalance", False),
        "quantization_bits": getattr(config, "quantization_bits", None),
        "seed": seed,
    }
    values.update(overrides)
    return SyntheticMIMODataset(**values)


def make_loaders(config, **dataset_overrides):
    train_set = make_dataset(
        config, config.train_samples, config.seed, **dataset_overrides
    )
    val_set = make_dataset(
        config, config.val_samples, config.seed + 1, **dataset_overrides
    )
    test_set = make_dataset(
        config, config.test_samples, config.seed + 2, **dataset_overrides
    )
    generator = torch.Generator().manual_seed(config.seed)
    common = {
        "batch_size": config.batch_size,
        "num_workers": getattr(config, "num_workers", 0),
        "pin_memory": select_device(config.device).type == "cuda",
    }
    return (
        DataLoader(train_set, shuffle=True, generator=generator, **common),
        DataLoader(val_set, shuffle=False, **common),
        DataLoader(test_set, shuffle=False, **common),
    )


def _train_or_validate(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    losses = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for h_ls, h_true, _, noise_variance in loader:
            h_ls = h_ls.to(device, non_blocking=True)
            h_true = h_true.to(device, non_blocking=True)
            noise_variance = noise_variance.to(device, non_blocking=True)
            prediction = model(h_ls, noise_variance)
            loss = torch.mean(nmse_per_sample(prediction, h_true))
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
    return float(np.mean(losses))


def train_model(model, train_loader, val_loader, config, output_dir, name):
    device = select_device(config.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    checkpoint = Path(output_dir) / f"{name}_best.pt"
    best_val = float("inf")
    history = {"train": [], "validation": []}
    for epoch in range(1, config.epochs + 1):
        train_loss = _train_or_validate(model, train_loader, device, optimizer)
        val_loss = _train_or_validate(model, val_loader, device)
        history["train"].append(train_loss)
        history["validation"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint)
        print(
            f"{name} epoch {epoch:02d}/{config.epochs}: "
            f"train={train_loss:.6f}, val={val_loss:.6f}"
        )
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    return model, history, best_val


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    result = {"true": [], "ls": [], "lmmse": [], "prediction": [], "snr": []}
    for h_ls, h_true, snr, noise_variance in loader:
        prediction = model(
            h_ls.to(device), noise_variance.to(device)
        ).cpu()
        result["true"].append(h_true)
        result["ls"].append(h_ls)
        result["lmmse"].append(lmmse_from_ls(h_ls, noise_variance))
        result["prediction"].append(prediction)
        result["snr"].append(snr)
    return {key: torch.cat(value) for key, value in result.items()}


def mean_nmse_db(prediction, target):
    return float(nmse_db(torch.mean(nmse_per_sample(prediction, target))))


def metric_summary(result):
    return {
        "LS": mean_nmse_db(result["ls"], result["true"]),
        "LMMSE": mean_nmse_db(result["lmmse"], result["true"]),
        "Model": mean_nmse_db(result["prediction"], result["true"]),
    }


def tap_project_channels(value, num_taps):
    complex_value = torch.complex(value[:, 0], value[:, 1])
    taps = torch.fft.ifft(complex_value, dim=-1)
    mask = torch.zeros_like(taps)
    mask[..., :num_taps] = 1.0
    projected = torch.fft.fft(taps * mask, dim=-1)
    return torch.stack((projected.real, projected.imag), dim=1).to(value.dtype)


def metric_value(prediction, target):
    return mean_nmse_db(prediction, target)


def snr_curve(prediction, target, snr, edges=None):
    edges = np.arange(0, 35, 5) if edges is None else np.asarray(edges)
    centers = (edges[:-1] + edges[1:]) / 2
    values = nmse_per_sample(prediction, target)
    curve = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (snr >= lower) & (snr < upper)
        curve.append(
            float(nmse_db(torch.mean(values[mask]))) if mask.any() else np.nan
        )
    return centers, curve


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def plot_histories(histories, path):
    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        plt.plot(history["validation"], label=f"{name} validation")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE")
    plt.title("Model Validation Convergence")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_grouped_bars(labels, series, ylabel, title, path):
    x = np.arange(len(labels))
    width = 0.8 / len(series)
    plt.figure(figsize=(8, 5))
    for index, (name, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=name)
    plt.xticks(x, labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_diagnostics(result, path_prefix, model_label):
    index = int(torch.argmin(torch.abs(result["snr"] - 10.0)))
    values = [
        ("True", result["true"][index]),
        ("LS", result["ls"][index]),
        ("LMMSE", result["lmmse"][index]),
        (model_label, result["prediction"][index]),
    ]
    magnitudes = [
        torch.sqrt(value[0] ** 2 + value[1] ** 2).mean(dim=-1).numpy()
        for _, value in values
    ]
    minimum = min(value.min() for value in magnitudes)
    maximum = max(value.max() for value in magnitudes)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, ((title, _), magnitude) in zip(axes.flat, zip(values, magnitudes)):
        image = axis.imshow(
            magnitude, aspect="auto", cmap="viridis", vmin=minimum, vmax=maximum
        )
        axis.set_title(title)
        axis.set_xlabel("TX antenna")
        axis.set_ylabel("RX antenna")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(f"Channel Magnitude at {result['snr'][index]:.1f} dB")
    figure.tight_layout()
    figure.savefig(f"{path_prefix}_heatmap.png", dpi=180)
    plt.close(figure)

    plt.figure(figsize=(8, 5))
    for key, label in (
        ("ls", "LS"),
        ("lmmse", "LMMSE"),
        ("prediction", model_label),
    ):
        residual = (result[key] - result["true"]).flatten().numpy()
        plt.hist(residual, bins=100, density=True, histtype="step", label=label)
    plt.xlabel("Real/imaginary component error")
    plt.ylabel("Density")
    plt.title(f"{model_label} Residual Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{path_prefix}_residual.png", dpi=180)
    plt.close()


def plot_legacy_style_figures(
    result,
    history,
    output_dir,
    scenario_label,
    model_name,
):
    output_dir = Path(output_dir)
    prefix = f"{scenario_label}_{model_name}"
    epochs = np.arange(1, len(history["train"]) + 1)
    lmmse_nmse = float(torch.mean(nmse_per_sample(result["lmmse"], result["true"])))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train"], marker="o", label="Train NMSE")
    plt.plot(
        epochs,
        history["validation"],
        marker="s",
        label="Validation NMSE",
    )
    plt.axhline(
        lmmse_nmse,
        color="red",
        linestyle="--",
        label="Test LMMSE NMSE",
    )
    plt.xlabel("Epoch")
    plt.ylabel("NMSE")
    plt.title(f"{scenario_label} {model_name.upper()} Training")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_LossCurve.png", dpi=180)
    plt.close()

    index = int(torch.argmin(torch.abs(result["snr"] - 10.0)))
    true_value = result["true"][index]
    prediction = result["prediction"][index]
    true_magnitude = torch.sqrt(
        true_value[0] ** 2 + true_value[1] ** 2
    ).mean(dim=-1).numpy()
    prediction_magnitude = torch.sqrt(
        prediction[0] ** 2 + prediction[1] ** 2
    ).mean(dim=-1).numpy()
    minimum = min(true_magnitude.min(), prediction_magnitude.min())
    maximum = max(true_magnitude.max(), prediction_magnitude.max())
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for axis, title, magnitude in (
        (axes[0], "True H", true_magnitude),
        (axes[1], "Predicted H", prediction_magnitude),
    ):
        image = axis.imshow(
            magnitude,
            aspect="auto",
            cmap="viridis",
            vmin=minimum,
            vmax=maximum,
        )
        axis.set_title(title)
        axis.set_xlabel("TX antenna")
        axis.set_ylabel("RX antenna")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(
        f"{scenario_label} {model_name.upper()} at {result['snr'][index]:.1f} dB"
    )
    figure.tight_layout()
    figure.savefig(output_dir / f"{prefix}_Heatmap.png", dpi=180)
    plt.close(figure)

    residual = (result["prediction"] - result["true"]).flatten().numpy()
    plt.figure(figsize=(7, 4.5))
    plt.hist(
        residual,
        bins=100,
        density=True,
        color="steelblue",
        edgecolor="black",
        alpha=0.75,
    )
    plt.xlabel("Real/imaginary component error")
    plt.ylabel("Density")
    plt.title(f"{scenario_label} {model_name.upper()} Residual Distribution")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_ResidualHistogram.png", dpi=180)
    plt.close()


@torch.no_grad()
def benchmark_model(model, sample, device, batch_sizes=(1, 32), runs=200):
    model.eval().to(device)
    h_ls, _, _, noise_variance = sample
    output = {}
    for batch_size in batch_sizes:
        x = h_ls.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).to(device)
        noise = noise_variance.repeat(batch_size).to(device)
        for _ in range(20):
            model(x, noise)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(runs):
            model(x, noise)
        if device.type == "cuda":
            torch.cuda.synchronize()
        output[str(batch_size)] = (time.perf_counter() - start) * 1000 / runs
    return output


def model_metadata(model):
    return {"parameters": count_parameters(model)}
