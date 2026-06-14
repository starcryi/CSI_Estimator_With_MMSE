import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import ResidualCSINet3D
from synthetic_mimo import (
    SyntheticMIMODataset,
    count_parameters,
    lmmse_from_ls,
    nmse_db,
    nmse_per_sample,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a residual CNN CSI estimator."
    )
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-rx", type=int, default=4)
    parser.add_argument("--num-tx", type=int, default=4)
    parser.add_argument("--num-subcarriers", type=int, default=32)
    parser.add_argument("--num-taps", type=int, default=6)
    parser.add_argument("--snr-min-db", type=float, default=0.0)
    parser.add_argument("--snr-max-db", type=float, default=30.0)
    parser.add_argument("--spatial-correlation", type=float, default=0.6)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic"))
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but this PyTorch build has no CUDA support.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def make_dataset(args, samples, seed):
    return SyntheticMIMODataset(
        num_samples=samples,
        num_rx=args.num_rx,
        num_tx=args.num_tx,
        num_subcarriers=args.num_subcarriers,
        num_taps=args.num_taps,
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        spatial_correlation=args.spatial_correlation,
        seed=seed,
    )


def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []
    for h_ls, h_true, _, noise_variance in loader:
        h_ls = h_ls.to(device)
        h_true = h_true.to(device)
        noise_variance = noise_variance.to(device)
        prediction = model(h_ls, noise_variance)
        loss = torch.mean(nmse_per_sample(prediction, h_true))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def validation_loss(model, loader, device):
    model.eval()
    losses = []
    for h_ls, h_true, _, noise_variance in loader:
        h_ls = h_ls.to(device)
        h_true = h_true.to(device)
        noise_variance = noise_variance.to(device)
        prediction = model(h_ls, noise_variance)
        losses.append(torch.mean(nmse_per_sample(prediction, h_true)).item())
    return float(np.mean(losses))


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    collected = {"ls": [], "lmmse": [], "cnn": [], "true": [], "snr": []}
    for h_ls, h_true, snr_db, noise_variance in loader:
        h_ls_device = h_ls.to(device)
        prediction = model(h_ls_device, noise_variance.to(device)).cpu()
        lmmse = lmmse_from_ls(h_ls, noise_variance)
        collected["ls"].append(h_ls)
        collected["lmmse"].append(lmmse)
        collected["cnn"].append(prediction)
        collected["true"].append(h_true)
        collected["snr"].append(snr_db)
    return {key: torch.cat(value) for key, value in collected.items()}


def mean_nmse_db(prediction, target):
    mean_nmse = torch.mean(nmse_per_sample(prediction, target))
    return float(nmse_db(mean_nmse))


def plot_losses(train_losses, val_losses, output_dir):
    plt.figure(figsize=(7, 4.5))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker="o", label="Train NMSE")
    plt.plot(epochs, val_losses, marker="s", label="Validation NMSE")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE")
    plt.title("Residual CNN Training")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close()


def plot_snr_curve(predictions, output_dir, snr_min, snr_max):
    bins = np.arange(np.floor(snr_min), np.ceil(snr_max) + 5, 5)
    centers = (bins[:-1] + bins[1:]) / 2
    plt.figure(figsize=(7, 4.5))
    for name, label in (("ls", "LS"), ("lmmse", "LMMSE"), ("cnn", "Residual CNN")):
        values = nmse_per_sample(predictions[name], predictions["true"])
        curve = []
        for lower, upper in zip(bins[:-1], bins[1:]):
            mask = (predictions["snr"] >= lower) & (predictions["snr"] < upper)
            curve.append(float(nmse_db(torch.mean(values[mask]))) if mask.any() else np.nan)
        plt.plot(centers, curve, marker="o", label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("CSI Estimation Across SNR")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "nmse_vs_snr.png", dpi=180)
    plt.close()


def plot_heatmap(predictions, output_dir):
    index = int(torch.argmin(torch.abs(predictions["snr"] - 10.0)))
    arrays = [
        ("True", predictions["true"][index]),
        ("LS", predictions["ls"][index]),
        ("LMMSE", predictions["lmmse"][index]),
        ("Residual CNN", predictions["cnn"][index]),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, (title, value) in zip(axes.flat, arrays):
        magnitude = torch.sqrt(value[0] ** 2 + value[1] ** 2).mean(dim=-1)
        image = axis.imshow(magnitude.numpy(), aspect="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("TX antenna")
        axis.set_ylabel("RX antenna")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(f"Mean Channel Magnitude at {predictions['snr'][index]:.1f} dB")
    figure.tight_layout()
    figure.savefig(output_dir / "channel_heatmap.png", dpi=180)
    plt.close(figure)


def plot_residuals(predictions, output_dir):
    target = predictions["true"]
    plt.figure(figsize=(7, 4.5))
    for name, label in (("ls", "LS"), ("lmmse", "LMMSE"), ("cnn", "Residual CNN")):
        residual = (predictions[name] - target).flatten().numpy()
        plt.hist(residual, bins=100, density=True, histtype="step", label=label)
    plt.xlabel("Real/imaginary component error")
    plt.ylabel("Density")
    plt.title("Estimation Residual Distribution")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "residual_histogram.png", dpi=180)
    plt.close()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_set = make_dataset(args, args.train_samples, args.seed)
    val_set = make_dataset(args, args.val_samples, args.seed + 1)
    test_set = make_dataset(args, args.test_samples, args.seed + 2)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    model = ResidualCSINet3D(args.hidden_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    checkpoint_path = args.output_dir / "best_model.pt"
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print(
        f"Device: {device} | Parameters: {count_parameters(model):,} | "
        f"Train/val/test: {len(train_set)}/{len(val_set)}/{len(test_set)}"
    )
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = validation_loss(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train NMSE {train_loss:.6f} | val NMSE {val_loss:.6f}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = collect_predictions(model, test_loader, device)
    metrics = {
        "device": str(device),
        "parameters": count_parameters(model),
        "best_validation_nmse": best_val_loss,
        "test_nmse_db": {
            "LS": mean_nmse_db(predictions["ls"], predictions["true"]),
            "LMMSE": mean_nmse_db(predictions["lmmse"], predictions["true"]),
            "Residual CNN": mean_nmse_db(predictions["cnn"], predictions["true"]),
        },
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    plot_losses(train_losses, val_losses, args.output_dir)
    plot_snr_curve(predictions, args.output_dir, args.snr_min_db, args.snr_max_db)
    plot_heatmap(predictions, args.output_dir)
    plot_residuals(predictions, args.output_dir)
    print(json.dumps(metrics["test_nmse_db"], indent=2))
    print(f"Saved checkpoint, metrics, and figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
