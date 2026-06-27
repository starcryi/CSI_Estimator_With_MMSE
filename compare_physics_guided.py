import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_args import common_parser
from experiment_models import build_model
from experiment_utils import (
    evaluate_model,
    make_loaders,
    metric_value,
    save_json,
    select_device,
    set_seed,
    snr_curve,
    tap_project_channels,
    train_model,
)


def main():
    parser = common_parser("Validate PDP-aware tap projection for multipath CSI.")
    parser.set_defaults(channel_type="multipath")
    args = parser.parse_args()
    output_dir = args.output_dir / "physics_guided"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_loader, val_loader, test_loader = make_loaders(
        args, channel_type="multipath"
    )
    device = select_device(args.device)

    model = build_model("cnn", args.num_rx, args.num_tx)
    model, history, best_val = train_model(
        model, train_loader, val_loader, args, output_dir, "cnn"
    )
    result = evaluate_model(model, test_loader, device)
    physics_model = build_model("physics_cnn", args.num_rx, args.num_tx)
    physics_model, physics_history, physics_best_val = train_model(
        physics_model,
        train_loader,
        val_loader,
        args,
        output_dir,
        "physics_cnn",
    )
    physics_result = evaluate_model(physics_model, test_loader, device)
    target = result["true"]
    estimates = {
        "LS": result["ls"],
        "LMMSE": result["lmmse"],
        "CNN": result["prediction"],
        "LS + Tap Projection": tap_project_channels(result["ls"], args.num_taps),
        "LMMSE + Tap Projection": tap_project_channels(
            result["lmmse"], args.num_taps
        ),
        "CNN + Tap Projection": tap_project_channels(
            result["prediction"], args.num_taps
        ),
        "Physics-Guided CNN": physics_result["prediction"],
    }
    metrics = {name: metric_value(value, target) for name, value in estimates.items()}
    metrics["best_validation_nmse"] = best_val
    metrics["physics_guided_best_validation_nmse"] = physics_best_val

    plt.figure(figsize=(9, 5))
    labels = list(estimates)
    values = [metrics[label] for label in labels]
    colors = ["#7AA6C2", "#F2A65A", "#5DAE61", "#3D6F8E", "#D47C28", "#248A3D"]
    plt.bar(np.arange(len(labels)), values, color=colors)
    plt.xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    plt.ylabel("NMSE (dB)")
    plt.title("Physics-Guided Tap Projection Improves Multipath CSI Estimation")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "physics_guided_bar.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    for label in ("LS", "LMMSE", "CNN", "CNN + Tap Projection", "Physics-Guided CNN"):
        snr, curve = snr_curve(estimates[label], target, result["snr"])
        plt.plot(snr, curve, marker="o", label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("SNR Curve With PDP-Aware Tap Projection")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "physics_guided_snr.png", dpi=180)
    plt.close()

    residuals = {
        "CNN": result["prediction"] - target,
        "CNN + Tap Projection": estimates["CNN + Tap Projection"] - target,
        "Physics-Guided CNN": estimates["Physics-Guided CNN"] - target,
    }
    plt.figure(figsize=(8, 5))
    for label, residual in residuals.items():
        plt.hist(
            residual.flatten().numpy(),
            bins=100,
            density=True,
            histtype="step",
            label=label,
        )
    plt.xlabel("Real/imaginary component error")
    plt.ylabel("Density")
    plt.title("Residual Distribution Before and After Tap Projection")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "physics_guided_residual.png", dpi=180)
    plt.close()

    save_json(output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
