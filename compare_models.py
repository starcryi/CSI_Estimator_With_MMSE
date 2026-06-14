from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_args import common_parser
from experiment_models import build_model
from experiment_utils import (
    evaluate_model,
    make_loaders,
    metric_summary,
    model_metadata,
    plot_diagnostics,
    plot_histories,
    save_json,
    select_device,
    set_seed,
    snr_curve,
    train_model,
)


def main():
    args = common_parser("Compare CNN, LSTM, and Transformer estimators.").parse_args()
    output_dir = args.output_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_loader, val_loader, test_loader = make_loaders(args)
    device = select_device(args.device)
    histories = {}
    metrics = {}
    curves = {}

    for name in ("cnn", "lstm", "transformer"):
        set_seed(args.seed)
        model = build_model(name, args.num_rx, args.num_tx)
        model, history, best_val = train_model(
            model, train_loader, val_loader, args, output_dir, name
        )
        result = evaluate_model(model, test_loader, device)
        plot_diagnostics(result, output_dir / name, name.upper())
        summary = metric_summary(result)
        summary.update(model_metadata(model))
        summary["best_validation_nmse"] = best_val
        metrics[name] = summary
        histories[name] = history
        curves[name] = snr_curve(
            result["prediction"], result["true"], result["snr"]
        )

    plot_histories(histories, output_dir / "model_convergence.png")
    plt.figure(figsize=(8, 5))
    for name, (snr, values) in curves.items():
        plt.plot(snr, values, marker="o", label=name.upper())
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("CNN vs LSTM vs Transformer")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_nmse_vs_snr.png", dpi=180)
    plt.close()
    save_json(output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
