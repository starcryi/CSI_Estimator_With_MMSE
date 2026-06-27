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
    parser = common_parser("Compare original CNN and enhanced residual CNN.")
    parser.set_defaults(channel_type="multipath")
    args = parser.parse_args()
    output_dir = args.output_dir / "improved"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    loaders = make_loaders(args, channel_type="multipath")
    device = select_device(args.device)
    histories = {}
    metrics = {}
    curves = {}

    for label, model_name in (("Original CNN", "cnn"), ("Enhanced CNN", "enhanced_cnn")):
        set_seed(args.seed)
        model = build_model(model_name, args.num_rx, args.num_tx)
        key = label.lower().replace(" ", "_")
        model, history, best_val = train_model(
            model, loaders[0], loaders[1], args, output_dir, key
        )
        result = evaluate_model(model, loaders[2], device)
        plot_diagnostics(result, output_dir / key, label)
        summary = metric_summary(result)
        summary.update(model_metadata(model))
        summary["best_validation_nmse"] = best_val
        metrics[label] = summary
        histories[label] = history
        curves[label] = snr_curve(result["prediction"], result["true"], result["snr"])

    plot_histories(histories, output_dir / "improved_convergence.png")
    plt.figure(figsize=(8, 5))
    for label, (snr, values) in curves.items():
        plt.plot(snr, values, marker="o", label=label)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("Original CNN vs Enhanced CNN")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "improved_nmse_vs_snr.png", dpi=180)
    plt.close()
    save_json(output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
