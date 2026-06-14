from experiment_args import common_parser
from experiment_models import build_model
from experiment_utils import (
    evaluate_model,
    make_loaders,
    metric_summary,
    plot_grouped_bars,
    save_json,
    select_device,
    set_seed,
    train_model,
)


def main():
    args = common_parser("Evaluate IQ imbalance and ADC quantization robustness.").parse_args()
    output_dir = args.output_dir / "impairments"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    scenarios = {
        "clean": {},
        "iq": {"iq_imbalance": True},
        "4bit": {"quantization_bits": 4},
        "iq_4bit": {"iq_imbalance": True, "quantization_bits": 4},
    }

    set_seed(args.seed)
    clean_loaders = make_loaders(args)
    clean_model = build_model("cnn", args.num_rx, args.num_tx)
    clean_model, _, _ = train_model(
        clean_model, clean_loaders[0], clean_loaders[1], args, output_dir, "clean"
    )

    robust_loaders = make_loaders(
        args, iq_imbalance=True, quantization_bits=4
    )
    robust_model = build_model("cnn", args.num_rx, args.num_tx)
    robust_model, _, _ = train_model(
        robust_model,
        robust_loaders[0],
        robust_loaders[1],
        args,
        output_dir,
        "robust",
    )

    metrics = {}
    for name, overrides in scenarios.items():
        test_loader = make_loaders(args, **overrides)[2]
        clean_result = evaluate_model(clean_model, test_loader, device)
        robust_result = evaluate_model(robust_model, test_loader, device)
        metrics[name] = {
            "LS": metric_summary(clean_result)["LS"],
            "LMMSE": metric_summary(clean_result)["LMMSE"],
            "clean_trained_cnn": metric_summary(clean_result)["Model"],
            "robust_trained_cnn": metric_summary(robust_result)["Model"],
        }

    plot_grouped_bars(
        list(scenarios),
        {
            method: [metrics[name][method] for name in scenarios]
            for method in (
                "LS",
                "LMMSE",
                "clean_trained_cnn",
                "robust_trained_cnn",
            )
        },
        "NMSE (dB)",
        "Hardware Impairment Robustness",
        output_dir / "impairment_comparison.png",
    )
    save_json(output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
