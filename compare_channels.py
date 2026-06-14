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
    args = common_parser("Compare i.i.d. Rayleigh and correlated multipath channels.").parse_args()
    output_dir = args.output_dir / "channels"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    metrics = {}
    for index, channel_type in enumerate(("iid", "multipath")):
        set_seed(args.seed + index)
        loaders = make_loaders(args, channel_type=channel_type)
        model = build_model("cnn", args.num_rx, args.num_tx)
        model, _, best_val = train_model(
            model, loaders[0], loaders[1], args, output_dir, channel_type
        )
        summary = metric_summary(evaluate_model(model, loaders[2], device))
        summary["best_validation_nmse"] = best_val
        metrics[channel_type] = summary

    methods = ("LS", "LMMSE", "Model")
    plot_grouped_bars(
        ["i.i.d. Rayleigh", "Correlated multipath"],
        {
            method: [
                metrics["iid"][method],
                metrics["multipath"][method],
            ]
            for method in methods
        },
        "NMSE (dB)",
        "Channel Scenario Comparison",
        output_dir / "channel_comparison.png",
    )
    save_json(output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
