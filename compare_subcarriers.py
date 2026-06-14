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
    parser = common_parser("Compare different OFDM subcarrier sequence lengths.")
    parser.add_argument("--lengths", type=int, nargs="+", default=[8, 16, 32, 64])
    args = parser.parse_args()
    output_dir = args.output_dir / "subcarriers"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    metrics = {}
    for index, length in enumerate(args.lengths):
        args.num_subcarriers = length
        set_seed(args.seed + index)
        loaders = make_loaders(args)
        model = build_model("cnn", args.num_rx, args.num_tx)
        model, _, _ = train_model(
            model, loaders[0], loaders[1], args, output_dir, f"length_{length}"
        )
        metrics[str(length)] = metric_summary(
            evaluate_model(model, loaders[2], device)
        )

    plot_grouped_bars(
        [str(value) for value in args.lengths],
        {
            method: [metrics[str(length)][method] for length in args.lengths]
            for method in ("LS", "LMMSE", "Model")
        },
        "NMSE (dB)",
        "Subcarrier Length Comparison",
        output_dir / "subcarrier_comparison.png",
    )
    save_json(output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
