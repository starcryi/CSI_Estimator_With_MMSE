from experiment_args import common_parser
from experiment_models import build_model
from experiment_utils import (
    benchmark_model,
    make_dataset,
    model_metadata,
    save_json,
    select_device,
)


def main():
    parser = common_parser("Benchmark CNN, LSTM, and Transformer inference.")
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    args = parser.parse_args()
    output_dir = args.output_dir / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    sample = make_dataset(args, 1, args.seed)[0]
    metrics = {}
    for name in ("cnn", "lstm", "transformer"):
        model = build_model(name, args.num_rx, args.num_tx)
        metrics[name] = model_metadata(model)
        metrics[name]["latency_ms"] = benchmark_model(
            model, sample, device, tuple(args.batch_sizes), args.runs
        )
    save_json(output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
