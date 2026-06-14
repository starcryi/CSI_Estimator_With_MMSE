from experiment_args import common_parser
from experiment_models import build_model
from experiment_utils import (
    evaluate_model,
    make_loaders,
    metric_summary,
    plot_histories,
    save_json,
    select_device,
    set_seed,
    train_model,
)


def main():
    args = common_parser("CNN dropout ablation experiment.").parse_args()
    output_dir = args.output_dir / "dropout"
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = make_loaders(args)
    device = select_device(args.device)
    histories = {}
    metrics = {}
    for label, dropout in (("no_dropout", 0.0), ("dropout_0.2", 0.2)):
        set_seed(args.seed)
        model = build_model(
            "cnn", args.num_rx, args.num_tx, dropout=dropout
        )
        model, history, best_val = train_model(
            model, loaders[0], loaders[1], args, output_dir, label
        )
        metrics[label] = metric_summary(
            evaluate_model(model, loaders[2], device)
        )
        metrics[label]["best_validation_nmse"] = best_val
        histories[label] = history
    plot_histories(histories, output_dir / "dropout_convergence.png")
    save_json(output_dir / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
