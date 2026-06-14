from experiment_args import common_parser
from experiment_models import build_model
from experiment_utils import (
    evaluate_model,
    make_loaders,
    metric_summary,
    plot_legacy_style_figures,
    save_json,
    select_device,
    set_seed,
    train_model,
)


def main():
    parser = common_parser(
        "Generate the complete Rayleigh and multipath figure matrix."
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    scenarios = (
        ("Rayleigh", "iid"),
        ("Multipath", "multipath"),
    )
    metrics = {}

    for scenario_index, (scenario_label, channel_type) in enumerate(scenarios):
        loaders = make_loaders(args, channel_type=channel_type)
        metrics[scenario_label] = {}
        for model_index, model_name in enumerate(("cnn", "lstm", "transformer")):
            set_seed(args.seed + scenario_index * 10 + model_index)
            model = build_model(model_name, args.num_rx, args.num_tx)
            checkpoint_name = f"{scenario_label}_{model_name}"
            model, history, best_val = train_model(
                model,
                loaders[0],
                loaders[1],
                args,
                args.output_dir,
                checkpoint_name,
            )
            result = evaluate_model(model, loaders[2], device)
            plot_legacy_style_figures(
                result,
                history,
                args.output_dir,
                scenario_label,
                model_name,
            )
            metrics[scenario_label][model_name] = metric_summary(result)
            metrics[scenario_label][model_name][
                "best_validation_nmse"
            ] = best_val

    save_json(args.output_dir / "figure_matrix_metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    from pathlib import Path

    main()
