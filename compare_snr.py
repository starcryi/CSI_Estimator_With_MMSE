import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiment_args import common_parser
from experiment_models import build_model
from experiment_utils import (
    evaluate_model,
    make_dataset,
    make_loaders,
    mean_nmse_db,
    save_json,
    select_device,
    set_seed,
    train_model,
)
from torch.utils.data import DataLoader


def main():
    parser = common_parser("Evaluate LS, LMMSE, and CNN at fixed SNR values.")
    parser.add_argument(
        "--snr-values", type=float, nargs="+", default=[0, 5, 10, 15, 20, 25, 30]
    )
    args = parser.parse_args()
    output_dir = args.output_dir / "snr"
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    train_loader, val_loader, _ = make_loaders(args)
    model = build_model("cnn", args.num_rx, args.num_tx)
    model, _, _ = train_model(
        model, train_loader, val_loader, args, output_dir, "cnn"
    )
    device = select_device(args.device)
    metrics = {"LS": [], "LMMSE": [], "CNN": []}
    for index, snr in enumerate(args.snr_values):
        dataset = make_dataset(
            args,
            args.test_samples,
            args.seed + 100 + index,
            snr_min_db=snr,
            snr_max_db=snr,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        result = evaluate_model(model, loader, device)
        metrics["LS"].append(mean_nmse_db(result["ls"], result["true"]))
        metrics["LMMSE"].append(mean_nmse_db(result["lmmse"], result["true"]))
        metrics["CNN"].append(
            mean_nmse_db(result["prediction"], result["true"])
        )

    plt.figure(figsize=(8, 5))
    for name, values in metrics.items():
        plt.plot(args.snr_values, values, marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("Fixed-SNR CSI Estimation Comparison")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fixed_snr_comparison.png", dpi=180)
    plt.close()
    save_json(
        output_dir / "metrics.json",
        {"snr_db": args.snr_values, "nmse_db": metrics},
    )
    print(metrics)


if __name__ == "__main__":
    main()
