import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run the complete CSI experiment suite.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="results/complete")
    args = parser.parse_args()
    common = [
        "--device",
        args.device,
        "--output-dir",
        args.output_dir,
    ]
    if args.quick:
        common += [
            "--train-samples",
            "512",
            "--val-samples",
            "128",
            "--test-samples",
            "256",
            "--epochs",
            "2",
            "--batch-size",
            "64",
        ]
    scripts = [
        "compare_snr.py",
        "compare_models.py",
        "compare_channels.py",
        "compare_impairments.py",
        "compare_subcarriers.py",
        "compare_dropout.py",
        "benchmark_architectures.py",
    ]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for script in scripts:
        print(f"\nRunning {script}")
        subprocess.run([sys.executable, script, *common], check=True)


if __name__ == "__main__":
    main()
