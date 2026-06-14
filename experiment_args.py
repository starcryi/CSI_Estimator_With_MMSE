import argparse
from pathlib import Path


def add_common_arguments(parser):
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-rx", type=int, default=4)
    parser.add_argument("--num-tx", type=int, default=4)
    parser.add_argument("--num-subcarriers", type=int, default=32)
    parser.add_argument("--num-taps", type=int, default=6)
    parser.add_argument("--snr-min-db", type=float, default=0.0)
    parser.add_argument("--snr-max-db", type=float, default=30.0)
    parser.add_argument("--spatial-correlation", type=float, default=0.6)
    parser.add_argument(
        "--channel-type", choices=("iid", "multipath"), default="multipath"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("results/complete"))
    return parser


def common_parser(description):
    return add_common_arguments(argparse.ArgumentParser(description=description))
