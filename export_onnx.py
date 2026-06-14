import argparse
from pathlib import Path

import torch

from experiment_models import build_model


class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, h_ls, noise_variance):
        return self.model(h_ls, noise_variance)


def main():
    parser = argparse.ArgumentParser(description="Export a trained estimator to ONNX.")
    parser.add_argument("--model", choices=("cnn", "lstm", "transformer"), default="cnn")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/csi_model.onnx"))
    parser.add_argument("--num-rx", type=int, default=4)
    parser.add_argument("--num-tx", type=int, default=4)
    parser.add_argument("--num-subcarriers", type=int, default=32)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    model = build_model(args.model, args.num_rx, args.num_tx)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    wrapper = ExportWrapper(model)
    h_ls = torch.randn(1, 2, args.num_rx, args.num_tx, args.num_subcarriers)
    noise_variance = torch.ones(1) * 0.01
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (h_ls, noise_variance),
        args.output,
        input_names=["h_ls", "noise_variance"],
        output_names=["h_estimate"],
        dynamic_axes={
            "h_ls": {0: "batch", 4: "subcarriers"},
            "noise_variance": {0: "batch"},
            "h_estimate": {0: "batch", 4: "subcarriers"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported {args.model} to {args.output.resolve()}")


if __name__ == "__main__":
    main()
