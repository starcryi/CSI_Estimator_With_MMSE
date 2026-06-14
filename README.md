# MIMO CSI Estimation Experiments

This repository contains a self-contained CSI estimation experiment suite for
the local RTX 4070 GPU environment. It does not require DeepMIMO or downloaded
channel data.

## Environment

Activate the existing environment:

```powershell
.\gpu-env\.venv\Scripts\Activate.ps1
```

Install plotting support if it is not already installed:

```powershell
python -m pip install matplotlib
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Run

Quick pipeline check:

```powershell
python run_complete_experiments.py --device cuda --quick
```

Full experiment suite:

```powershell
python run_complete_experiments.py --device cuda
```

The detailed experiment list and individual commands are in
`COMPLETE_EXPERIMENTS.md`.

## Main Files

- `synthetic_mimo.py`: synthetic MIMO channel and received-signal generation
- `experiment_models.py`: CNN, LSTM, and Transformer estimators
- `experiment_utils.py`: shared training, evaluation, metrics, and plotting
- `compare_snr.py`: LS/LMMSE/CNN comparison across SNR
- `compare_models.py`: CNN/LSTM/Transformer comparison
- `compare_channels.py`: i.i.d. and correlated multipath comparison
- `compare_impairments.py`: IQ imbalance and 4-bit ADC comparison
- `compare_subcarriers.py`: subcarrier-length comparison
- `compare_dropout.py`: dropout ablation
- `benchmark_architectures.py`: inference latency and parameter counts
- `export_onnx.py`: ONNX export
- `run_complete_experiments.py`: complete experiment runner

## Outputs

Results are written below `results/complete`. Lower NMSE in dB is better.

The experiments use:

```text
Y = H X + N
```

with orthogonal DFT pilots, independent train/validation/test sets, LS and
scalar LMMSE baselines, and residual neural estimators.
