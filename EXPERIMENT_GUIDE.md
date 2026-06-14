# Synthetic MIMO CSI Experiment

This experiment does not require DeepMIMO or any downloaded channel data. It
generates reproducible frequency-selective MIMO channels, orthogonal pilots,
received signals, and AWGN locally.

## Current Machine

- GPU: NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB
- NVIDIA driver: 566.36
- Driver-supported CUDA version: 12.7
- Current Python: 3.12.4 from `D:\conda`
- Current PyTorch: 2.5.1 CPU build

The existing Conda Python cannot load its SSL module. The existing
`gpu-env/.venv` references the user-owned Python installation at
`C:\Users\aikub\AppData\Local\Programs\Python\Python312`. It cannot be
executed from Codex's isolated Windows account, but it should remain usable
from the `aikub` Windows account.

`gpu-env/.venv` already contains PyTorch 2.11.0 with CUDA 12.8, torchvision,
torchaudio, NumPy, CUDA runtime libraries, and cuDNN. It is missing
Matplotlib, which is required by `experiment.py` to create figures.

## Required Packages

Only these packages are required:

```text
torch
numpy
matplotlib
```

The current CPU environment already has all three packages.

## Use the Existing GPU Environment

Run these commands from a normal PowerShell window under the `aikub` account:

```powershell
.\gpu-env\.venv\Scripts\Activate.ps1
python -m pip install matplotlib
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If CUDA availability is `True`, use this environment directly.

## Recreate the GPU Environment If Needed

Only recreate the environment if activation or the CUDA verification above
fails from the `aikub` account. Install an official 64-bit Python 3.12 release
first, then open PowerShell in the project directory and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-experiment.txt
```

The NVIDIA driver is new enough for the CUDA 12.6 PyTorch wheels. A separate
CUDA Toolkit installation is not required for normal PyTorch training.

Verify the environment:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

The second line should be `True`.

## Run the Experiment

Quick CPU check:

```powershell
python experiment.py --train-samples 512 --val-samples 128 --test-samples 256 --epochs 3
```

Recommended GPU experiment:

```powershell
python experiment.py --device cuda --train-samples 20000 --val-samples 4000 --test-samples 5000 --epochs 30 --batch-size 128
```

Reproduce the included CPU result:

```powershell
python experiment.py --train-samples 2000 --val-samples 400 --test-samples 800 --epochs 12 --batch-size 64 --output-dir results/synthetic
```

## Outputs

The experiment writes:

- `best_model.pt`: best validation checkpoint
- `metrics.json`: configuration and test NMSE
- `loss_curve.png`: train and validation NMSE
- `nmse_vs_snr.png`: LS, LMMSE, and CNN comparison
- `channel_heatmap.png`: channel magnitude comparison
- `residual_histogram.png`: estimation error distribution

## Experiment Model

For each subcarrier, the received pilots follow:

```text
Y = H X + N
```

`X` is a unitary DFT pilot matrix. The LS estimate is computed first, and a
small residual 3D CNN refines it using spatial, frequency, and noise-level
information. The synthetic channel is generated from an exponential multipath
power-delay profile, FFT frequency response, and correlated TX/RX antennas.
