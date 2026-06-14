# Complete CSI Experiment Suite

The suite uses the physically consistent observation model `Y = H X + N`,
orthogonal DFT pilots, independent train/validation/test sets, and NMSE as the
main metric. No external dataset is required.

## Files

- `synthetic_mimo.py`: i.i.d. Rayleigh and correlated multipath channel data
- `experiment_models.py`: residual CNN, LSTM, and Transformer estimators
- `experiment_utils.py`: shared training, evaluation, metrics, and plotting
- `experiment_args.py`: shared command-line arguments
- `compare_snr.py`: fixed-SNR LS/LMMSE/CNN comparison
- `compare_models.py`: CNN/LSTM/Transformer accuracy and convergence
- `compare_channels.py`: i.i.d. Rayleigh vs correlated multipath channels
- `compare_impairments.py`: IQ imbalance and 4-bit ADC robustness
- `compare_subcarriers.py`: 8/16/32/64 subcarrier sequence comparison
- `compare_dropout.py`: CNN dropout ablation
- `benchmark_architectures.py`: parameter count and inference latency
- `export_onnx.py`: export a trained model with dynamic batch/subcarrier axes
- `run_complete_experiments.py`: run every experiment in sequence
- `generate_complete_figures.py`: generate the 18 Rayleigh/multipath model figures

## Quick Validation

```powershell
python run_complete_experiments.py --device cuda --quick
```

The quick mode only verifies the pipeline. Its two-epoch accuracy values must
not be used as final experimental conclusions.

## Full Experiment

```powershell
python run_complete_experiments.py --device cuda
```

The default full configuration uses 10,000 training samples, 2,000 validation
samples, 3,000 test samples, 20 epochs, and batch size 128 for each experiment.

For a stronger final report:

```powershell
python compare_models.py --device cuda --train-samples 20000 --val-samples 4000 --test-samples 5000 --epochs 30
python compare_snr.py --device cuda --train-samples 20000 --val-samples 4000 --test-samples 5000 --epochs 30
python compare_channels.py --device cuda --train-samples 20000 --val-samples 4000 --test-samples 5000 --epochs 30
python compare_impairments.py --device cuda --train-samples 20000 --val-samples 4000 --test-samples 5000 --epochs 30
python compare_subcarriers.py --device cuda --train-samples 20000 --val-samples 4000 --test-samples 5000 --epochs 30
python compare_dropout.py --device cuda --train-samples 20000 --val-samples 4000 --test-samples 5000 --epochs 30
python benchmark_architectures.py --device cuda --runs 500
```

Export the best CNN after `compare_models.py`:

```powershell
python -m pip install onnx
python export_onnx.py --model cnn --checkpoint results/complete/models/cnn_best.pt --output results/complete/models/cnn.onnx
```

## Scientific Questions

1. SNR comparison: how LS, scalar LMMSE, and CNN change from 0 to 30 dB.
2. Architecture comparison: whether convolution, recurrence, or attention is
   the best accuracy/parameter/latency tradeoff.
3. Channel comparison: whether neural estimation gains come from exploitable
   spatial-frequency correlation.
4. Robustness comparison: how IQ imbalance and low-resolution ADC quantization
   affect clean-trained and impairment-trained models.
5. Sequence comparison: whether longer frequency sequences improve learned
   denoising.
6. Dropout ablation: whether regularization helps this compact CNN.
7. Deployment comparison: batch-1 and batch-32 inference latency.

## Interpretation Rules

- Lower NMSE in dB is better.
- A quick run is only a software test.
- CNN gains on correlated multipath channels are expected because neighboring
  antennas and subcarriers contain shared structure.
- CNN should have limited advantage on fully i.i.d. channels; a large gain
  there would require checking for leakage or an unfair baseline.
- The included LMMSE is a scalar shrinkage baseline. It is not a full
  covariance-aware matrix LMMSE.
- Synthetic results should not be presented as real-world DeepMIMO results.
