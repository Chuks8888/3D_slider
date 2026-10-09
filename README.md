# 3D CT Inference Engine (WORK IN PROGRESS)

A lightweight, memory-bounded sliding-window inference pipeline for volumetric Computed Tomography (CT) scans under strict GPU VRAM constraints (< 4 GB).

## Features

- **Memory-Bounded 3D Inference:** Processes dense volumes in sub-patches without triggering CUDA OOMs.
- **Gaussian-Weighted Stitching:** Suppresses hard boundary seams during patch reassembly.
- **DICOM & NIfTI Ready:** Converts raw DICOM series (via `dcm2niix`) or accepts standardized `.nii.gz`.
- **Pre-calibrated Preprocessing:** Automated Hounsfield Unit (HU) window clipping and isotropic spline resampling.

## Quickstart

### Docker (Recommended)

```bash
# Work in progress

```

### Local Setup

```bash
pip install -r requirements.txt

python -m src.cli \
  --input ./data/scan.nii.gz \
  --output ./data/mask.nii.gz \
  --patch-size 96 96 96

```

## CLI Reference

| Flag           | Default    | Description                                                |
| -------------- | ---------- | ---------------------------------------------------------- |
| `--input`      | _Required_ | Path to `.nii.gz` file or DICOM series directory           |
| `--output`     | _Required_ | Output path for predicted 3D segmentation mask             |
| `--patch-size` | `96 96 96` | 3D window dimensions ($D \times H \times W$)               |
| `--overlap`    | `0.5`      | Fractional overlap between adjacent patches ($0.0$–$0.75$) |
| `--profile`    | `False`    | Exports VRAM usage and execution latency to JSON           |
| `--verbose`    | `False`    | Prints the logs to the standard output                     |
| `--recursive`  | `False`    | Searches for the target data recursively                   |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
