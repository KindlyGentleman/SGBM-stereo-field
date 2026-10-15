# Complete Research Reproduction Guide

This document specifies the exact environment requirements, directory conventions, and sequential commands to reproduce all experimental findings, optimization stages, cross-domain transfers, publication figures, video demonstrations, and the IEEE manuscript from the repository root.

---

## 1. System Requirements and Environment Setup

### Operating System and Runtimes
- **Host OS:** Windows 10/11 x64 (PowerShell 7+ or Windows PowerShell 5.1).
- **Python:** 3.11, 3.12, or 3.13 (Verified on Python 3.13.7).
- **FFmpeg:** 7.0+ on system PATH (Verified on FFmpeg 7.0.1).
- **LaTeX:** MiKTeX or TeX Live with `pdflatex` and `bibtex` on system PATH.

### Python Environment Setup
In PowerShell from the repository root:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

---

## 2. Dataset Setup and Verification

### Middlebury Stereo 2014 Benchmark (Domain A)
Quarter-resolution evaluation assets reside in `data/middlebury/MiddEval3/`:
- `data/middlebury/MiddEval3/trainingQ/` (15 scene instances)
- `data/middlebury/MiddEval3/testQ/`

To verify scene integrity and ground-truth decoding:
```powershell
python -c "from stereo_robustness.datasets import load_middlebury_scene; l, r, gt, m = load_middlebury_scene('data/middlebury/MiddEval3/trainingQ/Adirondack'); print('Adirondack loaded successfully:', l.shape, gt.shape)"
```

### Gunung Outdoor Stereo Dataset (Domain B)
Stereo captures reside in `D:\02_Riset\Research Paper\source_code\comparison\gunung` with camera calibration parameters in `D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz`.

To audit image resolution, channel integrity, and calibration parameters:
```powershell
python -c "import cv2, numpy as np; cal = np.load(r'D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz'); print('Baseline B (mm):', -cal['T'][0,0]); print('Focal length fx (px):', cal['K1'][0,0])"
```

---

## 3. Step-by-Step Experimental Execution

### Step 1: Middlebury Baseline Evaluation (Milestone 1)
Evaluates default OpenCV StereoBM and StereoSGBM on clean Middlebury training scenes with completeness-aware bad-pixel rates:
```powershell
python -m stereo_robustness.cli evaluate-baseline --config configs/middlebury_baseline.yaml --output results/middlebury/baseline
```
Output files:
- `results/middlebury/baseline/baseline_summary.json`
- `results/middlebury/baseline/baseline_per_scene.csv`

### Step 2: Photometric Robustness Benchmark (Milestone 2 - Phase B)
Applies asymmetric transformations to the right camera (brightness scaling, gamma, Gaussian noise) across 276 evaluated conditions:
```powershell
python -m stereo_robustness.cli benchmark-robustness --config configs/middlebury_robustness.yaml --output results/middlebury/robustness
```
Output files:
- `results/middlebury/robustness/robustness_results.csv`
- `results/middlebury/robustness/robustness_summary.json`

### Step 3: Staged Hyperparameter Tuning (Milestone 2 - Phase C)
Runs grid search across correlation window size, uniqueness ratios, smoothness penalties, and preprocessing filters on Development scenes:
```powershell
python -m stereo_robustness.cli tune --config configs/middlebury_tuning.yaml --output results/middlebury/optimization
```
Locks optimal configurations into:
- `results/middlebury/best_configs/bm_frozen_config.json`
- `results/middlebury/best_configs/sgbm_frozen_config.json`

### Step 4: Strict Held-Out Benchmark Verification (Milestone 3)
Executes a single-pass evaluation on held-out scenes (`Jadeplant`, `Shelves`, `ArtL`):
```powershell
python -m stereo_robustness.cli verify-heldout --config configs/middlebury_heldout.yaml --output results/middlebury/heldout
```
Output files:
- `results/middlebury/heldout/heldout_summary.json`
- `results/middlebury/heldout/heldout_per_scene.csv`

### Step 5: Gunung Geometry and Rectification Diagnostics (Milestone 4)
Computes epipolar correspondence residuals, audits raw vertical displacement, and validates calibrated rectification:
```powershell
python -m stereo_robustness.cli rectify-gunung --config configs/gunung_rectification.yaml --output results/gunung/rectification
```
Output files:
- `results/gunung/rectification/rectification_metrics.json`
- `results/gunung/rectification/puncakTinggi_epipolar_comparison.png`

### Step 6: Cross-Domain Transfer and ROI Failure Analysis (Milestone 5)
Transfers frozen G0, G1, and G2 configurations across 1280x720, 640x360, and 320x180 resolutions with unsupervised consistency metrics:
```powershell
python -m stereo_robustness.cli transfer-gunung --config configs/gunung_transfer.yaml --output results/gunung/transfer
```
Output files:
- `results/gunung/transfer/gunung_transfer_results.csv`
- `results/gunung/transfer/gunung_transfer_summary.json`
- `results/gunung/roi_analysis/puncakTinggi_roi_breakdown.json`

### Step 7: Target-Domain Sequential Adaptation (Milestone 6)
Executes isolated single-variable interventions (A1 CLAHE, A2 Unsharp, A3 Texture boost, A4 Search expansion, A5 LR consistency, A6 Bilateral):
```powershell
python -m stereo_robustness.cli adapt-gunung --config configs/gunung_adaptation.yaml --output results/gunung/adaptation
```
Output files:
- `results/gunung/adaptation/adaptation_results.csv`
- `results/gunung/adaptation/adaptation_summary.json`

### Step 8: Calibrated 3D Metric Reconstruction (Section 11)
Reconstructs metric point cloud from adapted disparity using reprojection matrix $Q$:
```powershell
python -m stereo_robustness.cli reconstruct-3d --disparity results/gunung/adaptation/puncakTinggi_adapted_disp.npy --output results/gunung/3d
```
Output files:
- `results/gunung/3d/puncakTinggi_pointcloud.ply`
- `results/gunung/3d/puncakTinggi_depth_metric.png`
- `results/gunung/3d/puncakTinggi_bev_scatter.png`
- `results/gunung/3d/reconstruction_metadata.json`

---

## 4. Visualization, Video, and Manuscript Generation

### Step 9: Publication Figures Generation (Milestone 7)
Generates IEEE standard vector illustrations and master comparative double-column figure:
```powershell
python -m stereo_robustness.cli generate-figures --config configs/ieee_figures.yaml --output figures/ieee
```
Output files:
- `figures/illustrations/illustration_1_pipeline.{pdf,svg,png}`
- `figures/illustrations/illustration_2_epipolar.{pdf,svg,png}`
- `figures/illustrations/illustration_3_sgm_paths.{pdf,svg,png}`
- `figures/illustrations/illustration_4_transfer_gap.{pdf,svg,png}`
- `figures/illustrations/illustration_5_failure_taxonomy.{pdf,svg,png}`
- `figures/ieee/master_comparative_figure.{pdf,png}`
- `figures/ieee/m4_optimization_history.pdf`
- `figures/ieee/m5_robustness_curves.pdf`
- `figures/ieee/m6_accuracy_vs_runtime.pdf`
- `figures/ieee/m8_heldout_breakdown.pdf`
- `figures/ieee/g10_gunung_consistency_runtime.pdf`

### Step 10: Demonstration Video Production (Milestones 8 & 9)
Renders high-definition 1080p demonstration videos:
```powershell
python -m stereo_robustness.cli render-videos --config configs/video.yaml --output video/final
```
Output files:
- `video/final/middlebury_optimization.mp4` (1920x1080, 30fps, 28.0s)
- `video/final/gunung_stereo_demonstration.mp4` (1920x1080, 30fps, 36.0s)
- `video/assets/middlebury_optimization.srt`
- `video/assets/gunung_stereo_demonstration.srt`
- `video/assets/thumb_middlebury_1080p.png`
- `video/assets/thumb_gunung_1080p.png`

### Step 11: IEEE Manuscript Compilation (Milestone 10)
Compiles LaTeX source, resolves bibliography, and generates final PDF:
```powershell
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
cd ..
```
Output file:
- `paper/main.pdf` (4 pages, double-column IEEEtran format)
