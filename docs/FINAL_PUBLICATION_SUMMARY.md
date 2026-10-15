# Final Publication and Research Project Summary

**Project Title:** Robust Classical Stereo Matching Under Photometric Domain Shifts: Benchmark Optimization and Cross-Domain Evaluation in an Unstructured Outdoor Environment  
**Date:** October 15, 2026  
**Research Team:** Advanced Agentic Coding and Computer Vision Group  
**Workspace:** `d:\03_Proyek\SGBM Research`

---

## 1. Executive Summary

This project executed a complete, reproducible research pipeline investigating the robustness, optimization, and cross-domain transfer of classical stereo matching algorithms (OpenCV `StereoBM` and `StereoSGBM`). The investigation spanned two distinct domains:

1. **Domain A (Middlebury Stereo 2014 Benchmark):**
   - Established baseline quantitative metrics on 12 physical scenes using a completeness-aware evaluation protocol that counts invalid predictions as matching failures.
   - Audited photometric robustness across 276 evaluated conditions (brightness scaling, gamma, and Gaussian noise).
   - Executed a staged hyperparameter optimization on 8 Development scenes, reducing completeness-aware bad-2.0 error by 4.86% on StereoBM and 2.20% on StereoSGBM.
   - Evaluated frozen configurations in a strict single-pass verification on held-out benchmark scenes (`Jadeplant`, `Shelves`, `ArtL`), achieving a 31.3% reduction in Mean Absolute Error (MAE) from 4.59 px to 3.16 px.

2. **Domain B (Gunung Unstructured Outdoor Stereo Dataset):**
   - Discovered and audited 4 outdoor volcanic stereo scenes (`puncakTinggi`, `puncakWarung`, `tanggaBiru`, `tanggaKuning`) at $1280 \times 720$ resolution.
   - Audited dual-camera epipolar geometry, uncovering a severe raw vertical correspondence displacement of 28.80 px that caused immediate failure in unrectified matching.
   - Proved that uncalibrated Hartley homographies fail in outdoor environments (P90 error exploding to 66.38 px), whereas calibrated rectification using physical camera parameters ($B = 97.5\text{ mm}$, $f \approx 927.9\text{ px}$) reduced vertical displacement to 2.40 px (and below 1.20 px on secondary scenes).
   - Transferred frozen benchmark configurations across 1280x720, 640x360, and 320x180 resolutions. On native imagery, transferred G1 StereoSGBM reduced photometric warp MAE by 8.1% (10.84 to 9.96 DN).
   - Executed single-variable target-domain adaptations: expanding the search range to $d_{\max}=160\text{ px}$ captured near-field rocks (<0.7 m), while left-right consistency masking doubled disparity edge alignment from 0.061 to 0.129.
   - Generated a calibrated 3D metric point cloud (127,310 points) spanning depths from 0.59 m to 35.00 m.
   - Produced five publication vector illustrations, five empirical SciencePlots figures, a master comparative double-column figure, two 1080p demonstration videos with time-coded subtitles, and a 4-page IEEEtran manuscript compiled to PDF.

---

## 2. Research Tasks Status Matrix

The following table distinguishes tasks that were completed, experimentally tested, independently verified, or designated for future research.

| Task ID | Research Task Description | Category | Method / Tool | Verification Status | Key Quantitative Output |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **T01** | Middlebury 2014 Q-resolution dataset ingestion & PFM decoder | Benchmark Setup | NumPy, OpenCV | **Verified** | 15 scene instances decoded without scaling bias |
| **T02** | Completeness-aware bad-pixel rate metric definition | Methodology | Python/NumPy | **Verified** | Equation: $(N_{\text{err}>T} + N_{\text{miss}}) / N_{\text{GT valid}}$ |
| **T03** | Middlebury baseline evaluation (StereoBM, StereoSGBM) | Baseline Evaluation | OpenCV 4.14 | **Verified** | StereoBM Bad-2: 44.63%; StereoSGBM Bad-2: 30.23% |
| **T04** | Photometric robustness perturbation benchmark (Phase B) | Controlled Study | Asymmetric synth | **Verified** | 276 conditions evaluated; $\sigma=30 \implies \text{Bad-2} > 58\%$ |
| **T05** | Staged hyperparameter tuning on Development split (Phase C) | Optimization | Grid Search | **Verified** | G1 StereoBM: -4.86% Bad-2; G1 StereoSGBM: -2.20% Bad-2 |
| **T06** | Preprocessing filter evaluation (CLAHE, HistMatch, Census) | Preprocessing | Image Processing | **Verified** | HistMatch achieved lowest SGBM Bad-2 (27.31%) |
| **T07** | Freezing optimal parameter configurations | Reproducibility | JSON Manifests | **Verified** | `bm_frozen_config.json`, `sgbm_frozen_config.json` |
| **T08** | Single-pass evaluation on held-out scenes (Milestone 3) | Verification | Zero-leakage pass | **Verified** | StereoSGBM MAE dropped 31.3% (4.59 px to 3.16 px) |
| **T09** | Gunung dataset physical discovery and saturation audit | Dataset Audit | ORB, OpenCV | **Verified** | 4 scenes; 16.8% - 17.8% saturation (intensity $\ge 250$) |
| **T10** | Epipolar correspondence and vertical displacement audit | Geometry Diagnostics| ORB RANSAC | **Verified** | Raw $\Delta y = 28.80\text{ px}$; unrectified stereo invalid |
| **T11** | Epipolar rectification pathways comparison (3 methods) | Calibration | OpenCV Calib | **Verified** | Calibrated $\Delta y = 2.40\text{ px}$; Hartley failed (P90: 66.4 px) |
| **T12** | Cross-domain transfer across 3 resolutions (Milestone 5) | Domain Transfer | Multi-scale | **Verified** | Warp MAE reduced 8.1% (10.84 to 9.96 DN) |
| **T13** | Physical Region-of-Interest (ROI) failure mode analysis | Failure Analysis | Masked Stats | **Verified** | Sky (38.4% cov), Building (44.3% cov), Foliage (5.4% cov) |
| **T14** | Single-variable target-domain adaptation ablation (Phase G) | Adaptation | Isolated Ablation | **Verified** | A4 $d_{\max}=160\text{ px} \implies$ cov 29.5%; A5 LR $\implies$ edge 0.129 |
| **T15** | Metric 3D point cloud generation from reprojection matrix $Q$ | 3D Reconstruction | Reprojection ($Q$) | **Verified** | 127,310 metric points ($Z \in [0.59\text{ m}, 35.00\text{ m}]$) |
| **T16** | Publication vector diagrams generation (Illustrations 1-5) | Scientific Viz | Matplotlib/Vector | **Completed & Tested** | Formats: PDF, SVG, PNG (300 DPI) |
| **T17** | Empirical benchmark figures generation (M4-M8, G10, Master) | Scientific Viz | SciencePlots | **Completed & Tested** | `master_comparative_figure.pdf` (IEEE double-column) |
| **T18** | Demonstration video production (Middlebury & Gunung) | Video Production | FFmpeg, Python | **Completed & Tested** | 1080p, 30fps, H.264, yuv420p, SRT subtitles, thumbnails |
| **T19** | Full IEEEtran manuscript writing and LaTeX compilation | Academic Writing | pdflatex, bibtex | **Completed & Verified** | `paper/main.pdf` (4 pages, double-column, 0 errors) |
| **F01** | Active IMU-guided online epipolar recalibration | Future Research | Kalman Filtering | **Future Work** | Eliminate residual 2.40 px dynamic field shear |
| **F02** | HDR exposure bracketing for midday solar saturation | Future Research | Multi-exposure | **Future Work** | Recover contrast in over-exposed 17% sky/gravel |
| **F03** | Subpixel edge-preserving bilateral cost aggregation | Future Research | Guided Filtering | **Future Work** | Mitigate micro-occlusion bleeding in thin foliage |

---

## 3. Index of Generated Publication Artifacts

### Core Academic Documents
- **IEEE Manuscript Source:** `paper/main.tex`
- **Bibliography Database:** `paper/references.bib`
- **Compiled IEEE Manuscript PDF:** `paper/main.pdf` (4 pages, double-column IEEEtran format)
- **Dataset Audit Report:** `docs/GUNUNG_DATASET_AUDIT.md`
- **Environment & Toolchain Report:** `docs/ENVIRONMENT_REPORT.md`
- **Middlebury Optimization Report:** `docs/MIDDLEBURY_OPTIMIZATION_REPORT.md`
- **Held-Out Benchmark Report:** `docs/MIDDLEBURY_HELDOUT_REPORT.md`
- **Gunung Rectification Report:** `docs/GUNUNG_RECTIFICATION_REPORT.md`
- **Gunung Adaptation Audit:** `docs/GUNUNG_ADAPTATION_AUDIT.md`
- **Reproduction Guide:** `docs/REPRODUCTION_GUIDE.md`
- **Limitations & Integrity Audit:** `docs/LIMITATIONS_AND_INTEGRITY_AUDIT.md`

### Demonstration Videos and Multimedia Assets
- **Middlebury Optimization Video:** `video/final/middlebury_optimization.mp4` (1920x1080, 30fps, 28.0s)
- **Gunung Outdoor Demonstration Video:** `video/final/gunung_stereo_demonstration.mp4` (1920x1080, 30fps, 36.0s)
- **Time-Coded Subtitles:** `video/assets/middlebury_optimization.srt`, `video/assets/gunung_stereo_demonstration.srt`
- **Narration Scripts and Storyboards:** `video/scripts/middlebury_narration_script.md`, `video/scripts/gunung_narration_script.md`
- **High-Resolution Thumbnails:** `video/assets/thumb_middlebury_1080p.png`, `video/assets/thumb_gunung_1080p.png`
- **Video Technical Validation Report:** `video/VIDEO_VALIDATION_REPORT.md`

### Publication Figures and Vector Diagrams
- **Master Comparative Figure:** `figures/ieee/master_comparative_figure.pdf` and `.png`
- **Optimization History (M4):** `figures/ieee/m4_optimization_history.pdf`
- **Robustness Curves (M5):** `figures/ieee/m5_robustness_curves.pdf`
- **Accuracy vs Runtime (M6):** `figures/ieee/m6_accuracy_vs_runtime.pdf`
- **Held-Out Scene Breakdown (M8):** `figures/ieee/m8_heldout_breakdown.pdf`
- **Gunung Consistency vs Runtime (G10):** `figures/ieee/g10_gunung_consistency_runtime.pdf`
- **Vector Pipeline Architecture (Ill. 1):** `figures/illustrations/illustration_1_pipeline.pdf`
- **Vector Epipolar Geometry (Ill. 2):** `figures/illustrations/illustration_2_epipolar.pdf`
- **Vector SGM Path Aggregation (Ill. 3):** `figures/illustrations/illustration_3_sgm_paths.pdf`
- **Vector Cross-Domain Transfer Gap (Ill. 4):** `figures/illustrations/illustration_4_transfer_gap.pdf`
- **Vector Outdoor Failure Taxonomy (Ill. 5):** `figures/illustrations/illustration_5_failure_taxonomy.pdf`

### 3D Metric Point Cloud and Reconstruction Assets
- **Calibrated 3D Point Cloud:** `results/gunung/3d/puncakTinggi_pointcloud.ply` (127,310 points)
- **Calibrated Metric Depth Map:** `results/gunung/3d/puncakTinggi_depth_metric.png`
- **Bird's-Eye View Topography Scatter:** `results/gunung/3d/puncakTinggi_bev_scatter.png`
- **Reconstruction Metadata Manifest:** `results/gunung/3d/reconstruction_metadata.json`
