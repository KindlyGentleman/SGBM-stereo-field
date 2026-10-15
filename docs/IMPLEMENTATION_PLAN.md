# Stereo Robustness Research Implementation Plan

**Title:** Robust Classical Stereo Matching Under Photometric Domain Shifts: Benchmark Optimization and Cross-Domain Evaluation in an Unstructured Outdoor Environment  
**Execution Order:** Sequential acceptance gates strictly enforced.  

---

## Acceptance Gates and Milestones

```
+-------------------------------------------------------------+
| Milestone 0: Workspace and Data Inspection (COMPLETED)      |
| Verified Python 3.13, OpenCV 4.14, FFmpeg, MiKTeX, Datasets |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Milestone 1: Middlebury Baseline (Phase A)                  |
| - Evaluate OpenCV StereoBM and StereoSGBM on clean dev data |
| - Measure completeness-aware bad-pixel rates, EPE, coverage |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Milestone 2: Middlebury Photometric Robustness & Tuning     |
| - Phase B: Brightness scaling, Gamma, Gaussian noise        |
| - Phase C: Bounded grid/Optuna optimization on Dev set      |
| - Preprocessing comparison: CLAHE, HistMatch, Norm, Census  |
| - Validation on independent Val scenes                      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Milestone 3: Held-out Middlebury Evaluation                 |
| - Freeze G0 (Default), G1 (Optimized), G2 (+Preprocessing)  |
| - Single evaluation pass on held-out benchmark scenes       |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Milestone 4: Gunung Camera Geometry Diagnostics (Phase D)   |
| - Feature correspondence, vertical displacement (dy) check  |
| - Calibrated vs Uncalibrated vs Provisional Rectification   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Milestone 5: Gunung Transfer Experiment (Phase E & F)       |
| - Transfer frozen G0, G1, G2 models to Gunung scenes        |
| - Unsupervised diagnostics: Left-Right consistency, Warp    |
|   photometric residual, edge alignment                      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Milestone 6: Optional Target-Domain Adaptation (Phase G)    |
| - Test sequential isolated interventions on target scenes   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Milestone 7: IEEE Publication Figures                       |
| - M1-M9 Middlebury figures, G1-G12 Gunung figures, Master   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Milestone 8 & 9: Scientific Demonstration Videos            |
| - Video A: Middlebury optimization (1080p MP4)              |
| - Video B: Gunung real-world demonstration (1080p MP4)       |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
| Milestone 10: IEEE Manuscript & Reproducibility Audit       |
| - Complete LaTeX paper, compiled PDF, audit certificates   |
+-------------------------------------------------------------+
```
