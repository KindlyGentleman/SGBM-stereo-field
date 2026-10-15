# Middlebury 2014 Hyperparameter Optimization Report

**Date:** 2026-10-15  
**Auditor:** Computer Vision and Stereo Geometry Specialist  
**Dataset:** Middlebury Stereo Evaluation v3 (Quarter-Resolution, `trainingQ`)  

---

## 1. Experimental Partitioning Protocol

To prevent information leakage between exposures and geometry variants, scenes were partitioned by physical object identity:
- **Development / Tuning Split (8 instances):** `Adirondack`, `Motorcycle`, `MotorcycleE`, `Pipes`, `Playtable`, `PlaytableP`, `Recycle`, `Vintage`.
- **Validation Split (4 instances):** `Piano`, `PianoL`, `Playroom`, `Teddy`.
- **Held-out Benchmark Verification Split (3 instances):** `Jadeplant`, `Shelves`, `ArtL`.

---

## 2. Optimization Objectives

- **Primary Objective:** Minimize completeness-aware bad-pixel percentage ($> 2.0\text{ px}$ error or invalid prediction) on Development scenes:
  $$\text{Bad-2.0}_{\text{complete}} = \frac{N_{\text{err} > 2.0} + N_{\text{missing}}}{N_{\text{GT valid}}} \times 100\%$$
- **Coverage Constraint:** Valid prediction coverage $\ge 60\%$.
- **Runtime Budget:** Mean execution time $< 150\text{ ms}$ on standard CPU.

---

## 3. Optimization Trajectory and Results

### StereoBM
- **Default Baseline (G0):**
  - Dev Bad-2.0: **43.69%** | Coverage: 61.8% | Mean Runtime: 12.9 ms
- **Optimized Configuration (G1):**
  - Parameters: `blockSize = 9`, `uniquenessRatio = 5`, `preFilterCap = 63`, `textureThreshold = 0`, `speckleWindowSize = 50`, `speckleRange = 32`.
  - Dev Bad-2.0: **38.83%** (**-4.86% absolute improvement**)
  - Dev Coverage: **67.0%** (**+5.2% higher valid density**)
  - Dev Runtime: 14.5 ms

### StereoSGBM
- **Default Baseline (G0):**
  - Dev Bad-2.0: **29.93%** | Coverage: 80.0% | Mean Runtime: 26.0 ms
- **Optimized Configuration (G1):**
  - Parameters: `blockSize = 7`, `P1 = 150`, `P2 = 1200`, `uniquenessRatio = 10`, `speckleWindowSize = 100`, `speckleRange = 2`, `mode = SGBM_3WAY`.
  - Dev Bad-2.0: **27.73%** (**-2.20% absolute improvement**)
  - Dev Coverage: **80.1%**
  - Dev Runtime: 50.9 ms

---

## 4. Photometric Preprocessing Comparison (Development Split)

| Preprocessing Method | StereoBM Bad-2.0 | StereoBM Coverage | StereoSGBM Bad-2.0 | StereoSGBM Coverage | StereoSGBM Runtime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Grayscale Baseline** | 38.82% | 67.0% | 27.73% | 80.1% | 57.3 ms |
| **Histogram Matching** | 39.84% | 65.9% | **27.31%** | **80.5%** | 56.9 ms |
| **Global Normalization** | 39.21% | 66.6% | 27.62% | 80.1% | 60.7 ms |
| **Contrast Normalization**| 40.25% | 65.6% | 27.64% | 80.1% | 69.7 ms |
| **CLAHE** | 40.08% | 65.3% | 28.64% | 78.8% | 55.9 ms |
| **Census Transform (3x3)**| 63.54% | 38.3% | 41.30% | 63.9% | 61.8 ms |

**Finding:** Histogram matching to the reference camera distribution achieved the lowest bad-pixel rate (27.31%) on SGBM by reducing radiometric mismatch between stereo viewpoints.

---

## 5. Frozen Candidate Configurations

The frozen model configurations for cross-domain transfer:
- **G0:** Default baseline OpenCV settings.
- **G1:** Middlebury development-tuned parameters without auxiliary preprocessing.
- **G2:** Middlebury development-tuned parameters with Histogram Matching preprocessing.
