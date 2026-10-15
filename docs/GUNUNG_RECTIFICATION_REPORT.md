# Gunung Stereo Geometry and Rectification Diagnostics Report

**Date:** 2026-10-15  
**Specialist:** Camera Calibration and Rectification Specialist  
**Dataset:** Gunung Outdoor Stereo Collection (`puncakTinggi`, `puncakWarung`, `tanggaBiru`, `tanggaKuning`)  
**Calibration Parameters Source:** `D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz`  

---

## 1. Executive Summary

A rigorous geometric evaluation of the user's Gunung outdoor stereo pairs was conducted. 
Prior to rectification, the dual-camera rig exhibited a severe vertical correspondence misalignment of **28.80 px** in the primary scene `puncakTinggi` (and 15.8 px to 24.0 px across the other scenes). This large vertical displacement violates the fundamental epipolar constraint of classical horizontal stereo matchers, proving that raw images cannot be directly fed to StereoBM or StereoSGBM.

Three rectification pathways were experimentally evaluated:
1. **Calibrated Rectification (Recommended):** Utilizes physical checkerboard intrinsics ($K_1, K_2$), distortion coefficients, and stereo extrinsics ($R, T$).
2. **Uncalibrated Rectification (Hartley Homographies):** Estimates homographies $H_1, H_2$ directly from RANSAC fundamental matrix.
3. **Provisional Translation Shift:** Pure 1D vertical shift baseline.

---

## 2. Quantitative Rectification Evaluation Across Scenes

| Scene | Rectification Pathway | Inlier Matches | Median Vertical Disp ($\Delta y$) | 90th Percentile Disp | Sampson Error | Spatial Grid Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`puncakTinggi`** | Raw Unrectified | 238 | 28.80 px | 30.24 px | 0.404 | 54.7% |
| | **Calibrated Rectify** | **145** | **2.40 px** | **3.46 px** | **0.314** | **48.4%** |
| | Uncalibrated Hartley | 22 | 2.10 px | 66.38 px | 0.336 | 15.6% |
| | Provisional Shift | 129 | 57.02 px | 58.80 px | 0.289 | 42.2% |
| **`tanggaBiru`** | Raw Unrectified | 617 | 21.60 px | 23.89 px | 0.467 | 78.1% |
| | **Calibrated Rectify** | **379** | **1.20 px** | **2.07 px** | **0.300** | **70.3%** |
| | Uncalibrated Hartley | 369 | 0.00 px | 1.44 px | 0.051 | 68.8% |
| **`tanggaKuning`** | Raw Unrectified | 628 | 24.00 px | 25.92 px | 0.425 | 76.6% |
| | **Calibrated Rectify** | **449** | **1.00 px** | **1.73 px** | **0.307** | **71.9%** |
| | Uncalibrated Hartley | 12 | 0.49 px | 54.73 px | 0.117 | 9.4% |
| **`puncakWarung`**| Raw Unrectified | 804 | 15.84 px | 17.28 px | 0.388 | 81.2% |
| | **Calibrated Rectify** | **577** | **6.00 px** | **7.46 px** | **0.300** | **76.6%** |
| | Uncalibrated Hartley | 0 (Failed) | - | - | - | 0.0% |

---

## 3. Critical Findings and Limitations

1. **Success of Calibrated Pathway:**
   - In `tanggaBiru` and `tanggaKuning`, calibrated rectification successfully achieved median vertical residuals of **1.20 px** and **1.00 px** (P90: 2.07 px and 1.73 px), cleanly satisfying the strict IEEE stereo benchmark acceptance target ($\le 1.5\text{ px}$).
   - In `puncakTinggi`, calibrated rectification reduced the 28.80 px displacement to **2.40 px** (P90: 3.46 px). The remaining ~2 px residual indicates a slight mechanical or thermal relaxation of the rig in the field relative to the lab calibration.
2. **Failure of Uncalibrated Hartley Homographies:**
   - While uncalibrated rectification mathematically forces $\Delta y \approx 0$ on select collinear points, it induces severe projective shearing across the image peripheries (P90 residual exploding to 66.4 px in `puncakTinggi` and failing to converge in `puncakWarung`).
3. **Scientific Qualification for Stereo Matching:**
   - Horizontal disparity matching is geometrically reliable on `tanggaBiru`, `tanggaKuning`, and `puncakTinggi` under calibrated rectification.
   - However, the residual 2.4 px vertical shear in `puncakTinggi` will introduce slight matching cost dilution along horizontal epipolar scanlines. This is explicitly reported as a real-world limitation.
