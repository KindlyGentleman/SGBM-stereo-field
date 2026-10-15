# Gunung Target-Domain Sequential Adaptation Audit

**Date:** 2026-10-15  
**Auditor:** Scientific Experimental Design and Statistics Reviewer  
**Primary Target Scene:** `puncakTinggi` (Outdoor volcanic terrain, 1280x720)  
**Baseline Model:** G1 Transferred StereoSGBM (`blockSize=7, P1=150, P2=1200, unq=10, mode=3WAY`)  

---

## 1. Experimental Protocol and Sequential Intervention Rules

Target-domain adaptation was conducted using a strict single-variable intervention protocol:
1. One isolated component was modified per trial against the unchanged transfer baseline (A0).
2. All evaluations used the identical calibrated rectified stereo pair.
3. Unsupervised diagnostic metrics were evaluated in tandem: valid coverage, left-right consistency, photometric warp residual, and edge alignment.
4. No deep neural networks or fabricated ground-truth depths were introduced.

---

## 2. Systematic Intervention Findings

| Intervention ID | Target Variable | Hypothesis | Valid Coverage | LR Consistency (Valid) | Photometric MAE | Edge Score | Runtime (ms) | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A0_Baseline** | None (Frozen G1) | Frozen Middlebury configuration transferred directly | 27.59% | 63.94% | **9.67 DN** | 0.061 | 68.9 | **Reference** |
| **A1_CLAHE** | Radiometric | Boosts shadow texture in volcanic rocks | 27.45% | 59.28% | 9.84 DN | 0.059 | 109.0 | **HURTS** |
| **A2_HistMatch** | Radiometric | Equalizes sunlight dynamic range mismatch | 28.16% | 63.33% | 10.19 DN | 0.074 | 99.2 | **INCONCLUSIVE** |
| **A3_Window_9** | Window Size | Disambiguates repetitive volcanic gravel | 27.05% | 63.99% | 10.45 DN | 0.071 | 90.3 | **HURTS** |
| **A4_Range_160** | Disparity Range | Captures near-field rocks (<0.6m) | **29.47%** | **67.54%** | **9.64 DN** | **0.066** | 97.7 | **HELPS** |
| **A5_LR_Filter** | Outlier Mask | Rejects erroneous occlusion matches | 17.64% | **100.00%** | 10.17 DN | **0.129** | 79.1 | **HELPS (Precision)** |
| **A6_Bilateral** | Post-Filter | Smooths planar rock surfaces | 27.59% | 53.01% | 10.77 DN | 0.062 | 67.2 | **HURTS** |

---

## 3. Detailed Scientific Rationale

1. **Why Extended Disparity Range (A4) Helps:**
   The physical camera baseline ($B = 97.5\text{ mm}$) and focal length ($f \approx 928\text{ px}$) produce disparities $> 128\text{ px}$ for scene elements closer than $0.7\text{ m}$. With `numDisparities = 128`, close foreground rocks were artificially truncated, forcing the algorithm to match them to distant background points. Expanding to 160 pixels successfully resolved near-field foreground geometry, improving valid coverage from 27.59% to 29.47% and consistency from 63.94% to 67.54%.

2. **Why CLAHE (A1) Hurts:**
   Local contrast enhancement amplifies high-frequency camera sensor noise in the over-exposed sky and specular rock facets, creating false gradient peaks that degrade block matching stability.

3. **Why Left-Right Consistency Filtering (A5) is Essential for 3D Reconstruction:**
   Invalidating pixels failing the 1.5 px round-trip correspondence check removes spurious foliage and sky artifacts, doubling the edge alignment score from 0.061 to 0.129. While total coverage is reduced, the resulting point cloud is clean and structurally trustworthy.
