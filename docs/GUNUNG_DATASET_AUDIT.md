# Gunung Outdoor Stereo Dataset Audit

**Audit Date:** 2026-10-15  
**Auditor:** Research Reproducibility Auditor & Camera Calibration Specialist  
**Workspace:** `d:\03_Proyek\SGBM Research`  
**Dataset Primary Source:** `D:\02_Riset\Research Paper\source_code\comparison\gunung`  

---

## 1. Executive Summary

An exhaustive filesystem discovery and file integrity audit was performed on the user's Gunung stereo image collection. 
The directory contains four independent outdoor scenes captured in an unstructured volcanic/mountain environment:
1. `puncakTinggi`
2. `puncakWarung`
3. `tanggaBiru`
4. `tanggaKuning`

All four scenes have synchronized left and right image pairs at **1280 x 720** resolution.
No physical ground truth depth/disparity exists for the stereo pairs in this domain (unlabeled outdoor real-world domain). 
A secondary Intel RealSense sensor was mounted concurrently, storing 640 x 480 depth maps, but without co-planar optical alignment or shared optical axes, RealSense data cannot be treated as pixel-level ground truth for the stereo rig without verified cross-sensor recalibration.

---

## 2. Resolved Paths and Directory Inventory

| Scene Identifier | Left Image Path | Right Image Path | File Size Left (bytes) | File Size Right (bytes) |
| :--- | :--- | :--- | :--- | :--- |
| `puncakTinggi` | `.../puncakTinggi/stereo/left_0000.png` | `.../puncakTinggi/stereo/right_0000.png` | 1,559,485 | 1,707,935 |
| `puncakWarung` | `.../puncakWarung/stereo/left_0000.png` | `.../puncakWarung/stereo/right_0000.png` | 917,337 | 1,027,584 |
| `tanggaBiru` | `.../tanggaBiru/stereo/left_0000.png` | `.../tanggaBiru/stereo/right_0000.png` | 1,667,320 | 1,804,843 |
| `tanggaKuning` | `.../tanggaKuning/stereo/left_0000.png` | `.../tanggaKuning/stereo/right_0000.png` | 1,601,302 | 1,678,886 |

All files were verified intact with valid PNG chunk headers and zero missing bytes.

---

## 3. Cryptographic Hashes (SHA-256 Integrity Verification)

| Scene | Camera | SHA-256 Checksum |
| :--- | :--- | :--- |
| `puncakTinggi` | Left | `8790b7babe56f16c0cc63d1751b67799bb4c50189897aef961463266199282a3` |
| `puncakTinggi` | Right | `6800e39f53f235620b45e8821dbdaeeef6787c7a27552ede8a389bf9c12f7d09` |
| `puncakWarung` | Left | `34a3aa38850df630cab42e7e8120d35c46035ca5d6319752332af2ffebe12499` |
| `puncakWarung` | Right | `329f9f1368160c2bbc42315c0ccd0bcd28b117ab8cf83a3fb1a7b523d3ef74c1` |
| `tanggaBiru` | Left | `a766926a175be2425228a39ccb16a7376f034f83a98c78e41f724be6d091b813` |
| `tanggaBiru` | Right | `3c99319dfc27f10e4dbcd0c6e135b4fe73e362df9ecf5a9ce56b79ce08c9dc13` |
| `tanggaKuning` | Left | `93c855d893205af0b60d5fc522291104ca83fa1bcc9f8ec12d37f67a946efcfd` |
| `tanggaKuning` | Right | `a70e96de58f33531d25b005265c63f1be230c08219975447681e61a14ebd2119` |

---

## 4. Photometric Analysis and Saturated Regions

Outdoor mountain scenes exhibit severe dynamic range challenges due to direct sunlight and bright skies:

| Scene | Left Mean Intensity | Right Mean Intensity | Near-Saturated Pixels (>= 250) Left | Near-Saturated Pixels (>= 250) Right | Under-Exposed (0) Left / Right |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `puncakTinggi` | 137.79 | 131.43 | **16.82%** | **17.83%** | 0.00% / 0.01% |
| `puncakWarung` | 132.49 | 123.82 | **0.00%** | **0.00%** | 0.00% / 0.00% |
| `tanggaBiru` | 124.21 | 128.77 | **2.90%** | **2.07%** | 0.00% / 0.00% |
| `tanggaKuning` | 145.89 | 143.17 | **9.14%** | **9.97%** | 0.00% / 0.00% |

**Key Finding:** `puncakTinggi` contains ~17.3% near-saturated pixels (primarily sky and reflective rocks), creating a severe low-texture and over-exposure failure mode for classical block matchers.

---

## 5. Camera Calibration and Geometry Discovery

A discovery scan across the project directories identified the physical stereo calibration file:
- **Location:** `D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz`
- **Checkerboard Configuration:** 8x6 corners, 30.0 mm square size
- **Left Camera Matrix ($K_1$):**
  $$K_1 = \begin{bmatrix} 927.93 & 0 & 619.50 \\ 0 & 923.38 & 374.67 \\ 0 & 0 & 1 \end{bmatrix}$$
- **Right Camera Matrix ($K_2$):**
  $$K_2 = \begin{bmatrix} 940.82 & 0 & 613.24 \\ 0 & 937.20 & 360.11 \\ 0 & 0 & 1 \end{bmatrix}$$
- **Baseline:** $T_x = -97.40\text{ mm}$, $T_y = 0.96\text{ mm}$, $T_z = -4.27\text{ mm}$ (approx. **97.5 mm** baseline).
- **Distortion:** Both cameras have 5-parameter radial/tangential distortion vectors (`dist_left`, `dist_right`).

---

## 6. Evaluation Protocol and Integrity Declarations

1. **Absence of Ground Truth:** No ground truth depth is fabricated or assumed for Gunung.
2. **Evaluation Metrics:** Field evaluation relies on:
   - Left-right round-trip consistency ($|d_L - d_R| \le 1.5\text{ px}$)
   - Valid disparity coverage percentage
   - Photometric correspondence warp residual
   - Disparity edge alignment with natural intensity boundaries
3. **Domain Transfer Protocol:** Middlebury configurations are frozen prior to application on Gunung to ensure scientific validity.
