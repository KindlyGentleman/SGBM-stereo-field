# Video Demonstrations Storyboard & Production Guide

**Project:** Robust Classical Stereo Matching Under Photometric Domain Shifts  
**Specifications:** 1920x1080 Full HD, 30.0 fps, H.264 Codec, yuv420p Color Space, Professional Slate Navy Theme  

---

## Video A: Middlebury Optimization
**File:** `video/final/middlebury_optimization.mp4`  
**Duration:** 28 seconds (840 frames)  

| Scene | Timestamp | Topic | Visual Content | Core Metric |
| :--- | :--- | :--- | :--- | :--- |
| **1** | 00:00 - 00:03 | Title Slate | High-contrast publication title card | Title & affiliations |
| **2** | 00:03 - 00:07 | Benchmark Rig | Adirondack stereo input and PFM ground truth | Baseline = 176.3mm |
| **3** | 00:07 - 00:12 | Baselines | StereoBM vs StereoSGBM disparity and error maps | BM Bad2 = 36.6%, SGBM Bad2 = 23.3% |
| **4** | 00:12 - 00:16 | Optimization | Staged hyperparameter search and Pareto frontiers | Dev Bad2 = 38.8% (BM), 27.7% (SGBM) |
| **5** | 00:16 - 00:21 | Perturbations | Asymmetric brightness, gamma, and noise degradation | Tested alphas [0.6 - 1.4], sigmas [0 - 30] |
| **6** | 00:21 - 00:25 | Held-Out Test | Unseen verification on Jadeplant, Shelves, ArtL | 31.3% MAE reduction on held-out |
| **7** | 00:25 - 00:28 | Summary | Key findings and model freeze declarations | G0, G1, G2 frozen |

---

## Video B: Gunung Real-World Demonstration
**File:** `video/final/gunung_stereo_demonstration.mp4`  
**Duration:** 36 seconds (1080 frames)  

| Scene | Timestamp | Topic | Visual Content | Core Metric |
| :--- | :--- | :--- | :--- | :--- |
| **1** | 00:00 - 00:03 | Title Slate | Outdoor transfer title card | Unstructured volcanic terrain |
| **2** | 00:03 - 00:08 | Geometry Issue | Raw puncakTinggi stereo pair and vertical residual histogram | Raw $\Delta y = 28.80\text{ px}$ |
| **3** | 00:08 - 00:13 | Rectification | Before vs After epipolar collinear scanline overlay | Calibrated $\Delta y \le 2.40\text{ px}$ |
| **4** | 00:13 - 00:18 | Model Transfer | G0 Default vs G1 Transferred Disparity & LR Consistency | Photometric MAE: 10.84 -> 9.96 DN |
| **5** | 00:18 - 00:23 | Physical ROIs | Zoom into sky (63% sat), building, grass, and gravel | Heterogeneous surface degradation |
| **6** | 00:23 - 00:27 | Adaptation | Range expansion (160px) and consistency filtering | Edge score: 0.061 -> 0.129 |
| **7** | 00:27 - 00:32 | 3D Reconstruction | Calibrated metric depth map Z and top-down point cloud | 127,310 points, $Z \in [0.6\text{m}, 35\text{m}]$ |
| **8** | 00:32 - 00:36 | Conclusions | Scientific takeaways and integrity declarations | Verified metric transfer |
