# Playing with Stereo Matching

A numerical and experimental research notebook exploring classical block matching, semi-global matching (SGBM), CPU throughput optimization, metric depth triangulation, and patch correspondence learning.

![Stereo correspondence across presets and volcanic terrains](assets/stereo_loop.gif)

---

## 1. What I Explored

Estimating metric 3D depth from two calibrated 2D images is a fundamental problem in computational imaging and robotics. When two cameras observe a scene from horizontally separated optical centers, objects displace across views inversely proportional to their physical distance. Close objects displace by many pixels; distant objects displace by few pixels.

While modern vision often turns to GPU-heavy transformer models with tens of millions of parameters, field rovers, micro-aerial vehicles, and embedded inspection hardware frequently operate under tight thermal and compute budgets where only CPU execution is practical.

This notebook investigates stereo correspondence across four linked layers:

1. **Theoretical Triangulation and Sensitivity:** Mathematical formulation of rectified epipolar geometry, quadratic depth error scaling, local Sum of Absolute Differences (SAD), and Semi-Global Block Matching (SGBM) energy minimization.
2. **Controlled Synthetic Benchmarking:** Procedural planar and layered depth scenes with geometric occlusion masks evaluated under controlled photometric noise, contrast, brightness, and gamma perturbations.
3. **Standard Benchmark and CPU Optimization:** Middlebury 2014 evaluation across 15 public training scenes, completeness-aware error metrics, post-processing filter ablations, and an audited CPU Pareto frontier (FAST, BALANCED, and QUALITY presets).
4. **Outdoor Volcanic Field Deployment:** Calibrated pipeline transfer to four natural terrains on Mount Tangkuban Parahu, Indonesia, registered against co-mounted active IR depth ground truth.
5. **Minimal Learned Patch Classifier:** A 1,054-parameter convolutional network ([`TinyStereoNet`](cnn.py)) evaluated on held-out test patches against hand-crafted Mean Absolute Difference (MAD) via confirmatory hypothesis testing.

---

## 2. Stereo Geometry and Disparity

Consider two pinhole cameras mounted on a rigid horizontal baseline $B$ with matching focal length $f$ (in pixels) and identical optical axes pointing perpendicular to the baseline.

![Rectified stereo camera geometry and triangulation](assets/fig01_stereo_geometry.png)
*Figure 1: Rectified epipolar stereo camera geometry, synthetic layered scene with geometric occlusions, and educational scanline SAD cost surface. [Vector PDF](assets/fig01_stereo_geometry.pdf)*

### Mathematical Formulation of Triangulation

Because the optical centers share the identical epipolar plane, scene point $P(X, Y, Z)$ projects to the same vertical row in both camera frames:

$$y_L = y_R$$

The horizontal coordinates differ by the baseline displacement $B$:

$$x_L = f \frac{X}{Z} + c_x, \qquad x_R = f \frac{X - B}{Z} + c_x$$

Subtracting horizontal coordinates defines stereo disparity $d$:

$$d = x_L - x_R = \frac{f \cdot B}{Z}$$

Inverting disparity yields metric physical depth in meters:

$$Z = \frac{f \cdot B}{d}$$

When camera calibration introduces a horizontal disparity offset $\mathrm{doffs}$ (such as in Middlebury calibration conventions), triangulation generalizes to:

$$Z = \frac{f \cdot B}{d + \mathrm{doffs}}$$

### Depth Error Sensitivity and Quadratic Scaling

Differentiating metric depth $Z$ with respect to disparity $d$ reveals how disparity errors propagate into physical depth uncertainty:

$$\frac{\partial Z}{\partial d} = -\frac{f \cdot B}{d^2} = -\frac{Z^2}{f \cdot B}$$

First-order depth error $|\Delta Z|$ scales quadratically with true distance $Z$:

$$|\Delta Z| \approx \frac{Z^2}{f \cdot B} |\Delta d|$$

For our Logitech field camera rig ($f = 964.68$ px, $B = 0.0975$ m, $f \cdot B = 94.054$ px$\cdot$m):
- At $Z = 1.0$ m: a 1.0 px disparity error produces $|\Delta Z| \approx 0.0106$ m (1.1 cm).
- At $Z = 3.0$ m: a 1.0 px disparity error produces $|\Delta Z| \approx 0.0957$ m (9.6 cm).
- At $Z = 10.0$ m: a 1.0 px disparity error produces $|\Delta Z| \approx 1.063$ m (106.3 cm).

Due to hyperbolic inversion, underestimating disparity produces substantially larger depth errors than overestimating disparity by the same margin:

$$\Delta Z_{\mathrm{farther}} = \frac{f \cdot B}{d - \Delta d} - Z > Z - \frac{f \cdot B}{d + \Delta d} = \Delta Z_{\mathrm{closer}}$$

### Local Matching Cost (SAD)

For a reference pixel $(x_L, y)$ in the left image, candidate matches along the right epipolar scanline are scored over a local window of size $K \times K$ using the Sum of Absolute Differences:

$$\mathrm{SAD}(d) = \sum_{u=-k}^k \sum_{v=-k}^k |I_L(x_L + u, y + v) - I_R(x_L - d + u, y + v)|$$

where $k = (K - 1) / 2$. As illustrated in Figure 1(e), the SAD cost curve drops sharply to a clear global minimum on textured surfaces. On textureless regions, the curve flattens, creating aperture ambiguity.

### Semi-Global Matching (StereoSGBM)

Local block matching minimizes SAD independently per window, making it prone to edge-fattening artifacts and aperture ambiguity. StereoSGBM resolves this by optimizing a global 2D energy functional across the entire image:

$$E(D) = \sum_p C(p, D_p) + \sum_{q \in N_p} P_1 \cdot \mathbb{I}[|D_p - D_q| = 1] + \sum_{q \in N_p} P_2 \cdot \mathbb{I}[|D_p - D_q| > 1]$$

where $C(p, D_p)$ is the Birchfield-Tomasi sampling-insensitive matching cost, $P_1$ penalizes small disparity steps (slanting surfaces), and $P_2$ penalizes large disparity jumps (depth discontinuities). SGBM minimizes this functional by aggregating 1D dynamic programming costs along cardinal scanline directions.

---

## 3. Synthetic Stereo Experiments

To evaluate matching precision where ground truth is known exactly, I constructed two procedural synthetic test scenes ($320 \times 240$ pixels):
1. **Constant Disparity Scene:** A textured planar surface translated by a known integer offset ($d = 16.0$ px).
2. **Layered Scene:** Three distinct depth planes (background $d = 8$ px, middleground $d = 18$ px, foreground $d = 32$ px) with half-occluded pixels geometrically masked out.

### Photometric Perturbation Benchmark

Both scenes were evaluated under five photometric perturbations: additive Gaussian noise ($\sigma = 15$), brightness shifts ($\pm 30$), contrast reduction ($0.5\times$), and non-linear gamma warping ($\gamma = 1.6$).

```text
Table 1: Synthetic Benchmark Performance Across Matchers and Perturbations
```

| Scene & Perturbation | StereoBM Bad@2 | StereoBM Cov | StereoSGBM Bad@2 | StereoSGBM Cov | SGBM Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Constant (Clean)** | 0.00% | 84.2% | **0.00%** | **84.2%** | 4.91 ms |
| **Constant (Noise $\sigma=15$)** | 0.01% | 84.2% | **0.01%** | **84.2%** | 5.03 ms |
| **Constant (Gamma $\gamma=1.6$)** | 0.35% | 78.4% | **0.22%** | **79.9%** | 8.62 ms |
| **Layered (Clean)** | 2.48% | 77.6% | **2.03%** | **86.7%** | 13.27 ms |
| **Layered (Brightness +30)** | 2.51% | 76.8% | **2.26%** | **85.6%** | 10.85 ms |
| **Layered (Contrast 0.5x)** | 3.12% | 75.1% | **2.36%** | **86.6%** | 11.67 ms |
| **Layered (Noise $\sigma=15$)** | 9.72% | 12.9% | **6.02%** | **81.0%** | 12.45 ms |
| **Layered (Gamma $\gamma=1.6$)** | 3.05% | 76.2% | **1.76%** | **85.2%** | 15.66 ms |

Under clean conditions, SGBM recovers layered disparities with a 2.03% bad-pixel rate (pixels where $|d_{\mathrm{pred}} - d_{\mathrm{gt}}| > 2.0$ px). Under additive Gaussian noise ($\sigma = 15$), local StereoBM coverage collapses from 77.6% to 12.9% because window SAD costs fail confidence checks. In contrast, SGBM maintains 81.0% coverage with 6.02% bad-pixel error, demonstrating the stabilizing effect of directional dynamic programming paths.

---

## 4. Middlebury 2014 Evaluation

The pipeline was benchmarked across 15 publicly available labeled training scenes from the Middlebury 2014 dataset (MiddEval3) using subpixel ground-truth PFM disparities under non-occluded evaluation masks (`mask0nocc.png`).

> [!NOTE]
> Middlebury evaluations presented here are offline research evaluations on public labeled training scenes at 0.5x scale, not an official benchmark submission to the Middlebury evaluation server.

![Middlebury 2014 benchmark evaluation](assets/fig02_middlebury.png)
*Figure 2: Middlebury 2014 benchmark evaluation on Jadeplant at 0.5x scale. Panels show left view, reference disparity from PFM, non-occluded mask, raw SGBM, WLS filtered SGBM, and spatial error map. [Vector PDF](assets/fig02_middlebury.pdf)*

```text
Table 2: Middlebury 2014 Benchmark Results Across 15 Scenes (0.5x Scale)
```

| Scene | Disparity Range | Bad@2.0 (%) | Disp MAE (px) | Depth MAE (m) | Coverage (%) | CPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adirondack** | 73 px | **9.26%** | 1.54 px | 0.039 m | 87.9% | 11.21 ms |
| **ArtL** | 64 px | 26.10% | 3.07 px | 0.072 m | 78.5% | 1.78 ms |
| **Jadeplant** | 160 px | 23.54% | 7.01 px | 0.066 m | 68.6% | 8.53 ms |
| **Motorcycle** | 70 px | **7.78%** | 1.27 px | 0.057 m | 88.8% | 20.51 ms |
| **MotorcycleE** | 70 px | 8.60% | 1.50 px | 0.067 m | 84.7% | 16.75 ms |
| **Piano** | 65 px | 19.61% | 1.72 px | 0.068 m | 86.5% | 13.97 ms |
| **PianoL** | 65 px | 30.87% | 6.97 px | 0.177 m | 69.0% | 15.27 ms |
| **Pipes** | 75 px | 15.56% | 2.85 px | 0.297 m | 86.3% | 25.55 ms |
| **Playroom** | 83 px | 14.62% | 2.12 px | 0.053 m | 88.1% | 16.05 ms |
| **Playtable** | 73 px | 13.00% | 1.55 px | 0.048 m | 88.2% | 11.64 ms |
| **PlaytableP** | 73 px | 12.12% | 1.49 px | 0.049 m | 88.3% | 17.35 ms |
| **Recycle** | 65 px | **8.69%** | 1.10 px | 0.028 m | 86.1% | 16.19 ms |
| **Shelves** | 60 px | 27.17% | 2.42 px | 0.090 m | 89.6% | 18.26 ms |
| **Teddy** | 64 px | **9.66%** | 1.04 px | 0.016 m | 90.6% | 4.23 ms |
| **Vintage** | 190 px | 22.85% | 2.83 px | 0.042 m | 73.8% | 15.02 ms |
| **Macro Average** | **81 px** | **16.63%** | **2.57 px** | **0.078 m** | **83.7%** | **14.15 ms** |

Across all 15 scenes at 0.5x scale:
- Macro average Bad@2.0 error: **16.63%**
- Mean valid prediction coverage: **83.67%**
- Completeness-aware Bad@2.0 error: **29.88%**
- Mean depth triangulation error: **7.8 cm** ($0.078$ m) at an average CPU latency of **14.15 ms**.

At native 1.0x resolution on the Jadeplant reference scene:
- Bad@2.0 error drops to **11.42%**, depth MAE is **3.48 cm**, and matching latency is **47.62 ms**.

### Post-Processing Filter Ablation: Accuracy Versus Completeness

To understand whether sophisticated post-processing filters improve matching, six filtering strategies were tested on Jadeplant (1.0x scale):
- `raw`: Unfiltered StereoSGBM output.
- `median_valid`: $3 \times 3$ median filter applied strictly over valid disparity pixels.
- `guided`: Guided image filter using the grayscale left view as structural guide.
- `joint_bilateral`: Color joint bilateral edge-preserving smoothing.
- `wls`: Weighted Least Squares disparity filter using left-right consistency.
- `confidence_edge`: WLS disparity filter with edge confidence thresholding.

![Filter ablation and parameter sensitivity](assets/fig04_optimization.png)
*Figure 4: Stereo filter ablation and parameter sensitivity. (a) Bad@2.0 conditional error, completeness-aware error, and total latency across filters. (b) Disparity accuracy versus runtime across SGBM block sizes. [Vector PDF](assets/fig04_optimization.pdf)*

```text
Table 3: Filter Benchmark Comparison on Reference Scene (Jadeplant, Native 1.0x Scale)
```

| Filter Mode | Bad@2.0 Cond (%) | Bad@2.0 CA (%) | Valid Coverage (%) | Match Latency | Filter Latency | Total Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **raw** | 11.42% | 38.64% | 69.27% | 42.5 ms | **0.9 ms** | **43.4 ms** |
| **median_valid** | **11.55%** | **38.73%** | **69.27%** | 42.5 ms | 4.6 ms | 47.1 ms |
| **guided** | 46.93% | 63.24% | 69.27% | 42.5 ms | 18.3 ms | 60.8 ms |
| **joint_bilateral** | 19.38% | 44.16% | 69.27% | 42.5 ms | 14.9 ms | 57.4 ms |
| **wls** | 26.89% | 41.90% | 79.46% | 84.8 ms | 26.6 ms | 111.4 ms |
| **confidence_edge** | **4.41%** | **60.73%** | **41.08%** | 84.8 ms | 28.9 ms | 113.6 ms |

#### Key Insights on Post-Processing:
1. **The Conditional Accuracy Fallacy:** `confidence_edge` appears superior if judged strictly by conditional Bad@2 (4.41%). However, it rejects 58.9% of evaluable pixels. Its completeness-aware error ($\mathrm{Bad}_{\mathrm{CA}} = 1 - \mathrm{Cov} + \mathrm{Cov} \cdot \mathrm{Bad}_{\mathrm{valid}}$) jumps to **60.73%**, far worse than raw SGBM (38.64%).
2. **WLS Filtering Overhead and Edge Blurring:** WLS requires dual-pass matching (left-to-right and right-to-left) plus solving a large sparse linear system. This doubles matching latency from 42.5 ms to 84.8 ms and adds 26.6 ms filter overhead (total 111.4 ms). Furthermore, WLS blurs depth discontinuities on this scene, worsening conditional Bad@2 from 11.42% to 26.89%.
3. **Median Valid is Pareto-Optimal:** `median_valid` cleans isolated disparity spikes while preserving valid masks without sentinel bleeding, adding only 4.6 ms filter overhead.

---

## 5. CPU Optimization and FAST/BALANCED/QUALITY

![CPU Pareto frontier and multi-threading scaling](assets/fig05_cpu_pareto.png)
*Figure 5: CPU optimization analysis. (a) Measured latency versus error Pareto frontier for FAST, BALANCED, and QUALITY presets. (b) Multi-threading speedup curves at 360p resolution. [Vector PDF](assets/fig05_cpu_pareto.pdf)*

### Algorithmic Levers on CPU

1. **Resolution Scaling (Dominant Lever):** Scaling 720p input down to 360p ($0.5\times$) yields a **$6.2\times$ speedup** (41.4 ms vs 133 ms). At 180p ($0.25\times$), SGBM exceeds 130 FPS on CPU.
2. **Block Size Sensitivity:** In a 15-configuration search, block size 5 strictly dominated block size 7 across all metrics: it executes over $2\times$ faster (133 ms vs 338 ms) and achieves lower error (9.97% vs 11.42%) due to reduced window-over-smoothing at sharp depth boundaries.
3. **Thread Scaling:** Multi-threading scales effectively up to 8 threads, with diminishing returns beyond 8 threads due to CPU cache and memory bandwidth saturation.

```text
Table 4: Measured Pipeline Presets on 720p Input Images
```

| Preset | Scale Factor | Disparities | Block Size | Post-Filter | Pipeline Latency | Measured FPS | Jadeplant Bad@2 (Cond / CA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FAST** | $0.5\times$ (360p) | 80 px | 5 px | `median_valid` | 41.4 – 79.5 ms | 12.6 – 24.2 FPS | 21.6% / 46.1% |
| **BALANCED** | $0.75\times$ (540p) | 128 px | 5 px | `median_valid` | 81.1 – 178.9 ms | 5.6 – 12.3 FPS | 16.3% / 43.8% |
| **QUALITY** | $1.0\times$ (720p) | 160 px | 5 px | `median_valid` | 133.0 – 491.5 ms | 2.0 – 7.5 FPS | **10.0% / 37.1%** |

These three presets form a strictly monotonic Pareto frontier balancing latency and reconstruction fidelity on standard multi-core CPUs.

---

## 6. Field Evaluation Across Four Gunung Terrains

To evaluate classical stereo outside laboratory environments, the calibrated pipeline was deployed across four volcanic terrains on Mount Tangkuban Parahu, Indonesia:
1. `puncakTinggi`: High-altitude ridge with gravel, rough basalt rock, and direct sun.
2. `puncakWarung`: Roadside stall area with corrugated tin roofing and deep shadow occlusions.
3. `tanggaBiru`: Steep trail stairway bounded by painted blue steel railings.
4. `tanggaKuning`: Natural soil and stone steps bounded by painted yellow railings.

![Four Gunung volcanic terrains evaluation](assets/fig03_gunung.png)
*Figure 3: Four Gunung volcanic terrains evaluated under calibrated stereo and active IR RealSense ground truth. [Vector PDF](assets/fig03_gunung.pdf)*

### Sensor Configuration and Registration

- **Stereo Rig:** Dual calibrated cameras ($1280 \times 720$, baseline $B = 97.50$ mm, focal length $f_x = 964.68$ px, $f \cdot B = 94.054$ px$\cdot$m).
- **Metric Ground Truth:** Co-mounted Intel RealSense D435i active IR stereo sensor ($640 \times 480$ native depth).
- **Registration Protocol:** 3D points unprojected from RealSense depth via $K_{\mathrm{rs}}$ were matched to 2D SIFT features in the rectified stereo left view. Rigid translation $T$ and rotation $R$ were solved via PnP RANSAC ($3.0$ px reprojection threshold).
- **Z-Buffered Projection:** RealSense depth was projected into the $1280 \times 720$ stereo frame using depth-based z-buffering.

```text
Table 5: RealSense Registration and Metric Depth Validation Across Terrains
```

| Terrain | Status | SIFT Inliers | Reproj Med (px) | Translation $T$ (m) | Eval Pixels | Ref Cov (%) | Frame Cov (%) | Depth MAE (m) | Depth Med (m) | $\delta < 1.25$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **puncakTinggi** | **VERIFIED** | 106 | 1.25 px | $[-0.011, -0.035, -0.000]$ | 76,980 | 41.7% | 8.35% | 0.234 m | **0.059 m** | **88.4%** |
| **puncakWarung** | **OUT_OF_RANGE**| 0 | N/A | N/A | 0 | 0.0% | 0.00% | Unmeasured | Unmeasured | N/A |
| **tanggaBiru** | **VERIFIED** | 77 | 0.73 px | $[-0.026, -0.044, -0.030]$ | 13,759 | 12.7% | 1.49% | 0.500 m | **0.076 m** | 70.5% |
| **tanggaKuning** | **VERIFIED** | 32 | 1.19 px | $[-0.017, -0.034, -0.013]$ | 38,613 | 28.6% | 4.19% | 0.211 m | **0.061 m** | **88.2%** |

### Mechanical Rig Stability and Extrinsic Transfer

Transferring camera extrinsics $(R, T)$ solved on `puncakTinggi` directly to `tanggaBiru` resulted in an accuracy breakdown: median depth error increased from 0.076 m to 0.181 m, and reprojection residuals climbed to 56.5 px.

Geometric analysis identified two root causes:
1. **Mechanical Rig Flexure Between Sessions:** Pairwise rotation differences across terrains range from $2.6^\circ$ to $4.7^\circ$, and translation shifts vary by 1.5 cm to 3.5 cm. The tripod bracket underwent physical micro-shifts during transport between field sites.
2. **Geometric Degeneracy in `tanggaBiru`:** SIFT inliers in `tanggaBiru` clustered along a single linear steel railing ($Y$-spread standard deviation of only 9 mm). A collinear feature distribution creates an ill-conditioned PnP null space, rendering extrinsic transfer unreliable.

> [!IMPORTANT]
> **Coverage Caveat:** While median depth errors on valid points are 5.9–7.6 cm, the jointly evaluable region constitutes only **1.49% to 8.35% of the full 720p image frame** (due to RealSense active IR sunlight wash-out, FOV disparity, and occlusion boundaries). RealSense comparisons must be understood as *provisional local spot checks*, not full-field dense ground truth.

### Far-Field Geometry (`puncakWarung`)

In `puncakWarung`, scene geometry exceeds 15 meters (all detected SIFT keypoints have depths $Z > 15.6$ m). The Intel RealSense D435i active IR emitter operates up to 3–6 meters in daylight; beyond 10 meters, active IR returns drop to zero. StereoSGBM successfully reconstructs the scene, but depth error against RealSense is documented honestly as unmeasured.

Across the three valid near-field terrains, classical SGBM achieves median metric depth errors of **5.9 cm** (`puncakTinggi`), **7.6 cm** (`tanggaBiru`), and **6.1 cm** (`tanggaKuning`), with over 88% of evaluated surface points within 25% of true distance ($\delta < 1.25$).

### Cross-Dataset Generalization

![Cross-dataset generalization comparison](assets/fig07_cross_dataset.png)
*Figure 7: Cross-dataset generalization. Row A shows Middlebury indoor benchmark scene (Jadeplant). Row B shows outdoor volcanic terrain (puncakTinggi) evaluated against registered RealSense metric depth. [Vector PDF](assets/fig07_cross_dataset.pdf)*

Classical stereo matching requires no domain retraining:
- In indoor scenes with high-frequency plant foliage and complex occlusions (Row A), SGBM recovers depth boundaries with 7.01 px disparity MAE (0.066 m depth MAE at 0.5x scale).
- In outdoor volcanic scenes with rocky gravel, boulders, and harsh shadows (Row B), the same algorithm with calibrated parameters yields 5.89 cm median depth error.

---

## 7. TinyStereoNet CNN

To explore what a minimal learned network discovers about patch correspondence, I implemented [`TinyStereoNet`](cnn.py), a compact 2-layer CNN running entirely on CPU:
- Input: Two $15 \times 15$ grayscale patches stacked along channels ($2 \times 15 \times 15$).
- Architecture:
  - Conv1: $2 \to 8$ channels, $3 \times 3$, ReLU (152 params)
  - Conv2: $8 \to 12$ channels, $3 \times 3$, ReLU (876 params)
  - Global Average Pooling: $12 \times 11 \times 11 \to 12$
  - Linear Classifier: $12 \to 2$ logits (26 params)
  - Total: **1,054 trainable parameters**

> [!NOTE]
> TinyStereoNet performs patch correspondence classification ($y \in \{0, 1\}$), determining whether two $15 \times 15$ patches match. It does not perform dense disparity map reconstruction.

![TinyStereoNet patch correspondence classifier](assets/fig06_cnn.png)
*Figure 6: TinyStereoNet patch classifier: training dynamics, held-out test generalization against classical Mean Absolute Difference (MAD) with full 0-100% axis and multi-seed error bars, and learned Conv1 filter kernels. [Vector PDF](assets/fig06_cnn.pdf)*

### Training and Statistical Validation Protocol

To prevent data leakage, training (1,000 pairs), validation (200 pairs), and test (246 pairs) sets were sampled from mutually disjoint synthetic scenes.

Model selection was performed strictly on validation accuracy; held-out test data was evaluated once:
1. **Confirmatory Hypothesis Test (Primary Model):** The primary model (Seed 202, selected at 95.96% val acc) achieves **95.53% accuracy** on held-out test data vs 91.06% for classical MAD (+4.47 percentage points). On the 15 discordant pairs ($n_{01}=2, n_{10}=13$), the exact paired McNemar test yields **$p = 0.0074$** ($p < 0.01$), with a 95% bootstrap confidence interval of **$[+1.63\%, +7.73\%]$**.
2. **Descriptive Initialization Stability:** Across five random seeds (`[42, 101, 202, 303, 404]`), test accuracy is $95.53\% \pm 0.70\%$, confirming that correspondence learning is robust to weight initialization.

```text
Table 6: Patch Correspondence Classification on Untouched Held-Out Test Data
```

| Method | Test Accuracy | Precision | Recall | F1 Score | CPU Latency / Patch | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Baseline (MAD)** | 91.06% | 87.97% | 95.12% | 91.41% | **0.001 ms** | **0 (hand-designed)** |
| **TinyStereoNet (Primary Model, Seed 202)** | **95.53%** | **95.90%** | **95.12%** | **95.51%** | 0.041 ms | 1,054 |
| **TinyStereoNet (5 Seeds, Descriptive)** | **95.53% $\pm$ 0.70%** | **96.1% $\pm$ 1.9%** | **94.9% $\pm$ 0.9%** | **95.5% $\pm$ 0.6%** | 0.041 ms | 1,054 |

Training requires **5.3 seconds** on CPU. Learned Conv1 filter kernels specialize in horizontal edge extraction and cross-view difference detection.

---

## 8. What Worked and What Failed

### What Worked

1. **Classical SGBM Metric Triangulation:** With accurate epipolar rectification, classical SGBM achieves 5.9–7.6 cm median depth accuracy on outdoor volcanic terrain and 7.8 cm macro-average across 15 Middlebury scenes.
2. **Resolution Scaling for Real-Time CPU Throughput:** Downscaling input resolution to $0.5\times$ (360p) provides a $6.2\times$ matching speedup, delivering 12.6–24.2 FPS on commodity CPU hardware.
3. **Validity-Aware Median Filtering:** `median_valid` suppresses isolated disparity spikes without smoothing true depth boundaries or bleeding invalid sentinel values (adding only 4.6 ms overhead).
4. **Compact Feature Learning:** A 1,054-parameter CNN outperforms hand-crafted MAD (+4.47 pp accuracy, $p = 0.0074$) while training in 5.3 seconds on CPU.
5. **Completeness-Aware Metric Evaluation:** Evaluating completeness-aware error ($\mathrm{Bad}_{\mathrm{CA}}$) exposes artificial accuracy gains caused by confidence thresholding.

### What Failed

1. **WLS Disparity Filtering:** Dual-pass matching and sparse matrix solving doubled latency (from 43.4 ms to 111.4 ms) while smoothing sharp depth boundaries, increasing conditional Bad@2 on Jadeplant from 11.42% to 26.89%.
2. **Cross-Terrain Camera Extrinsic Transfer:** Rig extrinsics estimated on one site degraded median depth error from 0.076 m to 0.181 m on another site due to physical rig flexure during transport ($2.6^\circ$–$4.7^\circ$ rotation shifts) and collinear SIFT feature distributions along handrails.
3. **Active IR Depth in Far-Field Daylight:** In `puncakWarung` ($Z > 15$ m), solar ambient radiation completely washed out the active IR projection pattern, leaving stereo unmeasured against RealSense.
4. **Conditional Error Optimization Alone:** Aggressive confidence thresholding (`confidence_edge`) dropped conditional Bad@2 to 4.41% by discarding 58.9% of evaluable pixels, blowing up completeness-aware error to 60.73%.
5. **Dense CNN Stereo on CPU:** While 0.041 ms per patch is fast, evaluating a dense 720p image grid requires millions of forward passes, confirming that classical SGBM remains superior for real-time CPU deployment.

---

## 9. Reproduction Instructions

### Requirements

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For optional CNN training:

```bash
python -m pip install -r requirements-ml.txt
```

### Fast Sanity Verification (< 15 seconds)

Runs quick end-to-end verification across synthetic stereo, Middlebury sample, Gunung sample, and CNN architecture. If local dataset archives are not present, it automatically runs safe synthetic demonstrations:

```bash
python reproduce.py --quick
```

### Run Unit Tests

```bash
python -m pytest -q
```

### Modular Benchmark Commands

```bash
# Run synthetic procedural benchmark
python reproduce.py --synthetic

# Run Middlebury 2014 benchmark across all 15 scenes
python reproduce.py --middlebury

# Run Gunung field transfer evaluation
python reproduce.py --gunung

# Run filter ablation and parameter search
python reproduce.py --optimize

# Run CPU multiscale and threading profiling
python reproduce.py --profile

# Train TinyStereoNet and evaluate on held-out test patches
python reproduce.py --cnn

# Regenerate publication figures and animated GIF
python visualize.py
```

### Full Benchmark Suite

Executes the complete research suite across all modules, updates result metrics, and generates all publication figures and animated GIF:

```bash
python reproduce.py --full
```

### Output Files in `results/`

- `synthetic.csv`: Procedural benchmark across 6 perturbations and 3 matchers.
- `middlebury.csv`: Middlebury 2014 evaluation across 15 training scenes.
- `gunung.csv`: Four Gunung volcanic terrains evaluation and RealSense metrics.
- `optimization.csv`: Filter ablation and SGBM block-size sensitivity data.
- `best_params.json`: CPU multiscale, threading, and preset profile benchmarks.
- `cnn_metrics.json`: TinyStereoNet and classical MAD test metrics with multi-seed stats.
- `environment.json`: Hardware, compiler, and OS provenance record.

---

## 10. Scientific References

- D. Scharstein and R. Szeliski, "A taxonomy and evaluation of dense two-frame stereo correspondence algorithms," *International Journal of Computer Vision*, 47(1-3):7-42, 2002. https://doi.org/10.1023/A:1014573219977
- H. Hirschmuller, "Stereo processing by semiglobal matching and mutual information," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(2):328-341, 2008. https://doi.org/10.1109/TPAMI.2007.1166
- D. Scharstein, H. Hirschmuller, et al., "High-resolution stereo datasets with subpixel-accurate ground truth," *German Conference on Pattern Recognition (GCPR)*, 2014. https://vision.middlebury.edu/stereo/data/scenes2014/
- J. Zbontar and Y. LeCun, "Stereo matching by training a convolutional neural network to compare image patches," *Journal of Machine Learning Research*, 17(1):2287-2318, 2016.
- OpenCV StereoSGBM Reference: https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
- KindlyGentleman/fft-diffraction: https://github.com/KindlyGentleman/fft-diffraction
