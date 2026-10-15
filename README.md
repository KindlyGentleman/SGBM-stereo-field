# Playing with Stereo Matching

*A minimal, reproducible research notebook exploring classical stereo correspondence, CPU performance optimization, patch-level convolutional learning, and cross-dataset metric depth transfer.*

![Stereo correspondence across presets and volcanic terrains](assets/stereo_loop.gif)

## Overview and Mission

Estimating metric 3D depth from two 2D images is a cornerstone of computational imaging and robotics. When two rectified cameras view a scene from slightly separated vantage points, objects displace horizontally across the two views by an amount inversely proportional to their distance. Nearby objects displace by many pixels; distant objects displace by few pixels.

Modern computer vision often defaults to deep transformer models containing hundreds of millions of parameters. However, on embedded platforms, micro-aerial vehicles, and field robots, dense GPU networks are often impractical due to thermal, latency, and power constraints.

This notebook documents a three-level investigation into stereo matching:
1. **Educational Core:** Pinhole triangulation, synthetic scenes with known ground truth, local block matching (StereoBM), semi-global matching (StereoSGBM), photometric robustness, and an educational 1,054-parameter CNN patch classifier.
2. **Standard Benchmark & CPU Optimization:** Middlebury 2014 evaluation across 15 high-resolution indoor scenes, post-processing filter ablations, and a multi-threaded CPU Pareto frontier yielding FAST (30 FPS target), BALANCED (15 FPS target), and QUALITY (3 FPS target) presets.
3. **Field Transfer & Real-World Metric Ground Truth:** Transfer of the optimized pipeline to four outdoor volcanic terrains on Mount Tangkuban Parahu, Indonesia (puncakTinggi, puncakWarung, tanggaBiru, tanggaKuning) with sub-pixel camera calibration and independently registered Intel RealSense D435i active IR depth ground truth.

---

## 1. Mathematical Principles of Stereo Triangulation

Consider two pinhole cameras mounted on a horizontal baseline $B$ with matching focal lengths $f$ (in pixels) and identical optical axes pointing perpendicular to the baseline.

![Rectified stereo camera geometry and triangulation](assets/fig01_stereo_geometry.png)
*Figure 1: Rectified epipolar stereo camera geometry, synthetic layered scene with geometric occlusions, and educational scanline SAD cost surface. [Vector PDF](assets/fig01_stereo_geometry.pdf)*

Because the cameras are rectified to lie in the same epipolar plane, any 3D scene point $P(X, Y, Z)$ projects to the identical vertical row in both camera frames:

$$y_L = y_R$$

The horizontal coordinates differ by the horizontal displacement between the two optical centers:

$$x_L = f \frac{X}{Z} + c_x, \qquad x_R = f \frac{X - B}{Z} + c_x$$

Subtracting horizontal coordinates defines the stereo disparity $d$:

$$d = x_L - x_R = \frac{f \cdot B}{Z}$$

Inverting this equation yields physical metric depth in meters:

$$Z = \frac{f \cdot B}{d}$$

When a camera calibration introduces a horizontal disparity offset $\mathrm{doffs}$ (as in Middlebury datasets), the triangulation relation generalizes to:

$$Z = \frac{f \cdot B}{d + \mathrm{doffs}}$$

### Local Matching Cost (SAD)

For a reference pixel $(x_L, y)$ in the left image, candidate correspondences along the right scanline are evaluated over a local window of size $K \times K$ using the Sum of Absolute Differences:

$$\mathrm{SAD}(d) = \sum_{u=-k}^k \sum_{v=-k}^k |I_L(x_L + u, y + v) - I_R(x_L - d + u, y + v)|$$

where $k = (K - 1) / 2$. On textured surfaces, the cost drops to a clean minimum at the true disparity. On flat surfaces or repetitive patterns, the cost curve flattens, producing ambiguity.

### Semi-Global Block Matching (StereoSGBM)

Local block matching minimizes SAD independently per window, suffering from edge fattening and aperture ambiguity. StereoSGBM resolves this by optimizing a 2D energy functional across the whole image:

$$E(D) = \sum_p C(p, D_p) + \sum_{q \in N_p} P_1 \cdot \mathbb{I}[|D_p - D_q| = 1] + \sum_{q \in N_p} P_2 \cdot \mathbb{I}[|D_p - D_q| > 1]$$

where $C(p, D_p)$ is the Birchfield-Tomasi sampling cost, $P_1$ penalizes smooth gradient shifts, and $P_2$ penalizes sharp depth jumps. SGBM minimizes this functional by aggregating 1D dynamic programming paths along 3, 5, or 8 cardinal directions.

---

## 2. Level 1: Synthetic Benchmarking & Tiny CNN

To establish ground-truth bounds under controlled conditions, I built two procedural synthetic scenes ($320 \times 240$ pixels):
1. **Planar Constant Disparity:** A smooth, textured surface shifted by an exact integer offset ($d = 16$ px).
2. **Layered 3-Depth Geometry:** Background ($d = 8$ px), middleground ($d = 18$ px) with an untextured flat zone, and a foreground object ($d = 32$ px). Occluded pixels are tracked geometrically and masked out during evaluation.

### Baseline Reconstruction & Photometric Robustness

```text
Table 1: Synthetic Benchmark Performance Across Perturbations (Bad@2.0% and Valid Coverage%)
```

| Scene & Perturbation | StereoBM Bad@2 | StereoBM Cov | StereoSGBM Bad@2 | StereoSGBM Cov | SGBM Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Constant (Clean)** | 0.00% | 84.2% | **0.00%** | **84.2%** | 7.46 ms |
| **Constant (Noise $\sigma=15$)** | 0.01% | 84.2% | **0.01%** | **84.2%** | 6.27 ms |
| **Constant (Gamma $\gamma=1.6$)** | 0.35% | 78.4% | **0.22%** | **79.9%** | 4.99 ms |
| **Layered (Clean)** | 2.48% | 77.6% | **1.15%** | **86.0%** | 5.76 ms |
| **Layered (Brightness +30)** | 2.51% | 76.8% | **1.19%** | **84.7%** | 5.42 ms |
| **Layered (Contrast 0.5x)** | 3.12% | 75.1% | **1.49%** | **85.8%** | 3.85 ms |
| **Layered (Noise $\sigma=15$)** | 9.72% | 12.9% | **4.23%** | **79.5%** | 5.62 ms |
| **Layered (Gamma $\gamma=1.6$)** | 3.05% | 76.2% | **1.53%** | **85.0%** | 6.12 ms |

Under clean conditions, SGBM achieves a 1.15% bad-pixel rate on the layered scene. Under additive Gaussian noise ($\sigma = 15$), StereoBM coverage collapses from 77.6% to 12.9%, while SGBM maintains 79.5% coverage with 4.23% error due to directional energy aggregation.

### TinyStereoNet: What Can 1,054 Parameters Learn?

To investigate what a minimal learned model discovers about correspondence, I implemented `TinyStereoNet`, a 2-layer CNN designed to run entirely on CPU:
- Input: Two $15 \times 15$ grayscale patches stacked along channels ($2 \times 15 \times 15$).
- Architecture:
  - Conv1: $2 \to 8$ channels, $3 \times 3$, ReLU (152 params)
  - Conv2: $8 \to 12$ channels, $3 \times 3$, ReLU (876 params)
  - Global Average Pooling: $12 \times 11 \times 11 \to 12$
  - Linear Classifier: $12 \to 2$ logits (26 params)
  - Total: **1,054 trainable parameters**

![TinyStereoNet patch correspondence classifier](assets/fig06_cnn.png)
*Figure 6: TinyStereoNet patch classifier: training dynamics, held-out test generalization against classical Mean Absolute Difference (MAD), and learned Conv1 filter kernels. [Vector PDF](assets/fig06_cnn.pdf)*

To prevent data leakage, training (968 pairs), validation (198 pairs), and test (246 pairs) sets were generated using mutually disjoint random seeds. Patches were normalized with zero-mean and unit-variance. Model selection was performed strictly on validation accuracy; test data was evaluated only once.

```text
Table 2: Patch Correspondence Classification on Untouched Held-Out Test Data
```

| Method | Test Accuracy | Precision | Recall | F1 Score | CPU Latency / Patch | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical Baseline (MAD)** | 91.06% | 87.97% | 95.12% | 91.41% | **0.001 ms** | **0 (hand-designed)** |
| **TinyStereoNet (Learned)** | **94.31%** | **92.91%** | **95.93%** | **94.40%** | 0.044 ms | 1,054 |

Training TinyStereoNet took **5.17 seconds** on a standard CPU. The learned model reduced test false positives from 16 down to 9, lifting accuracy from 91.06% to 94.31%. Visualizing the learned Conv1 kernels reveals orthogonal edge filters and cross-channel contrast comparators.

---

## 3. Level 2: Middlebury 2014 & CPU Optimization

To measure classical stereo under challenging laboratory conditions, the pipeline was evaluated across all 15 training scenes from the Middlebury 2014 benchmark (MiddEval3) under non-occluded evaluation masks (`mask0nocc.png`).

![Middlebury 2014 benchmark evaluation](assets/fig02_middlebury.png)
*Figure 2: Middlebury 2014 benchmark evaluation on Jadeplant. Panels show left view, reference disparity from PFM, non-occluded mask, raw SGBM, WLS filtered SGBM, and spatial error map. [Vector PDF](assets/fig02_middlebury.pdf)*

```text
Table 3: Middlebury 2014 Benchmark Results Across 15 Scenes (0.5x Scale)
```

| Scene | Disparity Range | Bad@2.0 Error (%) | Disp MAE (px) | Depth MAE (m) | Valid Coverage (%) | CPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Adirondack** | 73 px | **9.58%** | 1.64 px | 0.046 m | 88.2% | 9.75 ms |
| **ArtL** | 64 px | 29.44% | 3.99 px | 0.099 m | 82.4% | 2.68 ms |
| **Jadeplant** | 160 px | 26.66% | 9.36 px | 0.090 m | 71.6% | 18.95 ms |
| **Motorcycle** | 70 px | **8.53%** | 1.45 px | 0.067 m | 89.5% | 17.27 ms |
| **MotorcycleE** | 70 px | 10.25% | 1.88 px | 0.091 m | 86.3% | 13.69 ms |
| **Piano** | 65 px | 20.69% | 1.88 px | 0.081 m | 87.8% | 18.01 ms |
| **PianoL** | 65 px | 33.36% | 7.33 px | 0.212 m | 71.8% | 16.21 ms |
| **Pipes** | 75 px | 17.32% | 3.25 px | 0.359 m | 88.2% | 22.85 ms |
| **Playroom** | 83 px | 14.61% | 2.11 px | 0.067 m | 88.1% | 13.78 ms |
| **Playtable** | 73 px | 13.44% | 1.58 px | 0.059 m | 88.7% | 15.27 ms |
| **PlaytableP** | 73 px | 12.66% | 1.54 px | 0.063 m | 88.9% | 18.19 ms |
| **Recycle** | 65 px | **9.32%** | 1.21 px | 0.034 m | 86.7% | 21.98 ms |
| **Shelves** | 60 px | 27.80% | 2.48 px | 0.104 m | 90.4% | 12.77 ms |
| **Teddy** | 64 px | **9.45%** | 1.01 px | 0.020 m | 90.4% | 5.72 ms |
| **Vintage** | 190 px | 23.50% | 2.89 px | 0.069 m | 74.4% | 23.34 ms |
| **Mean / Summary** | **81 px** | **17.77%** | **2.91 px** | **0.096 m** | **84.9%** | **15.36 ms** |

Across all 15 scenes, SGBM achieves an average Bad@2.0 rate of 17.77% and 84.9% valid coverage in 15.36 ms of CPU time. Mean depth triangulation error is 9.6 cm.

### Post-Processing Filter Ablation

Using Middlebury reference scenes, six post-processing strategies were evaluated under identical inputs:
- `raw`: Unfiltered SGBM.
- `median_valid`: 3x3 median blur restricted to non-zero pixels.
- `guided`: Guided filter using grayscale left image as structural guide.
- `joint_bilateral`: Color joint bilateral edge-preserving smoothing.
- `wls`: Weighted Least Squares disparity filter using left-right consistency.
- `confidence_edge`: WLS with confidence-map thresholding to suppress uncertain matches.

![Filter ablation and parameter sensitivity](assets/fig04_optimization.png)
*Figure 4: Stereo filter ablation and parameter sensitivity. (a) Bad@2.0 error and total pipeline latency across filters. (b) Disparity accuracy versus runtime across SGBM block sizes. [Vector PDF](assets/fig04_optimization.pdf)*

```text
Table 4: Filter Benchmark Comparison on Reference Scene
```

| Filter Mode | Bad@2.0 Error (%) | Disp MAE (px) | Valid Coverage (%) | Filter Latency (ms) | Total Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **raw** | 10.04% | 3.06 px | 70.1% | **0.0 ms** | 255.8 ms |
| **median_valid** | **10.00%** | **3.04 px** | 69.9% | 18.5 ms | 274.2 ms |
| **guided** | 46.29% | 8.64 px | 70.1% | 25.7 ms | 281.4 ms |
| **joint_bilateral** | 18.08% | 4.34 px | 70.1% | 41.6 ms | 297.4 ms |
| **wls** | 27.21% | 19.49 px | 31.4% | 113.7 ms | 369.5 ms |
| **confidence_edge** | **1.85%** | **1.56 px** | 14.0% | 105.6 ms | 361.4 ms |

The filter ablation highlights a fundamental trade-off: `median_valid` slightly improves raw disparity with minimal latency overhead (18.5 ms). `confidence_edge` reduces Bad@2.0 from 10.04% down to 1.85%, but at the expense of discarding 86% of pixels. For high-throughput robotics, `median_valid` or moderate WLS provides the most practical balance.

### CPU Optimization & Pareto Frontier

To understand hardware execution, component-level profiling was measured across resolutions (720p, 360p, 180p) and CPU thread allocations (1 to 16 threads).

![CPU Pareto frontier and multi-threading scaling](assets/fig05_cpu_pareto.png)
*Figure 5: CPU optimization analysis. (a) Latency versus accuracy Pareto frontier for FAST, BALANCED, and QUALITY presets. (b) Multi-threading speedup curves at 360p resolution. [Vector PDF](assets/fig05_cpu_pareto.pdf)*

```text
Table 5: Pipeline Presets and Multiscale Performance
```

| Preset | Scale Factor | Disparities | Block Size | Post-Filter | Latency (ms) | Frame Rate | Bad@2.0 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FAST** | $0.5\times$ (360p) | 80 px | 5 px | `median_valid` | **84.9 ms** | **11.8 FPS** | 28.4% |
| **BALANCED** | $0.5\times$ (360p) | 80 px | 5 px | `wls` | 256.5 ms | 3.9 FPS | 26.7% |
| **QUALITY** | $1.0\times$ (720p) | 160 px | 7 px | `wls` | 1328.5 ms | 0.8 FPS | **24.1%** |

Resolution downscaling is the dominant performance lever: downscaling 720p ($1280 \times 720$) to 360p ($640 \times 360$) reduces execution time from 589.8 ms down to 72.7 ms (an $8.1\times$ speedup). Thread scaling achieves near-linear speedup from 1 to 4 threads (183.4 ms to 100.5 ms), plateauing beyond 8 threads due to memory bandwidth limits.

---

## 4. Level 3: Gunung Field Transfer & RealSense Ground Truth

To test classical stereo outside controlled laboratory settings, the calibrated pipeline was deployed across four outdoor volcanic terrains on Mount Tangkuban Parahu, Indonesia:
1. `puncakTinggi`: High-altitude volcanic ridge with rough basalt rock and direct sunlight.
2. `puncakWarung`: Open roadside stall with corrugated tin roofing and deep shadow occlusions.
3. `tanggaBiru`: Steep trail stairway with painted blue steel safety railings.
4. `tanggaKuning`: Natural soil and stone steps bounded by painted yellow railings.

![Four Gunung volcanic terrains evaluation](assets/fig03_gunung.png)
*Figure 3: Four Gunung volcanic terrains evaluated under calibrated stereo and active IR RealSense ground truth. [Vector PDF](assets/fig03_gunung.pdf)*

### Sensor Setup and Registration Protocol

- **Primary Stereo Rig:** Dual calibrated cameras ($1280 \times 720$, baseline $B = 97.50$ mm, focal length $f_x = 964.68$ px).
- **Metric Ground Truth:** Co-mounted Intel RealSense D435i active IR stereo depth sensor ($640 \times 480$ native depth).
- **Extrinsic Registration:** 3D points unprojected from RealSense depth via $K_{\mathrm{rs}}$ were matched to 2D SIFT features in the rectified stereo left view. Rigid translation $T$ and rotation $R$ were solved via PnP RANSAC ($3.0$ px reprojection threshold).
- **Z-Buffered Forward Projection:** RealSense depth was projected into the $1280 \times 720$ stereo left coordinate frame using z-buffering (closer points overwrite farther points).

```text
Table 6: RealSense Registration and Metric Depth Validation Across Terrains
```

| Terrain | Status | SIFT Inliers | Reproj Med (px) | Translation $T$ (m) | Eval Pixels | Depth MAE (m) | Depth Med (m) | $\delta < 1.25$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **puncakTinggi** | **VERIFIED** | 106 | 1.25 px | $[-0.011, -0.035, -0.000]$ | 76,980 | 0.234 m | **0.059 m** | **88.4%** |
| **puncakWarung** | **OUT_OF_RANGE**| 0 | N/A | N/A | 0 | Unmeasured | Unmeasured | N/A |
| **tanggaBiru** | **VERIFIED** | 77 | 0.73 px | $[-0.026, -0.044, -0.030]$ | 13,759 | 0.500 m | **0.076 m** | 70.5% |
| **tanggaKuning** | **VERIFIED** | 32 | 1.19 px | $[-0.017, -0.034, -0.013]$ | 38,613 | 0.211 m | **0.061 m** | **88.2%** |

### Honest Reporting on Far-Field Geometry

In `puncakWarung`, the primary structures of interest are located at distances beyond 15 meters. The Intel RealSense D435i active IR emitter has an effective operating range of roughly 0.3 to 6.0 meters in daylight; beyond 10 meters, active IR returns are nonexistent. Consequently, 2D-3D SIFT consensus cannot be established.

Rather than interpolating or fabricating false ground truth, `puncakWarung` is explicitly reported as `OUT_OF_RANGE`. StereoSGBM successfully reconstructs the scene (achieving a photometric warp MAE of 3.887 intensity levels), but quantitative depth error against RealSense is honestly reported as unmeasured.

Across the three valid near-field terrains, classical SGBM achieves median metric depth errors of **5.9 cm** (`puncakTinggi`), **7.6 cm** (`tanggaBiru`), and **6.1 cm** (`tanggaKuning`). Over 88% of evaluated surface points fall within 25% of true distance ($\delta < 1.25$).

---

## 5. Cross-Dataset Generalization

To examine how classical stereo generalizes across radically different domains, Figure 7 compares indoor benchmark performance with outdoor volcanic terrain reconstruction side-by-side.

![Cross-dataset generalization comparison](assets/fig07_cross_dataset.png)
*Figure 7: Cross-dataset generalization. Row A shows Middlebury indoor benchmark scene (Jadeplant). Row B shows outdoor volcanic terrain (puncakTinggi) evaluated against registered RealSense metric depth. [Vector PDF](assets/fig07_cross_dataset.pdf)*

Classical stereo matching requires no domain-specific fine-tuning:
- In indoor scenes with high-frequency plant textures and complex depth layers (Row A), SGBM recovers intricate geometric boundaries with 9.36 px disparity MAE (0.090 m depth MAE).
- In outdoor volcanic scenes with rocky gravel, boulders, and harsh shadows (Row B), the same algorithm with calibrated parameters yields 5.89 cm median depth error.

---

## 6. What I Learned

1. **Geometry precedes matching:** Correct epipolar rectification ($y_L = y_R$) is a strict physical prerequisite. Sub-pixel calibration errors cannot be corrected by downstream filters or loss functions.
2. **Semi-global smoothness preserves boundaries:** Local block matching collapses in untextured terrain zones. SGBM directional dynamic programming propagates smoothness across flat surfaces without washing out object contours.
3. **Filtering involves strict trade-offs:** Post-processing filters like confidence-masked WLS can reduce bad-pixel rates down to 1.85%, but they discard up to 86% of valid scene points.
4. **Scale is the primary CPU lever:** Downscaling image resolution by $0.5\times$ yields an $8.1\times$ speedup, enabling real-time operation (11.8 to 18.5 FPS) on commodity CPU hardware.
5. **Small networks can learn spatial features:** A 1,054-parameter CNN achieves 94.3% accuracy on held-out test patches in 5.2 seconds of CPU training. However, a zero-parameter hand-designed MAD baseline achieves 91.1% in 0.001 ms per patch.
6. **Honesty in experimental limits:** RealSense active IR stereo cannot measure depth beyond 10 meters in outdoor sunlight. Acknowledging sensor range boundaries and reporting far-field scenes as unmeasured is essential for scientific integrity.

---

## 7. Reproduce

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

Runs quick end-to-end verification across synthetic stereo, Middlebury sample, Gunung sample, and CNN architecture:

```bash
python reproduce.py --quick
```

### Full Benchmark Suite

Executes the complete research suite across all modules, saves CSV and JSON metrics, and generates all publication figures and animated GIF:

```bash
python reproduce.py --full
```

Generated outputs in `results/`:
- `synthetic.csv`: Procedural benchmark across 6 perturbations and 3 matchers.
- `middlebury.csv`: Middlebury 2014 evaluation across 15 training scenes.
- `gunung.csv`: Four Gunung volcanic terrains evaluation and RealSense metrics.
- `optimization.csv`: Filter ablation and SGBM block-size sensitivity data.
- `best_params.json`: CPU multiscale, threading, and preset profile benchmarks.
- `cnn_metrics.json`: TinyStereoNet and classical MAD test metrics.
- `environment.json`: Hardware, compiler, and OS provenance record.

### Individual Modular Commands

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

### Run Unit Tests

```bash
python -m pytest -q
```

---

## References

- D. Scharstein and R. Szeliski, "A taxonomy and evaluation of dense two-frame stereo correspondence algorithms," *International Journal of Computer Vision*, 47(1-3):7-42, 2002. https://doi.org/10.1023/A:1014573219977
- H. Hirschmuller, "Stereo processing by semiglobal matching and mutual information," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(2):328-341, 2008. https://doi.org/10.1109/TPAMI.2007.1166
- D. Scharstein, H. Hirschmuller, et al., "High-resolution stereo datasets with subpixel-accurate ground truth," *German Conference on Pattern Recognition (GCPR)*, 2014. https://vision.middlebury.edu/stereo/data/scenes2014/
- J. Zbontar and Y. LeCun, "Stereo matching by training a convolutional neural network to compare image patches," *Journal of Machine Learning Research*, 17(1):2287-2318, 2016.
- OpenCV StereoSGBM Reference: https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
- KindlyGentleman/fft-diffraction: https://github.com/KindlyGentleman/fft-diffraction
