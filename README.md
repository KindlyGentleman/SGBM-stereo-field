# Playing with Stereo Matching

A small experiment investigating how far classical stereo matching (StereoSGBM and edge-aware WLS filtering) can go on four rugged outdoor terrains after physical camera calibration, verified against Middlebury ground truth and registered Intel RealSense reference depth.

![Stereo matching evolution across four mountain terrains](assets/stereo_loop.gif)

## Why stereo gets difficult outdoors

Deep neural networks dominate modern stereo benchmarks, but their compute and memory demands make them heavy for small embedded robots and low-power field platforms. Classical algorithms like Semi-Global Block Matching (StereoSGBM) run deterministically on a standard CPU with a minimal memory footprint.

On paper, SGBM needs only two conditions: collinear horizontal scanlines and roughly Lambertian surface reflectance. In unstructured volcanic environments, nature refuses to cooperate. Solar glare saturates sensors, wind-blown shrubs create micro-occlusions, volcanic gravel repeats fine texture patterns, and mechanical vibration knocks dual cameras out of collinear alignment. I wanted to see how far classical stereo reconstruction could be pushed using real camera calibration, sequential tuning, and edge-preserving filtering, and where classical matching hits fundamental physical limits.

## The cameras needed calibration first

Before touching matching parameters, I inspected the camera geometry. Running feature matching (ORB with RANSAC fundamental matrix estimation) on raw image pairs from the Gunung collection revealed a severe physical misalignment: a median vertical correspondence displacement of **28.0 px** in `puncakTinggi`. Standard stereo algorithms search strictly along horizontal scanlines ($\Delta y = 0$). With a 28 px vertical jump, matching failed immediately.

I loaded the physical calibration archive (`stereo_params_advanced.npz`) containing the camera matrices, distortion coefficients, rotation matrix $R$, and translation vector $T$ ($B = 97.50$ mm baseline, $f \approx 964.7$ px). Using `cv2.stereoRectify` and `cv2.initUndistortRectifyMap`, I remapped the dual image planes into collinear epipolar alignment.

The calibrated rectification collapsed the median vertical displacement from **28.0 px down to 2.1 px** (90th percentile: 3.6 px) on `puncakTinggi`, and down to **1.0 px** on `tanggaBiru` and `tanggaKuning`. Only after this geometric correction could horizontal stereo matching succeed.

## From Middlebury to the mountain

To benchmark the pipeline fairly, I used two distinct domains: the controlled Middlebury 2014 indoor benchmark (with structured-light ground truth) and the real-world outdoor Gunung dataset.

![Cross-dataset pipeline validation on Middlebury and Gunung](assets/fig01_cross_dataset.png)

On Middlebury (Row 1), ground truth is measured in sub-pixel disparity. On Gunung (Row 2), the physical world requires metric depth in meters. The calibrated classical pipeline transferred reliably from indoor benchmarks to outdoor volcanic terrain, producing sharp contours while exposing invalid sky regions.

## Why my first disparity maps looked wrong

My earliest outdoor reconstruction attempts produced four severe artifacts:

1. **Near-field truncation:** The nominal 128 disparity levels at $1280 \times 720$ only represented objects down to $Z \approx 0.70$ m. Boulders and gravel within arms reach simply vanished from the disparity map.
2. **Sky plane hallucination:** In over-exposed sky (where 16.8% of pixels exceed intensity 250), unconstrained matching assigned arbitrary disparities to featureless white pixels, which smoothing filters then diffused into solid walls.
3. **Boundary halo corruption:** Applying standard bilateral filtering directly to raw floating-point disparity averaged zero-valued invalid sentinels with positive foreground disparities. This created a blurry sludge border around every rock.
4. **Consistency check false rejections:** OpenCV `createRightMatcher` returns negative disparities scaled by 16. Inverting coordinates without handling the sign convention caused 95% of valid matches to be falsely rejected.

Fixing these bugs required setting signed masks before filtering, enforcing an explicit disparity search range of 160 levels ($Z_{\min} \approx 0.56$ m), and stripping low-confidence regions before bilateral smoothing.

## Does edge preservation help?

I evaluated seven filter candidates under identical frozen SGBM inputs across both Middlebury and Gunung:

![Filter comparison across Middlebury and Gunung](assets/fig02_filters.png)

- **F0: Raw SGBM:** Ground baseline. Middlebury Bad-2.0: 29.09%, MAE: 1.27 px. Gunung median depth error: 0.055 m (5.5 cm) on 19.9% reference coverage.
- **F1: Legacy Bilateral Filter:** The buggy implementation. Blurring zeros corrupted boundaries, inflating Middlebury Bad-2.0 to 34.22% and more than doubling Gunung depth MAE from 0.308 m to 0.764 m.
- **F2: Validity-Aware Median Filter:** Restricting a 3x3 median blur strictly to valid pixels reduced Middlebury MAE to 1.24 px without boundary halo artifacts.
- **F3: Guided Filter:** Using the grayscale left image as guidance smoothed textures but caused edge bleeding on high-contrast terrain boundaries (Bad-2.0: 37.98%).
- **F4: Joint Bilateral Filter:** Color-guided bilateral smoothing achieved 31.60% Bad-2.0 on Middlebury and 0.401 m depth MAE on Gunung.
- **F5: Standard Disparity WLS Filter:** Using `cv2.ximgproc.createDisparityWLSFilter` ($\lambda = 8000$, $\sigma = 1.5$) snapped boundaries to physical object contours. Valid coverage on Gunung almost doubled from 19.9% to 37.6% with a median depth error of only 0.102 m (10.2 cm).
- **F6: Confidence-Aware Edge Preservation:** Rejecting low-confidence regions reduced Middlebury MAE to 0.79 px, but dropped outdoor coverage to 4.5% due to strict confidence penalties on weak natural textures.

## Checking against real depth

To validate the outdoor reconstructions against metric ground truth, I inspected the Intel RealSense D435i depth recordings (`depth_0000.npz`) accompanying each Gunung capture.

![Error maps on Middlebury disparity and Gunung metric depth](assets/fig03_errors.png)

Using SIFT feature correspondences between RealSense RGB ($640 \times 480$) and rectified stereo left ($1280 \times 720$), I solved for the rigid Euclidean extrinsics $(R, T)$ via PnP RANSAC. The physical translation converged to $T \approx [-1.5\text{ cm}, -3.8\text{ cm}, -1.5\text{ cm}]$, matching the camera mount on the rigid rig:

- **puncakTinggi:** 106 inliers, median reprojection error 1.25 px. Over 140,638 jointly valid pixels ($Z < 4.0$ m), the median metric depth error was **0.083 m (8.3 cm)**, with 77.9% of predictions within $\delta < 1.25$.
- **tanggaBiru:** 77 inliers, median reprojection error 0.73 px. Median depth error was **0.151 m (15.1 cm)**.
- **tanggaKuning:** 32 inliers, median reprojection error 1.19 px. Median depth error was **0.135 m (13.5 cm)**.
- **puncakWarung:** Registration was intentionally omitted. Scene features reside at distances beyond 15 meters, where RealSense active IR stereo cannot provide reliable depth. Following scientific protocol, target depth accuracy is not fabricated for this terrain.

## What actually improved?

Ablating matching parameters in sequence demonstrated clear, measurable gains:

![Sequential optimization evidence across Middlebury and Gunung](assets/fig04_optimization.png)

| Stage | Intervention | Middlebury Bad-2.0 (%) | Middlebury MAE (px) | Gunung Coverage (%) | Gunung Consistency (%) | Runtime (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S0** | Rectified Default SGBM | 43.94% | 5.64 px | 56.8% | 35.6% | 825 ms |
| **S1** | Search Range 160 px | 47.64% | 6.70 px | 54.3% | 33.5% | 920 ms |
| **S2** | Block Size 7 | 37.30% | 3.93 px | 57.0% | 52.0% | 994 ms |
| **S3** | Smoothness Penalties ($P_1=150, P_2=1200$) | 27.92% | 1.57 px | 59.4% | 54.8% | 1003 ms |
| **S4** | 3-Way Mode (`MODE_SGBM_3WAY`) | 28.57% | 1.91 px | 60.4% | 52.7% | **264 ms** |
| **S5** | Histogram Matching | 31.79% | 3.60 px | 55.0% | 49.7% | 241 ms |
| **S6** | Speckle & Uniqueness Filtering | 28.31% | 3.20 px | **83.4%** | 40.6% | 243 ms |
| **S7** | Edge-Aware WLS Filter | **27.87%** | **1.10 px** | 77.1% | 26.3% | 303 ms |

Middlebury completeness-aware Bad-2.0 error dropped from **43.9% to 27.9%**, while MAE fell by **80.5%** (5.64 px to 1.10 px). 3-way matching cut runtime by **3.8x** (1003 ms to 264 ms) on standard CPU.

## Four terrains, different failure modes

Evaluating across all four terrains confirmed that outdoor reconstruction challenges vary significantly by geometry and lighting:

![Cross-terrain performance and disparity maps](assets/fig05_terrains.png)

1. **puncakTinggi (Ridge):** Strongest quantitative performance (median depth error 8.3 cm), but 16.8% saturated sky requires explicit masking to avoid flying planar artifacts.
2. **puncakWarung (Stall):** Highest coverage (85.9%) but lowest round-trip consistency (4.0%), reflecting deep porch shadows and distant background (>15 m).
3. **tanggaBiru (Blue Stairway):** High consistency (43.9%) and reliable railing reconstruction, but thin foliage causes micro-discontinuities.
4. **tanggaKuning (Yellow Stairway):** Repeating metal railing bars create periodic correlation ambiguities, but step edges achieve 0.135 m median depth error.

## What I learned

1. **Geometry matters before filtering:** Tuning matching parameters on unrectified imagery with 28 px vertical error was useless. Calibrated epipolar rectification accounted for almost all initial gains.
2. **Smooth disparity is not necessarily accurate:** Filters that fill untextured sky or blur invalid sentinels look visually pleasing but destroy metric accuracy.
3. **Missing predictions matter:** Completeness-aware metrics prevent matchers from gaming accuracy scores by rejecting difficult pixels.
4. **Active IR reference sensors have strict range envelopes:** Intel RealSense D435i provides verified sub-decimeter ground truth up to 4 meters, but cannot evaluate distant outdoor mountain scenes beyond 10 meters.

## Reproduce

### Requirements

```bash
python -m pip install -r requirements.txt
```

### Quick Verification (< 5 seconds)

Runs calibration verification, rectified matching, and metrics on single test scenes:

```bash
python reproduce.py --quick
```

### Full Benchmark Experiment

Runs complete benchmark evaluation across all Middlebury scenes and all four Gunung terrains:

```bash
python reproduce.py --full
```

Outputs:
- `results/metrics.csv`
- `results/best_params.json`
- `results/validation.json`

### Regenerate Visuals

Regenerates all publication figures and the hero looping GIF:

```bash
python visualize.py
```

Outputs:
- `assets/fig01_cross_dataset.png` and `.pdf`
- `assets/fig02_filters.png` and `.pdf`
- `assets/fig03_errors.png` and `.pdf`
- `assets/fig04_optimization.png` and `.pdf`
- `assets/fig05_terrains.png` and `.pdf`
- `assets/fig06_edge_vs_accuracy.png` and `.pdf`
- `assets/stereo_loop.gif`

### Run Unit Tests

```bash
python -m pytest -q
```

### Custom Data Paths

Provide custom paths explicitly if running from another location:

```bash
python reproduce.py --full --calib path/to/stereo_params.npz --gunung path/to/gunung --middlebury path/to/trainingQ
```

## References

- D. Scharstein and R. Szeliski, "A taxonomy and evaluation of dense two-frame stereo correspondence algorithms," *International Journal of Computer Vision*, 47(1-3):7-42, 2002. https://doi.org/10.1023/A:1014573219977
- H. Hirschmuller, "Stereo processing by semiglobal matching and mutual information," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 30(2):328-341, 2008. https://doi.org/10.1109/TPAMI.2007.1166
- D. Scharstein, H. Hirschmuller, et al., "High-resolution stereo datasets with subpixel-accurate ground truth," *German Conference on Pattern Recognition (GCPR)*, 2014. https://vision.middlebury.edu/stereo/data/scenes2014/
- OpenCV StereoSGBM Documentation: https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html
- OpenCV Disparity WLS Filter Tutorial: https://docs.opencv.org/4.x/d3/d14/tutorial_ximgproc_disparity_filtering.html
- SciencePlots: https://github.com/garrettj403/SciencePlots
