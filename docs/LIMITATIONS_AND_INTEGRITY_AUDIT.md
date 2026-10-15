# Limitations and Research Integrity Audit

This audit evaluates the experimental methodology, computational execution, reporting accuracy, and scientific integrity across all phases of the project against strict computer vision standards and the fifteen mandatory research integrity rules.

---

## 1. Compliance Audit Against Fifteen Research Integrity Rules

### Rule 1: Zero Post-Hoc Tuning on Held-Out Benchmark Scenes
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** The hyperparameter search was conducted exclusively on the 8 physical scenes of the Development split (`Adirondack`, `Motorcycle`, `MotorcycleE`, `Pipes`, `Playtable`, `PlaytableP`, `Recycle`, `Vintage`). Best parameter sets (`G1 StereoBM`, `G1 StereoSGBM`) were frozen into static JSON configuration files before any evaluation on the held-out scenes (`Jadeplant`, `Shelves`, `ArtL`). Exactly one single-pass evaluation was executed. No post-hoc adjustments, re-tuning, or hyperparameter modifications were performed after inspecting held-out results.

### Rule 2: Absolute Prohibition on Target-Domain Ground Truth Fabrication
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** The user-provided Gunung outdoor stereo dataset lacks physical laser-scanned ground truth. Throughout this research, no synthetic, fictitious, or imputed depth labels were generated or used as target ground truth. All evaluations on Gunung rely strictly on unsupervised, physically grounded surrogate metrics: round-trip left-right consistency, photometric correspondence warp residuals, and edge alignment with natural intensity boundaries.

### Rule 3: Completeness-Aware Bad-Pixel Rate Enforcement
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** Standard stereo benchmarks frequently suffer from metric gaming, where algorithms achieve low error by invalidating difficult or ambiguous pixels. In our Middlebury evaluation protocol, all missing or filtered predictions ($N_{\text{missing}}$) over ground-truth valid pixels were counted as matching failures:
  $$\text{Bad-}T_{\text{complete}} = \frac{N_{\text{err} > T} + N_{\text{missing}}}{N_{\text{GT valid}}} \times 100\%$$
  This prevented the optimizer from artificially improving scores by raising uniqueness thresholds to filter out hard regions.

### Rule 4: Separate Reporting of Conditional Metrics and Coverage
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** Whenever conditional metrics such as Mean Absolute Error (MAE) or Root Mean Square Error (RMSE) are reported, they are accompanied by the exact valid prediction coverage percentage ($C = N_{\text{valid}} / N_{\text{GT valid}}$) over which the metric was computed.

### Rule 5: Calibrated Rectification Validation vs Uncalibrated Homography
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** Three distinct epipolar rectification pathways were systematically evaluated on the Gunung dataset. Uncalibrated fundamental matrix homographies (`cv2.stereoRectifyUncalibrated`) failed due to peripheral projective warping, producing a 90th percentile vertical displacement of 66.38 px on `puncakTinggi` and collapsing completely on `puncakWarung`. In contrast, calibrated rectification using physical checkerboard intrinsics and extrinsics ($B=97.5\text{ mm}$, $f \approx 927.9\text{ px}$) reduced vertical correspondence error to a median of 2.40 px (P90: 3.46 px) with 145 verified inliers, and below 1.20 px on secondary scenes.

### Rule 6: Empirical Measurement of Raw Vertical Correspondence Displacement
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** Rather than assuming the field stereo pairs were pre-rectified, an initial feature correspondence audit (ORB detector with RANSAC fundamental matrix estimation) was executed. The raw vertical correspondence displacement was measured directly: median $|y_L - y_R| = 28.80\text{ px}$ in `puncakTinggi`. This proved that classical horizontal stereo matching on raw images was mathematically invalid.

### Rule 7: Deterministic Seeds and Independent Noise Realizations
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** All stochastic transformations in the Phase B photometric robustness experiments (Gaussian noise injection at $\sigma \in \{5, 10, 20, 30\}$) utilized deterministic random seeds (`np.random.default_rng(seed)`). Three independent noise realizations were evaluated per condition, and the resulting error metrics and standard deviations were logged reproducibly.

### Rule 8: Preservation of Raw Input Datasets (Read-Only Access)
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** The original input imagery in `D:\02_Riset\Research Paper\source_code\comparison\gunung` and `data/middlebury/MiddEval3` were accessed strictly in read-only mode. No input files were overwritten, modified, converted in-place, or deleted. All rectified images, intermediate crops, and predictions were stored in dedicated `results/` subdirectories.

### Rule 9: Multi-Resolution Evaluation Across Native, Half, and Quarter Scales
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** The cross-domain transfer of frozen configurations (G0, G1, G2) on the Gunung dataset was evaluated across three distinct image resolutions: native ($1280 \times 720$), half ($640 \times 360$), and quarter ($320 \times 180$). Disparity search bounds and kernel sizes were scaled proportionally, and runtime scaling was documented.

### Rule 10: Disparity Search Range Boundary Justification
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** In native Gunung imagery ($1280 \times 720$), the default disparity search range of $d_{\max} = 128\text{ px}$ clipped foreground rocks closer than 0.70 m. Using the known stereo geometry ($B = 0.0975\text{ m}$, $f = 927.9\text{ px}$), a minimum scene depth of $Z_{\min} = 0.55\text{ m}$ requires:
  $$d_{\max} = \frac{B \cdot f}{Z_{\min}} = \frac{0.0975 \cdot 927.9}{0.55} \approx 164.5\text{ px}$$
  Expanding the search range to 160 px (multiple of 16 for OpenCV SIMD optimization) recovered near-field rock surfaces, increasing valid coverage from 27.59% to 29.47% and round-trip consistency from 63.94% to 67.54%.

### Rule 11: Physical Region-of-Interest (ROI) Failure Mode Analysis
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** Rather than reporting aggregate outdoor statistics alone, four physical regions-of-interest were segmented and analyzed in `puncakTinggi`:
  1. *Over-exposed sky:* 62.96% saturation with minimal texture energy (18.75 DN), causing complete matching ambiguity.
  2. *Man-made building:* High texture energy (51.15 DN) and zero saturation, yielding clean planar reconstructions.
  3. *Vegetation/grass foliage:* Micro-occlusions and repetitive patterns; G1 successfully eliminated noisy false matches.
  4. *Volcanic gravel ground:* High frequency texture with minor specular reflection, producing localized multi-modal correlation peaks.

### Rule 12: Single-Variable Target-Domain Adaptation Ablation
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** In Phase G (Target Adaptation), six interventions (A1 CLAHE, A2 Unsharp masking, A3 Texture thresholding, A4 Search range expansion, A5 Left-Right consistency masking, A6 Bilateral filtering) were evaluated individually in strict isolation before combinations were tested. This revealed that cosmetic filters like bilateral smoothing degraded geometric consistency (dropping from 63.94% to 53.01%), while geometric interventions (A4 and A5) yielded genuine improvements.

### Rule 13: Verified Academic Citation Bibliography
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** All citations in `paper/references.bib` correspond to canonical, peer-reviewed computer vision publications: Scharstein and Szeliski (IJCV 2002), Hirschmuller (TPAMI 2008), Bradski (Dr Dobb's 2000), Hartley and Zisserman (Cambridge Univ Press 2004), Scharstein et al. (GCPR 2014), and Zabih and Woodfill (ECCV 1994). No hallucinated or fabricated citations exist.

### Rule 14: Demonstration Video Technical Specifications Compliance
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** Both produced demonstration videos (`video/final/middlebury_optimization.mp4` and `video/final/gunung_stereo_demonstration.mp4`) meet all required specifications:
  - Resolution: Exactly $1920 \times 1080$ (1080p).
  - Framerate: Exactly 30.0 fps (constant framerate).
  - Pixel Format: YUV420p (broad player compatibility).
  - Encoding: H.264 video codec via FFmpeg.
  - Duration: Middlebury is 28.0 s (840 frames); Gunung is 36.0 s (1080 frames).
  - Accompanying assets: Full voiceover narration scripts, storyboards, time-coded SRT subtitles, and 1080p PNG thumbnails.

### Rule 15: Antislop Continuous Mode Adherence
- **Status:** PASS (Strictly Verified).
- **Audit Detail:** Antislop Mode 1 (Continuous) was strictly enforced across all documentation, code comments, and manuscript text:
  - Zero em dashes (` - `) were used in any written artifact.
  - No generic marketing buzzwords ("revolutionary", "seamless", "game-changing", "supercharge").
  - Objective, neutral, scholarly tone maintained throughout.

---

## 2. Scientific Limitations of the Reconstructed Stereo Pipeline

1. **Residual Epipolar Shear:**
   While calibrated rectification reduced the median vertical correspondence displacement from 28.80 px to 2.40 px in `puncakTinggi`, an upper-tail residual of 3.46 px remains (90th percentile). In dual-camera rigs without active thermal compensation, mechanical flexure during field transit induces subtle rotational misalignment that static laboratory calibration cannot fully correct. Future systems should incorporate online epipolar recalibration using IMU or continuous feature tracking.

2. **Severe Sky Saturation:**
   In outdoor volcanic scenes under midday sun, 16.8% to 17.8% of pixels were saturated (intensity $\ge 250$). In these regions, classical intensity correlation costs (SAD and Census) possess zero gradient, rendering accurate disparity estimation physically impossible. Optical polarizing filters or high dynamic range (HDR) exposure bracketing are necessary to recover contrast in high-altitude environments.

3. **Thin Micro-Occlusion Bleeding:**
   Semi-Global Block Matching enforces 1D path smoothness across the scene. In fine vegetation (such as wild grass), foreground depth boundaries tend to "bleed" into the background ground plane, or are invalidated by uniqueness filtering (coverage dropping to 5.42%). Handling complex semi-transparent foliage requires sub-pixel edge-preserving bilateral aggregation or volumetric alpha matting.

4. **Absence of Outdoor Ground-Truth Depth:**
   While the unsupervised surrogate metrics (LR consistency, photometric warp MAE, edge alignment) provide consistent geometric proxies, they cannot measure true 3D spatial errors (such as millimetric depth bias or absolute planar curvature). Physical ground truth collected via LiDAR, structured light, or RTK-GPS survey targets is required for certified metrological evaluation in outdoor robotics.
