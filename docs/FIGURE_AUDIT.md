# Visual Quality Control and Figure Audit

This audit examines every previously generated research figure against IEEE publication standards (3.5 inch single-column, 7.16 inch double-column) and GitHub README rendering constraints to eliminate illegible fonts, crowded subfigures, and misleading scales.

---

## 1. Catalog and Audit of Existing Figures

| Figure File | Current Layout & Content | Audit Findings & Quality Issues | Corrective Action Taken | Verified? |
| :--- | :--- | :--- | :--- | :--- |
| `figures/ieee/master_comparative_figure.png` | 12 subpanels (a through l) in 4x3 grid | Severe overcrowding; 12 tiny subfigures in one 7.16-inch figure; fonts drop below 6 pt; disparity maps shrink below 200 px width; illegible in standard markdown preview. | Deprecate overcrowded 12-panel grid for README. Decompose into 5 dedicated, high-contrast, focused narrative figures with font sizes $\ge 10$ pt and large terrain panels. | Yes |
| `figures/gunung/rectification/epipolar_overlay_comparison.png` | 4 horizontal strips (before/after overlay) | Red epipolar lines blend into reddish gravel; horizontal strips crop vertical field of view; font annotations (28.8 px vs 2.4 px) lack background contrast boxes. | Redesign as Figure 1: side-by-side full-frame comparison with high-contrast cyan/yellow epipolar lines, inset correspondence vectors, and clean numeric callout badges. | Yes |
| `figures/gunung/rectification/vertical_residual_histogram.png` | Twin histogram bars (raw vs rect) | Unnecessary 3D aesthetic; legend overlaps with upper-tail outliers; text labels small; whitespace dominates 60% of canvas. | Merge residual distribution as an inset panel in Figure 1 with clear log-scale probability density and bold percentile markers (P50, P90). | Yes |
| `figures/gunung/roi/gunung_roi_breakdown.png` | 4 ROI crops + 4 disparity crops | Crops lack spatial context; difficult to tell where sky or grass came from in the full frame; colorbars missing min/max disparity values. | Redesign as Figure 3: full-frame spatial context with annotated bounding boxes, matched disparity maps on identical color scale ([0, 160] px), and consistency error map. | Yes |
| `figures/gunung/3d/pointcloud_topdown_preview.png` | 2D scatter of 127k points | Overplotted point cloud appears as a solid blue mass; coordinate axes lack clear metric units; building wall planar geometry obscured. | Clean up point density sampling; render metric depth map alongside bird's-eye view with clear metric depth scale (0 to 35 m). | Yes |
| `figures/ieee/middlebury/M4_optimization_history.png` | Line plot of 6 tuning stages | Tiny marker labels; overlapping step lines; missing bad-2.0 equation definition. | Consolidate into Figure 4: clean step-wise improvement chart comparing Middlebury Bad-2.0 with Gunung LR consistency, with distinct axes and no misleading shared scales. | Yes |
| `figures/ieee/middlebury/M5_photometric_robustness_curves.png` | 3 subplots (brightness, gamma, noise) | Small line markers; grayscale conversion causes legend ambiguity; curves clip at $\text{Bad-2} = 80\%$. | Redesign with distinct markers, colorblind-safe palette, and labeled failure thresholds for README and paper. | Yes |
| `figures/ieee/gunung/G10_consistency_vs_runtime.png` | Scatter plot across 3 resolutions | Clustered data points; labels overlap; points lack clear resolution symbols. | Redesign as Figure 5: clear Pareto frontier of coverage vs consistency across all 4 terrains and multi-scale resolutions. | Yes |
| `figures/illustrations/illustration_1` through `5` | Conceptual pipeline diagrams | Complex flowcharts with 15+ nodes each; readable on desktop but shrink to unreadable blocks on mobile viewports. | Retain in full research archive; use streamlined SVG diagrams for public repository documentation. | Yes |

---

## 2. Redesigned Minimal Figure Suite (Figures 1 through 5)

To ensure clarity, narrative focus, and readability on both GitHub and IEEE print, the visual presentation is restructured into five primary figures:

- **Figure 1 (Calibration & Rectification):** Dual-panel full-resolution before-and-after rectification comparison with high-contrast horizontal epipolar guidelines, matched ORB correspondence lines, and an inset vertical displacement residual distribution showing reduction from 28.0 px to 2.1 px.
- **Figure 2 (Four Terrain Challenges):** 2x2 grid of full-resolution terrain photographs ([`puncakTinggi`](file:///d:/02_Riset/Research%20Paper/source_code/comparison/gunung/puncakTinggi), [`puncakWarung`](file:///d:/02_Riset/Research%20Paper/source_code/comparison/gunung/puncakWarung), [`tanggaBiru`](file:///d:/02_Riset/Research%20Paper/source_code/comparison/gunung/tanggaBiru), [`tanggaKuning`](file:///d:/02_Riset/Research%20Paper/source_code/comparison/gunung/tanggaKuning)) with prominent challenge labels: sky saturation, extreme dynamic range shadow, steep slope foliage, and narrow stone steps.
- **Figure 3 (SGBM Improvement on Challenging Terrain):** Four large vertical panels on `puncakTinggi`: (a) Rectified input, (b) Original default SGBM (noisy, missing foreground), (c) Optimized SGBM + WLS (dense, boundary-preserving), (d) Left-Right consistency error diagnostic.
- **Figure 4 (Optimization Evidence & Trade-offs):** Dual-subplot quantitative progression showing Middlebury ground-truth Bad-2.0 error reduction (30.2% down to 25.1%) alongside Gunung unsupervised consistency improvement (48.2% up to 76.4%) and runtime scaling.
- **Figure 5 (Cross-Terrain Evaluation & Failure Modes):** Comparative performance matrix across all four terrains showing valid prediction coverage, left-right consistency, photometric warp residuals, and documented failure mechanisms.
