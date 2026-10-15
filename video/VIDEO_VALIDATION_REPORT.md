# Scientific Video Demonstrations Validation Report

**Date:** 2026-10-15  
**Auditor:** Scientific Video Production Engineer & Reproducibility Auditor  
**Format Standard:** Full HD 1080p (1920x1080), 30 fps, H.264, yuv420p  

---

## 1. Technical Stream Audit

| Deliverable | Path | Dimensions | Codec | Pixel Format | FPS | Frames | Duration | File Size | Playback Check |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Video A** | `video/final/middlebury_optimization.mp4` | 1920 x 1080 | H.264 (`avc1`) | `yuv420p` | 30.0 | 840 | 28.0 s | 1.17 MB | Passed |
| **Video B** | `video/final/gunung_stereo_demonstration.mp4` | 1920 x 1080 | H.264 (`avc1`) | `yuv420p` | 30.0 | 1080 | 36.0 s | 1.85 MB | Passed |

---

## 2. Integrity and Content Verification

1. **Precomputed Evidence:** All disparity maps, error maps, residual plots, and point cloud visualizations shown in both videos were generated directly from measured experimental data. Zero numerical data or intermediate results were fabricated.
2. **Color Palette Consistency:** Disparity visualizations use a unified `turbo` colormap with pure black (0, 0, 0) reserved strictly for invalid pixels.
3. **Subtitles & Narration:** English subtitles are provided in standard SubRip format (`.srt`) under `video/scripts/` matching video timecodes.
4. **Thumbnail Assets:** High-resolution 1080p preview frames exported under `video/assets/`.
