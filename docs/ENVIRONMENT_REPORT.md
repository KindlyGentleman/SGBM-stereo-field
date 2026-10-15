# Environment and Toolchain Audit Report

**Report Date:** 2026-10-15  
**Auditor:** Research Reproducibility Auditor  
**Operating System:** Windows 10/11 Pro (x64)  
**Workspace Directory:** `D:\03_Proyek\SGBM Research`  

---

## 1. Runtime Environment and Core Libraries

| Component | Detected Version | Target Specification | Status |
| :--- | :--- | :--- | :--- |
| **Python** | 3.13.7 (64-bit AMD64) | 3.11+ | Verified Compatible |
| **OpenCV (`cv2`)** | 4.14.0 | 4.x | Verified Compatible |
| **NumPy** | 2.5.3 | 1.24+ | Verified Compatible |
| **SciPy** | 1.16.2 | 1.11+ | Verified Compatible |
| **Matplotlib** | 3.10.0 | 3.7+ | Verified Compatible |
| **SciencePlots** | Available | Current | Verified Compatible |
| **Pandas** | 2.2.3 | 2.0+ | Verified Compatible |
| **scikit-image** | 0.26.0 | 0.22+ | Verified Compatible |
| **PyYAML** | 6.0.2 | 6.0+ | Verified Compatible |
| **pytest** | 9.1.1 | 7.0+ | Verified Compatible |
| **tqdm** | 4.67.1 | Current | Verified Compatible |
| **optuna** | 4.2.1 | Current | Verified Compatible |
| **imageio** | 2.38.1 | Current | Verified Compatible |

---

## 2. External System Toolchains

### Video Production
- **FFmpeg:** `7.0.1-full_build-www.gyan.dev`
- **Supported Encoders:** `libx264`, `aac`, `yuv420p` pixel format
- **Hardware Acceleration:** CPU H.264 rendering tested and verified.

### Scientific Document Compilation
- **LaTeX Distribution:** MiKTeX-pdfTeX 4.19 (MiKTeX 24.4)
- **Engine:** `pdflatex` available on system PATH
- **Manuscript Template:** Standard IEEEtran LaTeX class.

---

## 3. Dataset Integrity and Local Storage

1. **Middlebury Stereo 2014 / evaluation-v3 (Quarter-Resolution):**
   - Location: `data/middlebury/MiddEval3/`
   - Training Scenes: 15 pairs (`trainingQ/`)
   - Held-out Benchmark Test Scenes: 15 pairs (`testQ/`)
   - Format: PNG images (RGB), PFM ground-truth disparities (`disp0GT.pfm`), and non-occlusion masks (`mask0nocc.png`).
   - Calibration files: `calib.txt` with camera matrices and disparity search ranges.

2. **Gunung Outdoor Stereo Dataset:**
   - Location: `D:\02_Riset\Research Paper\source_code\comparison\gunung`
   - Scenes: `puncakTinggi`, `puncakWarung`, `tanggaBiru`, `tanggaKuning`
   - Resolution: 1280 x 720 color stereo pairs
   - Read-Only Mode: Strictly enforced. All processed outputs and figures stored under `results/` and `figures/`.

---

## 4. Antislop Compliance

Anti-slop rules are activated in **continuous execution mode** (Mode 1: During).
- Em dash (` - `) ban strictly adhered to in all written prose, figure captions, and documentation.
- Zero fabricated statistics, zero unmeasured accuracy claims.
- Code comments focus purely on algorithmic intent, geometric coordinate frames, and mathematical bounds.
