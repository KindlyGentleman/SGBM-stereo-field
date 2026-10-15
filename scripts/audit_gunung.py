import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np

GUNUNG_DIR = Path(r"D:\02_Riset\Research Paper\source_code\comparison\gunung")
CALIB_PARAM = Path(r"D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz")
RECT_PARAM = Path(r"D:\02_Riset\Research Paper\source_code\comparison\light\indoor_sekreHimafi\stereo\rectification_params.npz")

scenes = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]

audit_results = {}

for s in scenes:
    s_dir = GUNUNG_DIR / s
    stereo_dir = s_dir / "stereo"
    l_path = stereo_dir / "left_0000.png"
    r_path = stereo_dir / "right_0000.png"

    l_img = cv2.imread(str(l_path))
    r_img = cv2.imread(str(r_path))

    l_h, l_w, _ = l_img.shape
    r_h, r_w, _ = r_img.shape

    l_gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
    r_gray = cv2.cvtColor(r_img, cv2.COLOR_BGR2GRAY)

    l_sat250 = float(np.mean(l_gray >= 250) * 100.0)
    r_sat250 = float(np.mean(r_gray >= 250) * 100.0)

    l_sat0 = float(np.mean(l_gray == 0) * 100.0)
    r_sat0 = float(np.mean(r_gray == 0) * 100.0)

    with open(l_path, "rb") as f:
        l_sha = hashlib.sha256(f.read()).hexdigest()
    with open(r_path, "rb") as f:
        r_sha = hashlib.sha256(f.read()).hexdigest()

    audit_results[s] = {
        "left_shape": [l_h, l_w],
        "right_shape": [r_h, r_w],
        "left_size_bytes": l_path.stat().st_size,
        "right_size_bytes": r_path.stat().st_size,
        "left_sha256": l_sha,
        "right_sha256": r_sha,
        "left_mean_intensity": float(np.mean(l_gray)),
        "right_mean_intensity": float(np.mean(r_gray)),
        "left_sat_ge_250_pct": l_sat250,
        "right_sat_ge_250_pct": r_sat250,
        "left_sat_0_pct": l_sat0,
        "right_sat_0_pct": r_sat0,
    }

print(json.dumps(audit_results, indent=2))
