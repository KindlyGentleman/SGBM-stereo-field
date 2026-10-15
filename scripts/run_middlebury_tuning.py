"""
Milestone 2 (Phase C): Staged hyperparameter optimization and photometric preprocessing tuning.
Optimizes StereoBM and StereoSGBM on Middlebury development scenes, verifies on validation scenes,
and freezes the best candidate configurations (G1, G2).
"""

from __future__ import annotations
import itertools
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import yaml


from stereo_robustness.datasets import discover_middlebury_scenes
from stereo_robustness.stereo import (
    StereoBMConfig,
    StereoSGBMConfig,
    compute_disparity_bm,
    compute_disparity_sgbm,
)
from stereo_robustness.optimization import (
    TrialRecord,
    evaluate_bm_config_on_scenes,
    evaluate_sgbm_config_on_scenes,
)

OUT_DIR = Path("results/middlebury/optimization")
OUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_CFG_DIR = Path("results/middlebury/best_configs")
BEST_CFG_DIR.mkdir(parents=True, exist_ok=True)

with open("configs/middlebury_baseline.yaml", "r") as f:
    config = yaml.safe_load(f)

scenes_dict = discover_middlebury_scenes("data/middlebury/MiddEval3/trainingQ")

dev_scene_objs = [scenes_dict[s] for s in config["splits"]["dev_scenes"]]
val_scene_objs = [scenes_dict[s] for s in config["splits"]["val_scenes"]]

print(f"Loaded {len(dev_scene_objs)} Dev scenes and {len(val_scene_objs)} Val scenes.")

# --------------------------------------------------------------------------
# 1. OPTIMIZE STEREOBM
# --------------------------------------------------------------------------
print("\n--- STAGE 1: StereoBM Parameter Exploration ---")
bm_trials = []
trial_id = 1

# Bounded parameter combinations
bm_param_grid = {
    "blockSize": [9, 11, 15, 21],
    "uniquenessRatio": [5, 10, 15],
    "preFilterCap": [15, 31, 63],
    "textureThreshold": [0, 10],
    "speckleWindowSize": [50, 100],
    "speckleRange": [2, 32],
}

# Sample representative configurations across the search space
keys, values = zip(*bm_param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Select a well-distributed sample of 25 candidate configurations
rng = np.random.default_rng(42)
sampled_indices = sorted(rng.choice(len(combinations), size=min(25, len(combinations)), replace=False))
sampled_bm_cfgs = [combinations[i] for i in sampled_indices]
# Always include default baseline config as reference
sampled_bm_cfgs.insert(0, {
    "blockSize": 15,
    "uniquenessRatio": 15,
    "preFilterCap": 31,
    "textureThreshold": 10,
    "speckleWindowSize": 100,
    "speckleRange": 32,
})

for p in sampled_bm_cfgs:
    cfg = StereoBMConfig(
        blockSize=p["blockSize"],
        uniquenessRatio=p["uniquenessRatio"],
        preFilterCap=p["preFilterCap"],
        textureThreshold=p["textureThreshold"],
        speckleWindowSize=p["speckleWindowSize"],
        speckleRange=p["speckleRange"],
    )
    b2, b1, cov, mae, rmse, t_ms, per_sc = evaluate_bm_config_on_scenes(cfg, dev_scene_objs, prep_method="gray")
    bm_trials.append(TrialRecord(
        trial_id=trial_id,
        algorithm="StereoBM",
        params=p,
        preprocessing="gray",
        avg_dev_bad_2_0=b2,
        avg_dev_bad_1_0=b1,
        avg_dev_coverage=cov,
        avg_dev_mae=mae,
        avg_dev_rmse=rmse,
        avg_dev_runtime_ms=t_ms,
        per_scene_bad_2_0=per_sc,
    ))
    print(f"BM Trial {trial_id:02d} | blockSize={p['blockSize']:2d} unq={p['uniquenessRatio']:2d} pfc={p['preFilterCap']:2d} | Dev Bad2: {b2:.2f}% Cov: {cov*100:.1f}% Time: {t_ms:.1f}ms")
    trial_id += 1

# Rank BM candidates by completeness-aware Bad-2.0
bm_trials.sort(key=lambda t: t.avg_dev_bad_2_0)
best_bm_trial = bm_trials[0]
print(f"\nBest StereoBM Configuration: Trial {best_bm_trial.trial_id} with Dev Bad-2.0 = {best_bm_trial.avg_dev_bad_2_0:.2f}% (Coverage = {best_bm_trial.avg_dev_coverage*100:.1f}%)")

# --------------------------------------------------------------------------
# 2. OPTIMIZE STEREOSGBM
# --------------------------------------------------------------------------
print("\n--- STAGE 1: StereoSGBM Parameter Exploration ---")
sgbm_trials = []
trial_id = 1

sgbm_param_grid = {
    "blockSize": [3, 5, 7],
    "P1": [150, 300, 600],
    "P2": [600, 1200, 2400],
    "uniquenessRatio": [5, 10, 15],
    "speckleWindowSize": [50, 100],
    "speckleRange": [2, 32],
    "mode": [cv2.STEREO_SGBM_MODE_SGBM_3WAY, cv2.STEREO_SGBM_MODE_SGBM],
}

combinations_sgbm = []
for bs in [3, 5, 7]:
    for p1 in [150, 300, 600]:
        for p2 in [600, 1200, 2400]:
            if p2 <= p1:
                continue
            for unq in [5, 10, 15]:
                for sw in [50, 100]:
                    for sr in [2, 32]:
                        for mode in [cv2.STEREO_SGBM_MODE_SGBM_3WAY, cv2.STEREO_SGBM_MODE_SGBM]:
                            combinations_sgbm.append({
                                "blockSize": bs,
                                "P1": p1,
                                "P2": p2,
                                "uniquenessRatio": unq,
                                "speckleWindowSize": sw,
                                "speckleRange": sr,
                                "mode": mode,
                            })

sampled_indices_sgbm = sorted(rng.choice(len(combinations_sgbm), size=min(25, len(combinations_sgbm)), replace=False))
sampled_sgbm_cfgs = [combinations_sgbm[i] for i in sampled_indices_sgbm]
sampled_sgbm_cfgs.insert(0, {
    "blockSize": 5,
    "P1": 600,
    "P2": 2400,
    "uniquenessRatio": 10,
    "speckleWindowSize": 100,
    "speckleRange": 32,
    "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
})

for p in sampled_sgbm_cfgs:
    cfg = StereoSGBMConfig(
        blockSize=p["blockSize"],
        P1=p["P1"],
        P2=p["P2"],
        uniquenessRatio=p["uniquenessRatio"],
        speckleWindowSize=p["speckleWindowSize"],
        speckleRange=p["speckleRange"],
        mode=p["mode"],
    )
    b2, b1, cov, mae, rmse, t_ms, per_sc = evaluate_sgbm_config_on_scenes(cfg, dev_scene_objs, prep_method="gray")
    sgbm_trials.append(TrialRecord(
        trial_id=trial_id,
        algorithm="StereoSGBM",
        params=p,
        preprocessing="gray",
        avg_dev_bad_2_0=b2,
        avg_dev_bad_1_0=b1,
        avg_dev_coverage=cov,
        avg_dev_mae=mae,
        avg_dev_rmse=rmse,
        avg_dev_runtime_ms=t_ms,
        per_scene_bad_2_0=per_sc,
    ))
    print(f"SGBM Trial {trial_id:02d} | blockSize={p['blockSize']:2d} P1={p['P1']:4d} P2={p['P2']:4d} unq={p['uniquenessRatio']:2d} | Dev Bad2: {b2:.2f}% Cov: {cov*100:.1f}% Time: {t_ms:.1f}ms")
    trial_id += 1

sgbm_trials.sort(key=lambda t: t.avg_dev_bad_2_0)
best_sgbm_trial = sgbm_trials[0]
print(f"\nBest StereoSGBM Configuration: Trial {best_sgbm_trial.trial_id} with Dev Bad-2.0 = {best_sgbm_trial.avg_dev_bad_2_0:.2f}% (Coverage = {best_sgbm_trial.avg_dev_coverage*100:.1f}%)")

# --------------------------------------------------------------------------
# 3. PHOTOMETRIC PREPROCESSING COMPARISON (G2)
# --------------------------------------------------------------------------
print("\n--- STAGE 2: Photometric Preprocessing Evaluation on Best Configurations ---")
prep_methods = ["gray", "clahe", "hist_match", "global_norm", "contrast_norm", "census"]
prep_results = []

best_bm_cfg = StereoBMConfig(**best_bm_trial.params)
best_sgbm_cfg = StereoSGBMConfig(**best_sgbm_trial.params)

for pm in prep_methods:
    # Evaluate BM
    b2_bm, b1_bm, cov_bm, mae_bm, rmse_bm, t_bm, per_sc_bm = evaluate_bm_config_on_scenes(best_bm_cfg, dev_scene_objs, prep_method=pm)
    prep_results.append({
        "algorithm": "StereoBM",
        "preprocessing": pm,
        "dev_bad_2_0": b2_bm,
        "dev_bad_1_0": b1_bm,
        "dev_coverage": cov_bm,
        "dev_mae": mae_bm,
        "dev_rmse": rmse_bm,
        "runtime_ms": t_bm,
    })
    print(f"BM  + {pm:14s} | Dev Bad2: {b2_bm:.2f}% Cov: {cov_bm*100:.1f}% Time: {t_bm:.1f}ms")

    # Evaluate SGBM
    b2_sg, b1_sg, cov_sg, mae_sg, rmse_sg, t_sg, per_sc_sg = evaluate_sgbm_config_on_scenes(best_sgbm_cfg, dev_scene_objs, prep_method=pm)
    prep_results.append({
        "algorithm": "StereoSGBM",
        "preprocessing": pm,
        "dev_bad_2_0": b2_sg,
        "dev_bad_1_0": b1_sg,
        "dev_coverage": cov_sg,
        "dev_mae": mae_sg,
        "dev_rmse": rmse_sg,
        "runtime_ms": t_sg,
    })
    print(f"SGBM + {pm:14s} | Dev Bad2: {b2_sg:.2f}% Cov: {cov_sg*100:.1f}% Time: {t_sg:.1f}ms")

df_prep = pd.DataFrame(prep_results)
df_prep.to_csv(OUT_DIR / "preprocessing_comparison.csv", index=False)

# Identify best preprocessing for SGBM and BM
best_bm_prep = df_prep[df_prep["algorithm"] == "StereoBM"].sort_values("dev_bad_2_0").iloc[0]["preprocessing"]
best_sgbm_prep = df_prep[df_prep["algorithm"] == "StereoSGBM"].sort_values("dev_bad_2_0").iloc[0]["preprocessing"]

# --------------------------------------------------------------------------
# 4. VALIDATION ON INDEPENDENT SCENES
# --------------------------------------------------------------------------
print("\n--- STAGE 3: Independent Validation Set Evaluation ---")
val_results = []

# G0: Baselines
base_bm_cfg = StereoBMConfig(numDisparities=80, blockSize=15)
base_sgbm_cfg = StereoSGBMConfig(numDisparities=80, blockSize=5, P1=600, P2=2400)

b2, b1, cov, mae, rmse, t, _ = evaluate_bm_config_on_scenes(base_bm_cfg, val_scene_objs, prep_method="gray")
val_results.append({"model": "G0_BM_Default", "bad_2_0": b2, "coverage": cov, "mae": mae, "runtime_ms": t})

b2, b1, cov, mae, rmse, t, _ = evaluate_sgbm_config_on_scenes(base_sgbm_cfg, val_scene_objs, prep_method="gray")
val_results.append({"model": "G0_SGBM_Default", "bad_2_0": b2, "coverage": cov, "mae": mae, "runtime_ms": t})

# G1: Frozen Optimized
b2, b1, cov, mae, rmse, t, _ = evaluate_bm_config_on_scenes(best_bm_cfg, val_scene_objs, prep_method="gray")
val_results.append({"model": "G1_BM_Optimized", "bad_2_0": b2, "coverage": cov, "mae": mae, "runtime_ms": t})

b2, b1, cov, mae, rmse, t, _ = evaluate_sgbm_config_on_scenes(best_sgbm_cfg, val_scene_objs, prep_method="gray")
val_results.append({"model": "G1_SGBM_Optimized", "bad_2_0": b2, "coverage": cov, "mae": mae, "runtime_ms": t})

# G2: Frozen Optimized + Best Preprocessing
b2, b1, cov, mae, rmse, t, _ = evaluate_bm_config_on_scenes(best_bm_cfg, val_scene_objs, prep_method=best_bm_prep)
val_results.append({"model": f"G2_BM_Opt_{best_bm_prep}", "bad_2_0": b2, "coverage": cov, "mae": mae, "runtime_ms": t})

b2, b1, cov, mae, rmse, t, _ = evaluate_sgbm_config_on_scenes(best_sgbm_cfg, val_scene_objs, prep_method=best_sgbm_prep)
val_results.append({"model": f"G2_SGBM_Opt_{best_sgbm_prep}", "bad_2_0": b2, "coverage": cov, "mae": mae, "runtime_ms": t})

df_val = pd.DataFrame(val_results)
print(df_val.to_string(index=False))
df_val.to_csv(OUT_DIR / "validation_comparison.csv", index=False)

# --------------------------------------------------------------------------
# 5. FREEZE CANDIDATE CONFIGURATIONS
# --------------------------------------------------------------------------
frozen_bm = {
    "G0_default": StereoBMConfig().to_dict(),
    "G1_optimized": best_bm_cfg.to_dict(),
    "G2_preprocessing": best_bm_prep,
    "dev_bad_2_0": best_bm_trial.avg_dev_bad_2_0,
}
frozen_sgbm = {
    "G0_default": StereoSGBMConfig().to_dict(),
    "G1_optimized": best_sgbm_cfg.to_dict(),
    "G2_preprocessing": best_sgbm_prep,
    "dev_bad_2_0": best_sgbm_trial.avg_dev_bad_2_0,
}

with open(BEST_CFG_DIR / "bm_frozen_config.json", "w") as f:
    json.dump(frozen_bm, f, indent=2)

with open(BEST_CFG_DIR / "sgbm_frozen_config.json", "w") as f:
    json.dump(frozen_sgbm, f, indent=2)

# Save full trial history
pd.DataFrame([t.to_dict() for t in bm_trials]).to_csv(OUT_DIR / "bm_optimization_history.csv", index=False)
pd.DataFrame([t.to_dict() for t in sgbm_trials]).to_csv(OUT_DIR / "sgbm_optimization_history.csv", index=False)

print("\nOptimization history and frozen configurations successfully written to results/middlebury/best_configs/!")
