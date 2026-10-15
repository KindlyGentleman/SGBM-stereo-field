"""Audit all four Gunung terrains: image pairs, resolution, epipolar geometry, and challenges."""
from pathlib import Path
import cv2
import numpy as np

calib_path = r"D:\02_Riset\Research Paper\source_code\calibration\calib_params\stereo_params_advanced.npz"
data = np.load(calib_path)
mtx_l, dist_l = data["mtx_left"], data["dist_left"]
mtx_r, dist_r = data["mtx_right"], data["dist_right"]
R, T = data["R"], data["T"]
img_size = (1280, 720)

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T, alpha=0)
map_l1, map_l2 = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
map_r1, map_r2 = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)

root = Path(r"D:\02_Riset\Research Paper\source_code\comparison\gunung")
terrains = ["puncakTinggi", "puncakWarung", "tanggaBiru", "tanggaKuning"]

print("=" * 85)
print("AUDIT OF ALL FOUR GUNUNG TERRAIN DATASET FOLDERS")
print("=" * 85)

header = f"{'Terrain':<15} | {'Pairs':<5} | {'Resolution':<10} | {'Raw dy Med':<10} | {'Rect dy Med':<11} | {'Rect dy P90':<11} | {'Inliers':<7}"
print(header)
print("-" * 85)

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

results = []

for t in terrains:
    st_dir = root / t / "stereo"
    img_l = cv2.imread(str(st_dir / "left_0000.png"))
    img_r = cv2.imread(str(st_dir / "right_0000.png"))
    
    h, w, c = img_l.shape
    
    # Raw ORB matches
    kp1, des1 = orb.detectAndCompute(img_l, None)
    kp2, des2 = orb.detectAndCompute(img_r, None)
    matches = bf.match(des1, des2)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    if len(pts1) >= 8:
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0)
        inliers = mask.ravel() == 1
        raw_dys = np.abs(pts1[inliers, 1] - pts2[inliers, 1])
        raw_med = float(np.median(raw_dys))
        raw_p90 = float(np.percentile(raw_dys, 90))
    else:
        raw_med, raw_p90 = -1.0, -1.0

    # Rectified
    rect_l = cv2.remap(img_l, map_l1, map_l2, cv2.INTER_LINEAR)
    rect_r = cv2.remap(img_r, map_r1, map_r2, cv2.INTER_LINEAR)
    kp1_r, des1_r = orb.detectAndCompute(rect_l, None)
    kp2_r, des2_r = orb.detectAndCompute(rect_r, None)
    matches_r = bf.match(des1_r, des2_r)
    pts1_r = np.float32([kp1_r[m.queryIdx].pt for m in matches_r])
    pts2_r = np.float32([kp2_r[m.trainIdx].pt for m in matches_r])
    
    if len(pts1_r) >= 8:
        F_r, mask_r = cv2.findFundamentalMat(pts1_r, pts2_r, cv2.FM_RANSAC, 3.0)
        inliers_r = mask_r.ravel() == 1
        rect_dys = np.abs(pts1_r[inliers_r, 1] - pts2_r[inliers_r, 1])
        rect_med = float(np.median(rect_dys))
        rect_p90 = float(np.percentile(rect_dys, 90))
        inl_count = int(np.sum(inliers_r))
    else:
        rect_med, rect_p90, inl_count = -1.0, -1.0, 0

    print(f"{t:<15} | 1     | {w}x{h:<5} | {raw_med:<10.2f} | {rect_med:<11.2f} | {rect_p90:<11.2f} | {inl_count:<7}")
    
    # Pixel saturation & dynamic range
    gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
    sat_pct = float(np.mean(gray_l >= 250) * 100)
    results.append({
        "terrain": t,
        "width": w,
        "height": h,
        "raw_med": raw_med,
        "raw_p90": raw_p90,
        "rect_med": rect_med,
        "rect_p90": rect_p90,
        "inliers": inl_count,
        "saturation_pct": sat_pct
    })

print("=" * 85)
