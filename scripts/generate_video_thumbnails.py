import cv2
from pathlib import Path

OUT_DIR = Path("video/assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Extract frame 150 from Video A
cap_a = cv2.VideoCapture("video/final/middlebury_optimization.mp4")
cap_a.set(cv2.CAP_PROP_POS_FRAMES, 150)
ret_a, frame_a = cap_a.read()
if ret_a:
    cv2.imwrite(str(OUT_DIR / "middlebury_thumbnail.png"), frame_a)
cap_a.release()

# Extract frame 250 from Video B
cap_b = cv2.VideoCapture("video/final/gunung_stereo_demonstration.mp4")
cap_b.set(cv2.CAP_PROP_POS_FRAMES, 250)
ret_b, frame_b = cap_b.read()
if ret_b:
    cv2.imwrite(str(OUT_DIR / "gunung_thumbnail.png"), frame_b)
cap_b.release()

print("Extracted video thumbnails to video/assets/!")
