import os
import urllib.request
import zipfile
from pathlib import Path

DEST_DIR = Path("data/middlebury")
DEST_DIR.mkdir(parents=True, exist_ok=True)

URLS = [
    ("https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-Q.zip", "MiddEval3-data-Q.zip"),
    ("https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT0-Q.zip", "MiddEval3-GT0-Q.zip"),
    ("https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-GT1-Q.zip", "MiddEval3-GT1-Q.zip"),
]

def download_and_extract(url: str, filename: str):
    zip_path = DEST_DIR / filename
    print(f"Downloading {url} to {zip_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out_file:
        total_len = resp.headers.get("Content-Length")
        total_bytes = int(total_len) if total_len else None
        downloaded = 0
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
    print(f"Downloaded {filename} ({downloaded} bytes). Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DEST_DIR)
    print(f"Extracted {filename} successfully.")

if __name__ == "__main__":
    for url, fn in URLS:
        download_and_extract(url, fn)
    print("\nAll Middlebury evaluation-v3 quarter-resolution assets downloaded and extracted!")
