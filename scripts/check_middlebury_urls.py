import urllib.request
import re

url = "https://vision.middlebury.edu/stereo/submit3/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
        for line in html.splitlines():
            if any(term in line.lower() for term in ["zip", "tgz", "tar.gz", "download", ".php", "href"]):
                print(line)
except Exception as e:
    print("Error:", e)
