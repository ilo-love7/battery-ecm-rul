"""Download NASA PCoE Battery Data Set and verify integrity.

Source: https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip
(URL-encoded form; Python urllib verified HTTP 200, Content-Length 209,708,670)
"""
import hashlib
import os
import sys
import time
import urllib.request

URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.%20Battery%20Data%20Set.zip"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEST_DIR = os.path.join(ROOT, "data", "raw", "nasa")
DEST = os.path.join(DEST_DIR, "nasa_battery_dataset.zip")
MANIFEST = os.path.join(DEST_DIR, "nasa_manifest.txt")


def main() -> int:
    os.makedirs(DEST_DIR, exist_ok=True)
    print("Downloading:", URL)
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        print("Content-Length:", total)
        md5 = hashlib.md5()
        downloaded = 0
        start = time.time()
        with open(DEST, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                md5.update(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    rate = downloaded / max(time.time() - start, 1e-6) / 1024 / 1024
                    print(f"\r  {downloaded}/{total} ({pct:.1f}%)  {rate:.1f} MB/s", end="", flush=True)
        print()
    size = os.path.getsize(DEST)
    digest = md5.hexdigest()
    print("Downloaded bytes:", size)
    print("MD5:", digest)
    ok = (size == total) if total else True
    print("SIZE_OK:", ok)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        f.write(f"url={URL}\nbytes={size}\nmd5={digest}\nsize_ok={ok}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
