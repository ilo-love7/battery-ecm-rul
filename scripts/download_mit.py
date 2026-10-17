"""Download MIT (Severson 2019) dataset from HuggingFace mirror (requests + resume + retry).

Uses requests (urllib3) for robust redirect/connection handling; resumes partial
files via HTTP Range and retries with backoff on connection resets.
"""
import hashlib
import os
import sys
import time

import requests

BASE = "https://huggingface.co/datasets/bsebench-org/severson-2019-raw/resolve/main/"
DEST_DIR = r"D:\新建文件夹\battery-data\mit"
FILES = [
    ("2017-05-12_batchdata_updated_struct_errorcorrect.mat", 3025320241),
    ("2017-06-30_batchdata_updated_struct_errorcorrect.mat", 2007331155),
    ("2018-04-12_batchdata_updated_struct_errorcorrect.mat", 3236690412),
]
MAX_ATTEMPTS = 30
CHUNK = 4 * 1024 * 1024


def file_md5(path):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_one(session, fname, expect_size):
    url = BASE + fname
    dest = os.path.join(DEST_DIR, fname)
    existing = os.path.getsize(dest) if os.path.exists(dest) else 0
    if existing == expect_size:
        print(f"[complete] {fname}", flush=True)
        return fname, existing, file_md5(dest), True
    if existing > expect_size:
        os.remove(dest)
        existing = 0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            if existing > 0:
                headers["Range"] = f"bytes={existing}-"
            resp = session.get(url, headers=headers, stream=True, timeout=(30, 300))
            status = resp.status_code
            resumed = existing > 0 and status == 206
            mode = "ab" if resumed else "wb"
            md5 = hashlib.md5()
            if resumed:
                with open(dest, "rb") as f:
                    for chunk in iter(lambda: f.read(CHUNK), b""):
                        md5.update(chunk)
            downloaded = existing if resumed else 0
            clen = int(resp.headers.get("Content-Length", 0) or 0)
            total = clen + (downloaded if resumed else 0)
            start = time.time()
            with open(dest, mode) as f:
                for chunk in resp.iter_content(chunk_size=CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    md5.update(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        rate = downloaded / max(time.time() - start, 1e-6) / 1024 / 1024
                        print(
                            f"\r  {fname} {downloaded/1e9:.2f}/{total/1e9:.2f} GB "
                            f"({pct:.1f}%) {rate:.1f} MB/s  [resume={resumed}]",
                            end="", flush=True,
                        )
            resp.close()
            print(flush=True)
            size = os.path.getsize(dest)
            if size == expect_size:
                return fname, size, md5.hexdigest(), True
            existing = size
            print(f"  [attempt {attempt}] incomplete ({size}/{expect_size}), resume...", flush=True)
        except Exception as e:
            existing = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"\n  [attempt {attempt}] {type(e).__name__}: {e}", flush=True)
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(2 ** attempt, 20))
            else:
                return fname, existing, None, False
    return fname, existing, None, False


def main() -> int:
    os.makedirs(DEST_DIR, exist_ok=True)
    session = requests.Session()
    lines = []
    for fname, size in FILES:
        lines.append(download_one(session, fname, size))
    with open(os.path.join(DEST_DIR, "mit_manifest.txt"), "w", encoding="utf-8") as f:
        f.write("file\tbytes\tmd5\tsize_ok\n")
        for fname, sz, digest, ok in lines:
            f.write(f"{fname}\t{sz}\t{digest}\t{ok}\n")
    print("manifest ->", os.path.join(DEST_DIR, "mit_manifest.txt"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
