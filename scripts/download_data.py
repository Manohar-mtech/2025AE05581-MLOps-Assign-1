"""Download the UCI Heart Disease dataset and extract the Cleveland file.

Usage: python scripts/download_data.py
Manual alternative: download https://archive.ics.uci.edu/dataset/45/heart+disease,
unzip it and copy `processed.cleveland.data` into data/raw/.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
TARGET_FILE = "processed.cleveland.data"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / TARGET_FILE
    if out.exists():
        print(f"{out} already exists - nothing to do.")
        return 0
    print(f"Downloading {URL} ...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as r:
            payload = r.read()
    except Exception as exc:  # network / HTTP errors
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        out.write_bytes(zf.read(TARGET_FILE))
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
