"""
Pre-compute the DA3 depth cache used by run_vo --da3-cache DIR (python/loop_server.py --depth-cache): one fp16 depth map
per image, keyed by the MD5 of the decoded BGR image, so later runs are deterministic and need no GPU.

usage: python3 da3_cache.py --cache DIR 'seq/image_2/%06d.png' [more patterns ...]
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from loop_server import Da3Depth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("patterns", nargs="+")
    o = ap.parse_args()
    d = Da3Depth("cuda", o.cache)
    for pat in o.patterns:
        t0, n, new = time.time(), 0, 0
        while True:
            img = cv2.imread(pat % n)
            if img is None:
                break
            f = d.cache / f"{hashlib.md5(img.tobytes()).hexdigest()}_{d.RES}.npy"
            if not f.exists():
                d.depth_map(img)
                new += 1
            n += 1
        print(f"{pat}: {n} images, {new} new, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
