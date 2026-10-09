#!/usr/bin/env python3
"""Download the 40 official CWRU recordings used here into data/raw/cwru/ (about 300 MB) and verify each one."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tinyml_lab import cwru  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dir", default="data/raw/cwru")
p.add_argument("--base-url", default=os.environ.get("CWRU_BASE_URL", cwru.BASE_URL),
               help="where the .mat files are fetched from (default: the official CWRU site)")
a = p.parse_args()
res = cwru.download_all(a.dir, base_url=a.base_url)
print(f"downloaded {res['ok']}, already present and verified {res['skipped']}, failed {len(res['failed'])}, not attempted {res['not_attempted']}")
if res["failed"] or res["not_attempted"]:
    print("Files that failed or did not match their checksum:", ", ".join(str(n) for n, _ in res["failed"]) or "none")
    if res["not_attempted"]:
        print(f"{res['not_attempted']} more files were not attempted because the site did not answer. Check your internet connection and run this again.")
    print("Run this script again first: it resumes partial files and skips the ones already verified.")
    print("If it keeps failing, download them by hand from https://engineering.case.edu/bearingdatacenter (the 48 kHz drive-end and normal baseline pages),")
    print(f"save them as <number>.mat in {a.dir}, and run this script again to verify them.")
    sys.exit(1)
