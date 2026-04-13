"""
Download and extract the QAMPARI dataset from the official S3 host.

Downloads qampari_with_contexts.zip from the QAMPARI project page
(https://samsam3232.github.io/qampari/) and extracts it.

Usage:
    python -m recallm.tasks.citation_qa.download_qampari \
        --output_dir RecaLLM/data/qampari_raw

    # If you already have the zip downloaded:
    python -m recallm.tasks.citation_qa.download_qampari \
        --zip_path RecaLLM/data/qampari_raw/qampari_with_contexts.zip

The outer zip contains nested zips in qampari_preds/:
    qampari_fid_bm25.zip  → qampari_v2/qampari_fid_format/ (not used)
    qampari_fid_dpr.zip   → qampari_v2/qampari_dpr_retriever/ (what we need)
    qampari_rag.zip       → rag_data_for_paper/ (not used)

Output structure after extraction:
    {output_dir}/qampari_v2/qampari_dpr_retriever/
        full_train_data.jsonl.gz
        full_dev_data.jsonl.gz
        full_test_data.jsonl.gz
"""

import argparse
import os
import subprocess
import sys
import zipfile

_S3_URL = "https://aggreg-qa.s3.amazonaws.com/qampari_with_contexts.zip"
_EXPECTED_SIZE = 9191367104  # bytes (verified 2026-03-04)
_DEFAULT_OUTPUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "qampari_raw"
)


def download_qampari(output_dir: str, zip_path: str | None = None) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Check if already extracted
    expected_file = os.path.join(
        output_dir, "qampari_v2", "qampari_dpr_retriever", "full_train_data.jsonl.gz"
    )
    if os.path.isfile(expected_file):
        print(f"QAMPARI data already extracted at {expected_file}, skipping.")
        return

    # Determine zip path
    if zip_path is None:
        zip_path = os.path.join(output_dir, "qampari_with_contexts.zip")

    # Download if not present or truncated
    if os.path.isfile(zip_path):
        actual_size = os.path.getsize(zip_path)
        if actual_size < _EXPECTED_SIZE * 0.95:
            print(
                f"Existing zip is truncated ({actual_size:,} bytes, "
                f"expected ~{_EXPECTED_SIZE:,}). Re-downloading..."
            )
            os.remove(zip_path)

    if not os.path.isfile(zip_path):
        print(f"Downloading QAMPARI data from {_S3_URL} (~9.2 GB)...")
        try:
            subprocess.run(
                ["wget", "-c", "-O", zip_path, _S3_URL],
                check=True,
            )
        except FileNotFoundError:
            # wget not available, try curl
            subprocess.run(
                ["curl", "-L", "-o", zip_path, _S3_URL],
                check=True,
            )
        print(f"  Downloaded to {zip_path}")
    else:
        print(f"Zip already exists at {zip_path} ({os.path.getsize(zip_path):,} bytes)")

    # Extract outer zip → qampari_preds/ with nested zips inside
    print(f"Extracting outer zip to {output_dir}...")
    _extract_zip(zip_path, output_dir)

    # The DPR retriever data is inside a nested zip:
    # qampari_preds/qampari_fid_dpr.zip → qampari_v2/qampari_dpr_retriever/
    inner_zip = os.path.join(output_dir, "qampari_preds", "qampari_fid_dpr.zip")
    if not os.path.isfile(inner_zip):
        raise FileNotFoundError(
            f"Inner zip not found at {inner_zip}. "
            "Outer zip may have a different structure than expected."
        )
    print(f"Extracting inner zip {inner_zip}...")
    _extract_zip(inner_zip, output_dir)

    # Verify expected structure
    for split in ("train", "dev", "test"):
        jsonl_path = os.path.join(
            output_dir, "qampari_v2", "qampari_dpr_retriever",
            f"full_{split}_data.jsonl.gz"
        )
        if not os.path.isfile(jsonl_path):
            raise FileNotFoundError(
                f"Expected file not found after extraction: {jsonl_path}"
            )
    print("Verification passed: all expected JSONL.gz files found.")


def _extract_zip(zip_path: str, output_dir: str) -> None:
    """Extract a zip file, falling back to 7z if Python zipfile fails."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        print(f"  Extracted {os.path.basename(zip_path)} with Python zipfile.")
    except (zipfile.BadZipFile, Exception) as e:
        print(f"  Python zipfile failed ({e}), trying 7z...")
        try:
            subprocess.run(
                ["7z", "x", zip_path, f"-o{output_dir}", "-y"],
                check=True,
            )
            print(f"  Extracted {os.path.basename(zip_path)} with 7z.")
        except FileNotFoundError:
            print("ERROR: Both Python zipfile and 7z failed.")
            print("Install 7z: sudo apt install p7zip-full")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download QAMPARI dataset.")
    parser.add_argument(
        "--output_dir", type=str, default=_DEFAULT_OUTPUT,
        help=f"Directory to save/extract data. Default: {_DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--zip_path", type=str, default=None,
        help="Path to existing qampari_with_contexts.zip (skips download).",
    )
    args = parser.parse_args()
    download_qampari(args.output_dir, args.zip_path)
