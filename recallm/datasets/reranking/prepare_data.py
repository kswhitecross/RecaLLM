"""
One-time data preparation for MSMARCO V2 reranking dataset.

Steps:
  1. Load TREC DL 2021-2023 qrels + queries via ir_datasets
  2. Ensure the V2 passage corpus is extracted (tar → gz → flat text files)
  3. Build BM25 index over the full 138M-passage corpus (one-time, slow)
  4. Run BM25 queries to get per-query unjudged hard negatives
  5. Save all output files

Outputs (in --output_dir):
  - qrels.json:          {qid: {pid: grade, ...}, ...}
  - queries.json:        {qid: query_text, ...}
  - bm25_negatives.json: {qid: [pid, ...], ...}
  - split.json:          {train: [qid, ...], validation: [qid, ...]}
  - stats.json:          dataset statistics
  - bm25_index/          saved bm25s index + doc_ids.json

Usage:
    # Fast first run (extraction + qrels only, skip BM25):
    python -m recallm.datasets.reranking.prepare_data --skip_bm25

    # Full run (extraction + BM25 index + negatives):
    python -m recallm.datasets.reranking.prepare_data
"""

import argparse
import glob
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TREC_DL_DATASETS = {
    "2021": "msmarco-passage-v2/trec-dl-2021/judged",
    "2022": "msmarco-passage-v2/trec-dl-2022/judged",
    "2023": "msmarco-passage-v2/trec-dl-2023",
}

# 2023 qrels are not in ir_datasets; download directly from TREC
TREC_DL_2023_QRELS_URL = "https://trec.nist.gov/data/deep/2023.qrels.pass.withDupes.txt"

V2_CORPUS_ID = "msmarco-passage-v2"

DEFAULT_OUTPUT_DIR = os.environ.get("RECALLM_MSMARCO_DIR")
DEFAULT_IR_DATASETS_HOME = os.environ.get("IR_DATASETS_HOME")


# ---------------------------------------------------------------------------
# Step 1: Qrels and queries
# ---------------------------------------------------------------------------

def _download_trec_qrels(url: str, cache_dir: str) -> dict[str, dict[str, int]]:
    """Download and parse a TREC qrels file (format: qid Q0 docid grade).

    Returns {qid: {docid: grade}}.
    """
    cache_path = Path(cache_dir) / "2023.qrels.pass.withDupes.txt"
    if not cache_path.exists():
        print(f"  Downloading 2023 qrels from {url}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        with open(cache_path, "wb") as f:
            f.write(data)
        print(f"  Saved {len(data):,} bytes to {cache_path}")
    else:
        print(f"  Using cached 2023 qrels from {cache_path}")

    qrels: dict[str, dict[str, int]] = {}
    with open(cache_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, docid, grade = parts
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(grade)
    return qrels


def load_qrels_and_queries(ir_datasets_home: str, output_dir: str) -> tuple[dict, dict, dict]:
    """Load qrels and queries from all TREC DL years.

    For 2021/2022: uses ir_datasets (qrels available natively).
    For 2023: downloads qrels directly from TREC, queries from ir_datasets.

    Returns:
        qrels: {qid: {pid: grade, ...}}
        queries: {qid: query_text}
        query_years: {qid: year_str}
    """
    os.environ["IR_DATASETS_HOME"] = ir_datasets_home
    import ir_datasets

    qrels: dict[str, dict[str, int]] = {}
    queries: dict[str, str] = {}
    query_years: dict[str, str] = {}

    for year, ds_id in TREC_DL_DATASETS.items():
        print(f"  Loading TREC DL {year}: {ds_id}")
        ds = ir_datasets.load(ds_id)

        # Queries
        year_queries = 0
        for q in ds.queries_iter():
            queries[q.query_id] = q.text
            query_years[q.query_id] = year
            year_queries += 1

        # Qrels
        year_qrels = 0
        if ds.has_qrels():
            for qrel in ds.qrels_iter():
                if qrel.query_id not in qrels:
                    qrels[qrel.query_id] = {}
                qrels[qrel.query_id][qrel.doc_id] = qrel.relevance
                year_qrels += 1
        elif year == "2023":
            print("  Qrels not in ir_datasets, downloading from TREC...")
            dl_qrels = _download_trec_qrels(TREC_DL_2023_QRELS_URL, output_dir)
            for qid, judgments in dl_qrels.items():
                qrels[qid] = judgments
                year_qrels += len(judgments)
        else:
            print(f"  WARNING: No qrels available for {year}, skipping")

        print(f"    {year}: {year_queries} queries, {year_qrels:,} qrels")

    # Only keep queries that have qrels
    judged_qids = set(qrels.keys())
    queries = {qid: text for qid, text in queries.items() if qid in judged_qids}
    query_years = {qid: y for qid, y in query_years.items() if qid in judged_qids}

    # Count unique passage IDs
    all_pids = set()
    for judgments in qrels.values():
        all_pids.update(judgments.keys())

    print(f"  Total: {len(queries)} judged queries, {sum(len(v) for v in qrels.values()):,} qrels, {len(all_pids):,} unique passages")
    return qrels, queries, query_years


def build_train_val_split(query_years: dict[str, str]) -> dict[str, list[str]]:
    """Split queries: 2021+2022 → train, 2023 → validation."""
    split = {"train": [], "validation": []}
    for qid, year in sorted(query_years.items()):
        if year in ("2021", "2022"):
            split["train"].append(qid)
        else:
            split["validation"].append(qid)
    print(f"  Split: {len(split['train'])} train, {len(split['validation'])} validation")
    return split


def compute_stats(qrels: dict, queries: dict, query_years: dict) -> dict:
    """Compute dataset statistics."""
    grade_counts = defaultdict(int)
    per_query_stats = []

    for qid, judgments in qrels.items():
        n_relevant = sum(1 for g in judgments.values() if g >= 1)
        n_total = len(judgments)
        per_query_stats.append({"n_relevant": n_relevant, "n_total": n_total})
        for grade in judgments.values():
            grade_counts[grade] += 1

    year_counts = defaultdict(int)
    for y in query_years.values():
        year_counts[y] += 1

    return {
        "n_queries": len(queries),
        "queries_per_year": dict(year_counts),
        "total_qrels": sum(len(v) for v in qrels.values()),
        "grade_distribution": {str(k): v for k, v in sorted(grade_counts.items())},
        "avg_judged_per_query": float(np.mean([s["n_total"] for s in per_query_stats])),
        "avg_relevant_per_query": float(np.mean([s["n_relevant"] for s in per_query_stats])),
        "min_judged_per_query": min(s["n_total"] for s in per_query_stats),
        "max_judged_per_query": max(s["n_total"] for s in per_query_stats),
    }


# ---------------------------------------------------------------------------
# Step 2: Corpus extraction
# ---------------------------------------------------------------------------

def _decompress_one_gz(gz_path: str, out_path: str) -> tuple[str, float, float]:
    """Decompress a single gz file using gunzip. Returns (basename, elapsed_seconds, size_mb)."""
    basename = os.path.basename(out_path)
    t0 = time.time()
    with open(out_path, "wb") as fout:
        subprocess.run(
            ["gunzip", "-c", gz_path],
            stdout=fout,
            stderr=subprocess.PIPE,
            check=True,
        )
    elapsed = time.time() - t0
    size_mb = os.path.getsize(out_path) / 1e6
    return basename, elapsed, size_mb


def _create_pos_file(text_path: str) -> tuple[str, int, float]:
    """Create a .pos index file for one extracted text file.

    ir_datasets' docs_iter() needs these: a uint32 array of byte offsets
    marking the start of each JSON line in the decompressed file.

    Uses chunked binary reads (64MB) instead of readline() for speed.

    Returns (basename, n_lines, elapsed_seconds).
    """
    basename = os.path.basename(text_path)
    pos_path = text_path + ".pos"
    t0 = time.time()

    CHUNK_SIZE = 64 * 1024 * 1024  # 64MB
    positions = [0]  # first line always starts at byte 0
    file_offset = 0

    with open(text_path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            # Find all newlines in this chunk
            search_start = 0
            while True:
                idx = chunk.find(b"\n", search_start)
                if idx == -1:
                    break
                # Next line starts at byte after the newline
                positions.append(file_offset + idx + 1)
                search_start = idx + 1
            file_offset += len(chunk)

    # Remove trailing entry if it points past EOF (file ends with \n)
    file_size = os.path.getsize(text_path)
    if positions and positions[-1] >= file_size:
        positions.pop()

    pos_array = np.array(positions, dtype="<u4")
    with open(pos_path, "wb") as f:
        f.write(pos_array.tobytes())
    elapsed = time.time() - t0
    return basename, len(positions), elapsed


def _ensure_pos_files(extracted_dir: str):
    """Create .pos index files for all extracted text files if missing.

    These are needed by ir_datasets' docs_iter() for sequential iteration.
    """
    text_files = sorted(
        os.path.join(extracted_dir, f)
        for f in os.listdir(extracted_dir)
        if f.startswith("msmarco_passage_") and not f.endswith(".pos")
    )

    # Check which ones need pos files
    missing = [f for f in text_files if not os.path.exists(f + ".pos")]
    if not missing:
        print(f"  All {len(text_files)} .pos files already exist")
        return

    print(f"  Creating {len(missing)} .pos index files in parallel ({len(missing)} workers)...")
    t0 = time.time()
    completed = 0
    total_lines = 0
    with ThreadPoolExecutor(max_workers=len(missing)) as pool:
        futures = {pool.submit(_create_pos_file, f): f for f in missing}
        for future in as_completed(futures):
            basename, n_lines, elapsed = future.result()
            completed += 1
            total_lines += n_lines
            print(f"    [{completed}/{len(missing)}] {basename}.pos: {n_lines:,} lines in {elapsed:.1f}s")

    print(f"  Created {len(missing)} .pos files ({total_lines:,} total lines) in {time.time() - t0:.1f}s")


def ensure_corpus_extracted(ir_datasets_home: str, tmp_dir: str = "/tmp", force: bool = False):
    """Ensure the V2 passage corpus is extracted for ir_datasets.

    ir_datasets needs:
      1. Decompressed flat text files in .extracted/ (for docstore get_many)
      2. .pos byte-offset index files (for docs_iter sequential iteration)
      3. A _built sentinel file

    We extract with `tar xf` + parallel `gunzip`, then create .pos files.
    """
    corpus_dir = os.path.join(ir_datasets_home, "msmarco-passage-v2")
    tar_path = os.path.join(corpus_dir, "msmarco_v2_passage.tar")
    extracted_dir = os.path.join(corpus_dir, "msmarco_v2_passage.tar.extracted")
    built_sentinel = os.path.join(extracted_dir, "_built")

    if os.path.exists(built_sentinel) and not force:
        n_text = len([f for f in os.listdir(extracted_dir) if f.startswith("msmarco_passage_") and not f.endswith(".pos")])
        n_pos = len([f for f in os.listdir(extracted_dir) if f.endswith(".pos")])
        print(f"  Corpus already extracted ({n_text} text files, {n_pos} pos files)")

        # Ensure pos files exist even if extraction was done without them
        _ensure_pos_files(extracted_dir)
        return

    if not os.path.exists(tar_path):
        raise FileNotFoundError(
            f"Corpus tar not found at {tar_path}. "
            "Run ir_datasets.load('msmarco-passage-v2') first to download it."
        )

    tar_size_gb = os.path.getsize(tar_path) / 1e9
    print(f"  Tar file: {tar_path} ({tar_size_gb:.1f} GB)")
    print(f"  Extraction target: {extracted_dir}")
    os.makedirs(extracted_dir, exist_ok=True)

    # Step 2a: Extract tar to temp dir (gets 70 gz files)
    print(f"  Extracting tar to temp dir ({tmp_dir})...")
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=tmp_dir, prefix="msmarco_extract_") as tmpdir:
        subprocess.run(
            ["tar", "xf", tar_path, "-C", tmpdir],
            check=True,
        )
        tar_elapsed = time.time() - t0

        # Find extracted gz files
        gz_dir = os.path.join(tmpdir, "msmarco_v2_passage")
        gz_files = sorted(glob.glob(os.path.join(gz_dir, "msmarco_passage_*.gz")))
        total_gz_mb = sum(os.path.getsize(f) for f in gz_files) / 1e6
        print(f"  Extracted {len(gz_files)} gz files ({total_gz_mb:.0f} MB) in {tar_elapsed:.1f}s")

        # Step 2b: Decompress gz files in parallel
        n_workers = len(gz_files)
        print(f"  Decompressing {len(gz_files)} gz files in parallel ({n_workers} workers)...")
        t0 = time.time()

        tasks = []
        for gz_path in gz_files:
            basename = os.path.basename(gz_path).replace(".gz", "")
            out_path = os.path.join(extracted_dir, basename)
            tasks.append((gz_path, out_path))

        completed = 0
        total_size_mb = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_decompress_one_gz, gz_path, out_path): basename
                for gz_path, out_path in tasks
            }
            for future in as_completed(futures):
                basename, elapsed, size_mb = future.result()
                completed += 1
                total_size_mb += size_mb
                print(f"    [{completed}/{len(gz_files)}] {basename}: {size_mb:.0f} MB in {elapsed:.1f}s")

        decompress_elapsed = time.time() - t0
        print(f"  Decompressed {total_size_mb / 1e3:.1f} GB total in {decompress_elapsed:.1f}s")

    # Step 2c: Create .pos index files
    _ensure_pos_files(extracted_dir)

    # Touch sentinel
    Path(built_sentinel).touch()
    print(f"  Extraction complete. Sentinel: {built_sentinel}")


# ---------------------------------------------------------------------------
# Step 3: BM25 index
# ---------------------------------------------------------------------------

def build_bm25_index(ir_datasets_home: str, index_dir: str) -> "bm25s.BM25":
    """Build a BM25 index over the full V2 passage corpus using bm25s.

    Requires the corpus to be extracted first (Step 2).
    """
    import bm25s
    import Stemmer

    index_path = Path(index_dir)

    # Check if index already exists
    if index_path.exists() and (index_path / "params.index.json").exists():
        print(f"  BM25 index already exists at {index_path}")
        print(f"  Loading index (mmap=True)...")
        t0 = time.time()
        retriever = bm25s.BM25.load(str(index_path), mmap=True)
        print(f"  Loaded in {time.time() - t0:.1f}s")
        return retriever

    os.environ["IR_DATASETS_HOME"] = ir_datasets_home
    import ir_datasets

    print(f"  Loading corpus '{V2_CORPUS_ID}'...")
    ds = ir_datasets.load(V2_CORPUS_ID)
    total_docs = ds.docs_count()
    print(f"  Total passages: {total_docs:,}")

    stemmer = Stemmer.Stemmer("english")

    # Load all doc_ids and texts
    print(f"  Loading corpus text into memory (this requires ~40 GB RAM)...")
    t0 = time.time()
    doc_ids = []
    corpus = []
    for i, doc in enumerate(ds.docs_iter()):
        doc_ids.append(doc.doc_id)
        corpus.append(doc.text)
        if (i + 1) % 5_000_000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total_docs - i - 1) / rate
            print(f"    {i + 1:>12,} / {total_docs:,} ({(i + 1) / total_docs * 100:.1f}%) "
                  f"  [{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

    load_elapsed = time.time() - t0
    print(f"  Loaded {len(corpus):,} passages in {load_elapsed:.1f}s")

    # Tokenize
    print(f"  Tokenizing corpus with PyStemmer English...")
    t0 = time.time()
    corpus_tok = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer, show_progress=True)
    print(f"  Tokenized in {time.time() - t0:.1f}s")

    del corpus  # free ~40 GB

    # Build index
    print(f"  Building BM25 index...")
    t0 = time.time()
    retriever = bm25s.BM25()
    retriever.index(corpus_tok)
    print(f"  Indexed in {time.time() - t0:.1f}s")

    del corpus_tok

    # Save
    print(f"  Saving index to {index_path}...")
    index_path.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    retriever.save(str(index_path))
    print(f"  Index saved in {time.time() - t0:.1f}s")

    # Save doc_id mapping
    doc_ids_path = index_path / "doc_ids.json"
    print(f"  Saving {len(doc_ids):,} doc_ids...")
    t0 = time.time()
    with open(doc_ids_path, "w") as f:
        json.dump(doc_ids, f)
    print(f"  doc_ids saved in {time.time() - t0:.1f}s")

    return retriever


# ---------------------------------------------------------------------------
# Step 4: BM25 queries
# ---------------------------------------------------------------------------

def run_bm25_queries(
    retriever,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    index_dir: str,
    k: int = 2000,
) -> dict[str, list[str]]:
    """Run BM25 queries and return per-query unjudged negative PIDs."""
    import bm25s
    import Stemmer

    # Load doc_id mapping
    doc_ids_path = Path(index_dir) / "doc_ids.json"
    print(f"  Loading doc_ids from {doc_ids_path}...")
    t0 = time.time()
    with open(doc_ids_path, "r") as f:
        doc_ids = json.load(f)
    print(f"  Loaded {len(doc_ids):,} doc_ids in {time.time() - t0:.1f}s")

    stemmer = Stemmer.Stemmer("english")

    # Batch all queries
    qid_list = sorted(queries.keys())
    query_texts = [queries[qid] for qid in qid_list]

    print(f"  Tokenizing {len(query_texts)} queries...")
    query_toks = bm25s.tokenize(query_texts, stopwords="en", stemmer=stemmer, show_progress=False)

    print(f"  Retrieving top-{k} results per query...")
    t0 = time.time()
    results, scores = retriever.retrieve(query_toks, k=k)
    print(f"  Retrieved in {time.time() - t0:.1f}s")

    # Build per-query unjudged BM25 negatives
    print(f"  Filtering to unjudged negatives...")
    bm25_negatives: dict[str, list[str]] = {}
    for i, qid in enumerate(qid_list):
        judged_pids = set(qrels.get(qid, {}).keys())
        neg_pids = []
        for j in range(results.shape[1]):
            idx = int(results[i, j])
            pid = doc_ids[idx]
            if pid not in judged_pids:
                neg_pids.append(pid)
        bm25_negatives[qid] = neg_pids
        print(f"    Query {qid}: {len(neg_pids)} unjudged BM25 negatives (from {k} retrieved)")

    avg_negs = np.mean([len(v) for v in bm25_negatives.values()])
    print(f"  Average unjudged BM25 negatives per query: {avg_negs:.1f}")
    return bm25_negatives


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare MSMARCO V2 reranking data (extraction, BM25 index, negatives)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory for output index files",
    )
    parser.add_argument(
        "--ir_datasets_home", type=str, default=DEFAULT_IR_DATASETS_HOME,
        help="IR_DATASETS_HOME directory for corpus storage",
    )
    parser.add_argument(
        "--bm25_k", type=int, default=2000,
        help="Number of BM25 results to retrieve per query (default: 2000)",
    )
    parser.add_argument(
        "--skip_bm25", action="store_true",
        help="Skip BM25 index building and query retrieval",
    )
    parser.add_argument(
        "--force_extract", action="store_true",
        help="Re-run corpus extraction even if already done",
    )
    parser.add_argument(
        "--tmp_dir", type=str, default="/tmp",
        help="Temp directory for tar extraction (default: /tmp, use local NVMe for speed)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_t0 = time.time()

    # ---- Step 1: Load qrels and queries ----
    print("=" * 70)
    print("STEP 1: Loading qrels and queries")
    print("=" * 70)
    t0 = time.time()
    qrels, queries, query_years = load_qrels_and_queries(args.ir_datasets_home, args.output_dir)
    split = build_train_val_split(query_years)
    stats = compute_stats(qrels, queries, query_years)

    print(f"\n  Dataset stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    # Save index files
    print(f"\n  Saving index files to {output_dir}")
    with open(output_dir / "qrels.json", "w") as f:
        json.dump(qrels, f)
    with open(output_dir / "queries.json", "w") as f:
        json.dump(queries, f)
    with open(output_dir / "split.json", "w") as f:
        json.dump(split, f)
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: qrels.json, queries.json, split.json, stats.json")
    print(f"  Step 1 completed in {time.time() - t0:.1f}s")

    # ---- Step 2: Ensure corpus extraction ----
    print("\n" + "=" * 70)
    print("STEP 2: Ensuring corpus is extracted")
    print("=" * 70)
    t0 = time.time()
    ensure_corpus_extracted(args.ir_datasets_home, tmp_dir=args.tmp_dir, force=args.force_extract)
    print(f"  Step 2 completed in {time.time() - t0:.1f}s")

    # ---- Step 3 & 4: BM25 index + queries ----
    if args.skip_bm25:
        print("\n" + "=" * 70)
        print("STEP 3 & 4: SKIPPED (--skip_bm25)")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("STEP 3: Building BM25 index")
        print("=" * 70)
        t0 = time.time()
        index_dir = str(output_dir / "bm25_index")
        retriever = build_bm25_index(args.ir_datasets_home, index_dir)
        print(f"  Step 3 completed in {time.time() - t0:.1f}s")

        print("\n" + "=" * 70)
        print("STEP 4: Running BM25 queries")
        print("=" * 70)
        t0 = time.time()
        bm25_negatives = run_bm25_queries(
            retriever, queries, qrels, index_dir, k=args.bm25_k
        )

        print(f"  Saving bm25_negatives.json...")
        with open(output_dir / "bm25_negatives.json", "w") as f:
            json.dump(bm25_negatives, f)
        print(f"  Saved bm25_negatives.json")

        # Update stats
        stats["avg_bm25_negatives_per_query"] = float(
            np.mean([len(v) for v in bm25_negatives.values()])
        )
        with open(output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Step 4 completed in {time.time() - t0:.1f}s")

    # ---- Done ----
    total_elapsed = time.time() - total_t0
    print("\n" + "=" * 70)
    print(f"DONE in {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"Output directory: {output_dir}")
    print(f"Files: {', '.join(sorted(os.listdir(output_dir)))}")
    print("=" * 70)


if __name__ == "__main__":
    main()
