"""
HotPotQA data loading for qa_kilt.
"""

import bz2
import json
import os
import shutil
import tarfile
from pathlib import Path

import requests
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm

HOTPOT_CORPUS_LINK = "https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"


def download_hotpot_corpus(data_dir: str) -> None:
    response = requests.get(HOTPOT_CORPUS_LINK, stream=True)
    response.raise_for_status()
    destination = f"{data_dir}/hotpotqa_corpus.tar.bz2"
    total_size = int(response.headers.get("content-length", 0))
    with (
        open(destination, "wb") as handle,
        tqdm(total=total_size, unit="B", unit_scale=True, desc="Downloading HotPotQA corpus") as progress,
    ):
        for chunk in response.iter_content(chunk_size=2 * 1024 * 1024):
            handle.write(chunk)
            progress.update(len(chunk))
    print(f"HotPotQA corpus downloaded successfully to {destination}")


def build_hotpotqa_jsonl(data_dir: str, cleanup: bool = True) -> None:
    archive_file = f"{data_dir}/hotpotqa_corpus.tar.bz2"
    if not os.path.exists(archive_file):
        download_hotpot_corpus(data_dir)
    else:
        print(f"HotPotQA corpus archive already exists at {archive_file}. Skipping download.")

    print("Extracting HotPotQA corpus...")
    extract_path = f"{data_dir}/hotpotqa_corpus"
    with tarfile.open(archive_file, "r:bz2") as tar:
        tar.extractall(path=extract_path)

    output_file = f"{data_dir}/hotpotqa_corpus.jsonl"
    print("Creating hotpotqa_corpus.jsonl...")
    count = 0
    compressed_files = sorted(Path(extract_path).glob("**/wiki_*.bz2"))
    with open(output_file, "w") as out_file:
        for compressed_file in tqdm(compressed_files, desc="Processing compressed files"):
            with bz2.open(compressed_file, "rt", encoding="utf-8") as bz2_file:
                for line in bz2_file:
                    original = json.loads(line)
                    new_row = {
                        "id": count,
                        "old_id": original["id"],
                        "title": original["title"],
                        "text": "".join(original["text"]),
                    }
                    count += 1
                    out_file.write(json.dumps(new_row) + "\n")

    print(f"HotPotQA corpus JSONL file created at {output_file}")
    if cleanup:
        print("Cleaning up temporary files...")
        os.remove(archive_file)
        shutil.rmtree(extract_path)


def get_hotpotqa_documents(data_dir: str) -> Dataset:
    try:
        loaded = load_from_disk(data_dir)
        if isinstance(loaded, DatasetDict):
            loaded = loaded["train"]
        dataset = loaded
        print("HotPotQA corpus loaded successfully")
    except (FileNotFoundError, OSError):
        print("HotPotQA corpus not found. Building the corpus...")
        build_hotpotqa_jsonl(data_dir)
        dataset = load_dataset("json", data_files=os.path.join(data_dir, "hotpotqa_corpus.jsonl"), split="train")
        os.remove(os.path.join(data_dir, "hotpotqa_corpus.jsonl"))
        dataset.save_to_disk(data_dir)
    return dataset


def get_hotpotqa_questions(data_dir: str, hotpotqa_documents: Dataset) -> DatasetDict:
    try:
        questions = load_from_disk(data_dir)
        print("HotPotQA questions loaded successfully")
    except (FileNotFoundError, OSError):
        print("HotPotQA questions not found. Preprocessing the questions...")
        questions = load_dataset("hotpot_qa", "fullwiki")
        title_to_id = dict(zip(hotpotqa_documents["title"], hotpotqa_documents["id"]))
        old_columns = questions["train"].column_names
        questions = questions.map(
            lambda row, idx: {
                "id": idx,
                "old_id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "supporting_ids": [title_to_id[title] for title in row["supporting_facts"]["title"]],
                "supporting_sent_ids": row["supporting_facts"]["sent_id"],
            },
            with_indices=True,
            remove_columns=old_columns,
        )
        questions.save_to_disk(data_dir)
    return questions


def get_hotpotqa(data_dir: str) -> tuple[DatasetDict, Dataset]:
    documents_path = os.path.join(data_dir, "documents")
    os.makedirs(documents_path, exist_ok=True)
    documents = get_hotpotqa_documents(documents_path)

    questions_path = os.path.join(data_dir, "questions")
    os.makedirs(questions_path, exist_ok=True)
    questions = get_hotpotqa_questions(questions_path, documents)
    return questions, documents
