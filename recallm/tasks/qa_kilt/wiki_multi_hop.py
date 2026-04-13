"""
2WikiMultihopQA with KILT-window chunk mode and original paragraph mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm

from recallm.tasks.qa_kilt.multihop_base import BaseKILTMultiHopDataset

TWOWIKI_HF_DATASET_NAME = "framolfese/2WikiMultihopQA"


def _sentences_to_text(sentences: list[str]) -> str:
    cleaned = [sentence for sentence in sentences if isinstance(sentence, str) and sentence]
    return " ".join(cleaned)


@dataclass
class _DocumentsMap:
    current_doc_id: int = 0
    doc_ids_map: dict = field(default_factory=dict)
    documents: list[dict] = field(default_factory=list)

    def get_id_and_add_doc(self, title: str, text: str) -> int:
        if title in self.doc_ids_map:
            return self.doc_ids_map[title]
        doc_id = self.current_doc_id
        self.doc_ids_map[title] = doc_id
        self.documents.append({"id": doc_id, "title": title, "text": text})
        self.current_doc_id += 1
        return doc_id

    def to_dataset(self) -> HFDataset:
        return HFDataset.from_list(self.documents)


def _download_and_process_two_wiki() -> tuple[DatasetDict, HFDataset]:
    train_hf = load_dataset(TWOWIKI_HF_DATASET_NAME, split="train")
    val_hf = load_dataset(TWOWIKI_HF_DATASET_NAME, split="validation")
    documents_map = _DocumentsMap()

    current_id = 0
    train_questions = []
    val_questions = []
    for question_list, split_dataset, desc in [
        (train_questions, train_hf, "Processing 2Wiki train"),
        (val_questions, val_hf, "Processing 2Wiki validation"),
    ]:
        for example in tqdm(split_dataset, desc=desc):
            supporting_titles = set(example.get("supporting_facts", {}).get("title", []))
            pos_doc_ids = []
            for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
                doc_id = documents_map.get_id_and_add_doc(title=title, text=_sentences_to_text(sentences))
                if title in supporting_titles and doc_id not in pos_doc_ids:
                    pos_doc_ids.append(doc_id)
            question_list.append({
                "id": current_id,
                "question": example["question"],
                "question_id": example["id"],
                "answer": example["answer"],
                "pos_doc_ids": pos_doc_ids,
            })
            current_id += 1
    return DatasetDict(
        {
            "train": HFDataset.from_list(train_questions),
            "validation": HFDataset.from_list(val_questions),
        }
    ), documents_map.to_dataset()


def _get_two_wiki_datasets(data_dir: str) -> tuple[DatasetDict, HFDataset]:
    documents_path = os.path.join(data_dir, "documents")
    questions_path = os.path.join(data_dir, "questions")
    try:
        questions = load_from_disk(questions_path)
        documents = load_from_disk(documents_path)
        print("2Wiki datasets loaded from disk")
    except (FileNotFoundError, OSError):
        print("2Wiki datasets not found on disk. Downloading and processing...")
        questions, documents = _download_and_process_two_wiki()
        questions.save_to_disk(questions_path)
        documents.save_to_disk(documents_path)
    return questions, documents


class TwoWikiMultihopQALayer1Dataset(BaseKILTMultiHopDataset):
    DATASET_TYPE = "2wikimultihopqa"
    INDEX_NAME = "TwoWikiKiltParagraphs"

    def _load_source_data(self, split: str) -> None:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "data",
            "2wikimultihopqa",
        )
        os.makedirs(data_dir, exist_ok=True)
        self._dense_source_paths = {"twowiki_data_dir": os.path.abspath(data_dir)}
        print(f"  {self.DATASET_TYPE}: Loading source data")
        questions, documents = _get_two_wiki_datasets(data_dir)
        self._questions = questions[split]
        self._documents = documents
