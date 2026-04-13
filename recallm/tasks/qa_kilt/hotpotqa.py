"""
HotPotQA with KILT-window chunk mode and original paragraph mode.
"""

from __future__ import annotations

import os

from recallm.tasks.qa_kilt.load_hotpotqa import get_hotpotqa
from recallm.tasks.qa_kilt.multihop_base import BaseKILTMultiHopDataset


class HotPotQALayer1Dataset(BaseKILTMultiHopDataset):
    DATASET_TYPE = "hotpotqa"
    INDEX_NAME = "HotPotQAKiltParagraphs"

    def _load_source_data(self, split: str) -> None:
        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "data",
            "hotpotqa",
        )
        os.makedirs(data_dir, exist_ok=True)
        self._dense_source_paths = {"hotpotqa_data_dir": os.path.abspath(data_dir)}
        print(f"  {self.DATASET_TYPE}: Loading source data")
        questions, documents = get_hotpotqa(data_dir)
        self._questions = questions[split]
        self._documents = documents

    def _question_pos_ids(self, question_row: dict) -> list[int]:
        return list(set(question_row["supporting_ids"]))
