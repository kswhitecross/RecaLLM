"""
TriviaQA KILT-backed chunk dataset.
"""

from __future__ import annotations

from datasets import Dataset, load_dataset

from recallm.tasks.qa_kilt.single_hop import BaseKILTSingleHopDataset
from recallm.tasks.qa_kilt.utils import collect_answers_and_provenance


def _pair_kilt_and_trivia_rows(kilt_rows: list[dict], trivia_rows: list[dict]) -> list[dict]:
    trivia_by_id = {}
    duplicate_ids = set()
    for row in trivia_rows:
        question_id = row.get("question_id")
        if not question_id:
            continue
        if question_id in trivia_by_id:
            duplicate_ids.add(question_id)
            continue
        trivia_by_id[question_id] = row
    if duplicate_ids:
        raise RuntimeError(f"TriviaQA contains duplicate question_ids, e.g. {sorted(duplicate_ids)[:5]}")

    paired_rows = []
    missing_question_rows = 0
    missing_id_rows = 0
    for kilt_row in kilt_rows:
        question_id = kilt_row.get("id")
        if not question_id:
            missing_id_rows += 1
            continue
        trivia_row = trivia_by_id.get(question_id)
        if trivia_row is None:
            missing_question_rows += 1
            continue
        paired_rows.append({
            "id": question_id,
            "question": trivia_row.get("question") or trivia_row.get("question_text") or kilt_row.get("input", ""),
            "trivia_answer": trivia_row.get("answer"),
            "output": kilt_row.get("output", []),
        })

    if not paired_rows:
        raise RuntimeError("TriviaQA id join produced zero paired rows")
    print(
        f"  triviaqa: Paired {len(paired_rows):,}/{len(kilt_rows):,} KILT rows with original TriviaQA "
        f"(missing_id={missing_id_rows:,}, missing_trivia_match={missing_question_rows:,})"
    )
    return paired_rows


class TriviaQALayer1Dataset(BaseKILTSingleHopDataset):
    DATASET_TYPE = "triviaqa"
    TASK_NAME = "triviaqa_support_only"

    def _load_trivia_questions(self, split: str):
        try:
            return load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split=split)
        except Exception:
            return load_dataset("trivia_qa", "unfiltered.nocontext", split=split)

    def _load_examples(self, split: str):
        print(f"  {self.DATASET_TYPE}: Loading KILT support split={split}")
        kilt_split = load_dataset("facebook/kilt_tasks", self.TASK_NAME, split=split)
        print(f"  {self.DATASET_TYPE}: Loading original TriviaQA questions split={split}")
        trivia_split = self._load_trivia_questions(split)
        paired_rows = _pair_kilt_and_trivia_rows(list(kilt_split), list(trivia_split))
        return Dataset.from_list(paired_rows)

    def _extract_question_text(self, row: dict, idx: int) -> str:
        question = row.get("question", "")
        if not question:
            raise RuntimeError(f"{self.DATASET_TYPE} source example {idx} is missing question text")
        return question

    def _extract_answers_and_provenance(self, row: dict, idx: int) -> tuple[list[str], list[dict]]:
        answers, provenance = collect_answers_and_provenance(row.get("output", []))
        trivia_answer = row.get("trivia_answer")
        if isinstance(trivia_answer, dict):
            aliases = trivia_answer.get("aliases") or []
            for alias in aliases:
                alias = str(alias).strip()
                if alias and alias not in answers:
                    answers.append(alias)
            value = trivia_answer.get("value")
            if isinstance(value, str):
                value = value.strip()
                if value and value not in answers:
                    answers.append(value)
        return answers, provenance
