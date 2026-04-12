"""
Master reward function for RecaLLM GRPO training.

Routes by category (10 dataset families) and computes a composite reward
combining format, answer quality, and in-context retrieval quality.
See the paper (Section 4.3, Appendix D.3) for formal definitions.
"""

from .reward_utils import (
    two_way_subEM,
    extract_boxed_answer,
    ANSWER_RE,
    copy_overlap_f1,
    copy_overlap_coverage,
    get_recall_spans,
    normalize_answer,
    compute_qampari_answer_coverage_stats,
)
import math
import re

# =============================================================================
# Category configuration for final_reward()
# =============================================================================

CATEGORY_CONFIG = {
    "multi_hop_qa": {
        "datasets": {"hotpotqa", "musique", "2wikimultihopqa"},
        "hit_threshold": 0.4,
        "min_recall_span_length": 7,
        "gold_doc_mode": "standard",
        "free_first_spans": 4,
    },
    "single_hop_qa": {
        "datasets": {"nq", "triviaqa"},
        "hit_threshold": 0.4,
        "min_recall_span_length": 7,
        "gold_doc_mode": "top_k_gold",
        "gold_doc_top_k": 1,
        "free_first_spans": 2,
    },
    "recall": {
        "datasets": {"retrieval", "multi_niah"},
        "hit_threshold": 0.9,
        "min_recall_span_length": 5,
        "min_recall_span_length_overrides": {"multi_niah": 7},
        "gold_doc_mode": "standard",
        "free_first_spans": 2,
        "free_first_spans_overrides": {"multi_niah": 6},
    },
    "reasoning_retrieval": {
        "datasets": {"math_retrieval", "retrieval_math"},
        "hit_threshold": 0.9,
        "min_recall_span_length": 5,
        "gold_doc_mode": "standard",
        "free_first_spans": 2,
    },
    "short_context_math": {
        "datasets": {"dapo_math", "mcqa_math"},
        "hit_threshold": None,
        "min_recall_span_length": 5,
        "gold_doc_mode": "always_1",
        "free_first_spans": 2,
    },
    "icl": {
        "datasets": {"banking77", "massive"},
        "hit_threshold": 0.95,
        "min_recall_span_length": 7,
        "gold_doc_mode": "always_1",  # default; overridable via icl_gold_doc_mode kwarg
        "free_first_spans": 2,
    },
    "long_doc_qa": {
        "datasets": {"quality"},
        "hit_threshold": None,
        "min_recall_span_length": 7,
        "gold_doc_mode": "used_recall",
        "free_first_spans": 4,
    },
    "aggregation": {
        "datasets": {"majority_vote", "threshold_filter", "top_n_vote"},
        "hit_threshold": None,
        "min_recall_span_length": 5,
        "gold_doc_mode": "always_1",  # No gold docs; recall is counterproductive
        "free_first_spans": 2,
    },
    "reranking": {
        "datasets": {"msmarco_v2"},
        "hit_threshold": 0.7,
        "min_recall_span_length": 5,
        "gold_doc_mode": "top_k_gold",
        "free_first_spans": 4,
    },
    "citation_qa": {
        "datasets": {"qampari"},
        "hit_threshold": 0.7,
        "min_recall_span_length": 7,
        "gold_doc_mode": "top_k_gold",
        "gold_doc_top_k": 5,  # cap to top-5 (QAMPARI avg ~20 gold passages)
        "free_first_spans": 4,
    },
}

# Reverse lookup: dataset name -> category name
DATASET_TO_CATEGORY = {}
for _cat_name, _cat_cfg in CATEGORY_CONFIG.items():
    for _ds in _cat_cfg["datasets"]:
        DATASET_TO_CATEGORY[_ds] = _cat_name

# =============================================================================
# Helper functions
# =============================================================================


def compute_gold_doc_score_for_category(
    recall_spans: list[str],
    gold_docs: list[str],
    hit_threshold: float,
    gold_doc_mode: str,
    used_recall: bool,
    icl_gold_doc_top_k: int = 3,
) -> float:
    """
    Category-aware gold doc overlap scoring.

    Modes:
      "always_1"  -- return 1.0 unconditionally
      "used_recall" -- 1.0 if any recall spans exist, else 0.0
      "standard"  -- average best overlap per gold doc
      "top_k_gold" -- average best overlap for only the top-K highest-scoring gold docs
    """
    if gold_doc_mode == "always_1":
        return 1.0
    if gold_doc_mode == "used_recall":
        return 1.0 if used_recall else 0.0

    # "standard" and "top_k_gold" both need recall spans and gold docs
    if len(gold_docs) == 0:
        return 1.0
    if len(recall_spans) == 0:
        return 0.0

    # Compute per-gold-doc best overlap
    per_doc_scores = []
    for gold_doc in gold_docs:
        best = max((copy_overlap_f1(gold_doc, rs) for rs in recall_spans), default=0.0)
        per_doc_scores.append(min(1.0, best / hit_threshold))

    if gold_doc_mode == "standard":
        return sum(per_doc_scores) / len(per_doc_scores)

    elif gold_doc_mode == "top_k_gold":
        per_doc_scores.sort(reverse=True)
        k = min(icl_gold_doc_top_k, len(per_doc_scores))
        return sum(per_doc_scores[:k]) / k

    raise ValueError(f"Unknown gold_doc_mode: {gold_doc_mode}")


def _apply_label_presence_penalty(
    recall_spans: list[str],
    gold_docs: list[str],
    pos_doc_labels: list[str],
    hit_threshold: float,
    gold_doc_mode: str,
    gold_doc_top_k: int,
    label_penalty: float = 0.5,
    label_prefix: str = "label: ",
) -> float:
    """Re-score gold_doc_overlap with a label-presence penalty.

    For each gold doc, if the best-matching recall span does NOT contain
    the label marker string, multiply that doc's score by ``label_penalty``.

    The label marker is: ``f"{label_prefix}{pos_doc_labels[i]}"``.
    - ICL: label_prefix="label: ", pos_doc_labels=["9"] -> checks "label: 9"
    - Citation: label_prefix="", pos_doc_labels=["[3]"] -> checks "[3]"
    - Reranking: label_prefix="", pos_doc_labels=["[ID: 0]"] -> checks "[ID: 0]"

    Only meaningful for "standard" and "top_k_gold" modes.
    """
    if not gold_docs or not recall_spans:
        return 0.0
    if len(pos_doc_labels) != len(gold_docs):
        # Length mismatch -- fall back to standard scoring (no penalty)
        return compute_gold_doc_score_for_category(
            recall_spans, gold_docs, hit_threshold, gold_doc_mode,
            used_recall=True, icl_gold_doc_top_k=gold_doc_top_k,
        )

    per_doc_scores = []
    for doc_idx, gold_doc in enumerate(gold_docs):
        best_f1 = 0.0
        best_span_idx = -1
        for span_idx, rs in enumerate(recall_spans):
            f1 = copy_overlap_f1(gold_doc, rs)
            if f1 > best_f1:
                best_f1 = f1
                best_span_idx = span_idx

        doc_score = min(1.0, best_f1 / hit_threshold)

        # Penalty: if best span doesn't contain the label marker, penalize
        label_marker = f"{label_prefix}{pos_doc_labels[doc_idx]}"
        if best_span_idx >= 0 and label_marker not in recall_spans[best_span_idx]:
            doc_score *= label_penalty

        per_doc_scores.append(doc_score)

    if gold_doc_mode == "standard":
        return sum(per_doc_scores) / len(per_doc_scores)
    elif gold_doc_mode == "top_k_gold":
        per_doc_scores.sort(reverse=True)
        k = min(gold_doc_top_k, len(per_doc_scores))
        return sum(per_doc_scores[:k]) / k

    return sum(per_doc_scores) / len(per_doc_scores)  # fallback


def is_correct_format(solution_str: str) -> bool:
    """
    Makes sure there is exactly one Answer: after thinking, one </think> and no <think>.
    """
    correct_think_tags = solution_str.count("</think>") == 1 and solution_str.count("<think>") == 0
    completion = solution_str.split("</think>")[-1]
    single_answer = completion.count("Answer:") == 1
    has_answer = ANSWER_RE.search(completion) is not None
    return correct_think_tags and single_answer and has_answer


def _compute_task_specific_format(
    category: str,
    model_answer: str,
    extra_info: dict,
    ground_truth: dict,
) -> float | None:
    """Compute task-specific format score for citation and reranking.

    Returns None for categories that don't have task-specific format.
    - citation_qa: 1.0 if answer contains at least one [N] citation, else 0.0.
    - reranking: sqrt(n_valid_ranked / n_total). Only counts doc IDs that exist
      in relevance_grades (prevents gaming with random IDs).
    """
    if category == "citation_qa":
        answer = model_answer or ""
        has_citation = bool(re.search(r'\[\d+\]', answer))
        # Strip citations and whitespace -- if nothing remains, it's citation-only
        text_only = re.sub(r'\[\d+\]', '', answer).strip()
        if not text_only:
            return 0.0
        return 1.0 if has_citation else 0.0

    elif category == "reranking":
        import json as _json
        ranked_ids = _parse_ranking(model_answer or "")

        # Get relevance_grades to validate ranked IDs and determine n_total
        relevance_grades = ground_truth.get("relevance_grades", "{}") if ground_truth else "{}"
        if isinstance(relevance_grades, str):
            try:
                relevance_grades = _json.loads(relevance_grades)
            except (ValueError, TypeError):
                relevance_grades = {}

        n_total = len(relevance_grades) if isinstance(relevance_grades, dict) else 0
        if n_total == 0:
            return 1.0  # Can't compute; don't penalize

        # Only count IDs that actually exist in the prompt
        valid_ids = set(relevance_grades.keys())
        n_valid_ranked = sum(1 for rid in ranked_ids if rid in valid_ids)

        return math.sqrt(min(n_valid_ranked, n_total) / n_total)

    return None


def _parse_ranking(text: str) -> list[str]:
    """Extract document IDs from a ranking string like '7 > 3 > 12'."""
    text = re.sub(r'[\[\]:]', '', text)
    text = text.lower().replace('id', '').replace('doc', '').replace('passage', '')
    ids = [tok.strip() for tok in text.split('>') if tok.strip()]
    seen = set()
    return [pid for pid in ids if not (pid in seen or seen.add(pid))]


def _ndcg_at_k(model_answer: str, ground_truth: dict, k: int = 10) -> float:
    """Compute NDCG@K for a reranking task.

    Args:
        model_answer: Model's ranking output (e.g., "7 > 3 > 12 > ...")
        ground_truth: Must contain 'relevance_grades' dict mapping assigned ID -> grade
        k: Cutoff for NDCG computation
    """
    import json as _json
    ranking = _parse_ranking(model_answer)
    relevance_grades = ground_truth.get('relevance_grades')
    if isinstance(relevance_grades, str):
        try:
            relevance_grades = _json.loads(relevance_grades)
        except (ValueError, TypeError):
            relevance_grades = {}
    if not relevance_grades:
        return 0.0

    # DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(ranking[:k]):
        grade = relevance_grades.get(doc_id, 0)
        dcg += grade / math.log2(i + 2)

    # IDCG@k: ideal ranking (all grades sorted descending)
    ideal_grades = sorted(relevance_grades.values(), reverse=True)
    idcg = 0.0
    for i, grade in enumerate(ideal_grades[:k]):
        idcg += grade / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_citation_f1(
    model_answer: str,
    recall_spans: list[str],
    extra_info: dict,
) -> dict:
    """
    Compute Citation F1 for QAMPARI with recall-backed and partial credit.

    Two tiers of citation credit:
    - **Recall-backed** (1.0 each): citation [N] in answer AND [N] in a recall span.
    - **Non-recall-backed** (0.5 each): citation [N] in answer but NOT in any recall span.

    Recall is capped at @5: ``recall = weighted_correct / min(5, len(gold_doc_ids))``.
    """
    import json as _json

    _zero = {
        "citation_f1": 0.0, "citation_recall": 0.0, "citation_precision": 0.0,
        "n_cited": 0, "n_recalled": 0,
    }

    # 1. Parse cited doc IDs from the answer text
    cited_doc_ids = set(int(x) for x in re.findall(r'\[(\d+)\]', model_answer or ""))
    if not cited_doc_ids:
        return _zero

    # 2. Determine which doc IDs are recall-backed
    recalled_doc_ids: set[int] = set()
    for span in recall_spans:
        for doc_id in re.findall(r'\[(\d+)\]', span):
            recalled_doc_ids.add(int(doc_id))

    # 3. Load gold doc IDs from settings metadata
    settings = extra_info.get('settings', '{}')
    if isinstance(settings, str):
        settings = _json.loads(settings)
    gold_doc_ids = set(settings.get('gold_doc_ids', []))

    if not gold_doc_ids:
        return _zero

    # 4. Two-tier credit: recall-backed = 1.0, non-recall-backed = 0.5
    recalled_cited = cited_doc_ids & recalled_doc_ids
    non_recalled_cited = cited_doc_ids - recalled_doc_ids

    correct_recalled = recalled_cited & gold_doc_ids
    correct_non_recalled = non_recalled_cited & gold_doc_ids

    weighted_correct = len(correct_recalled) + 0.5 * len(correct_non_recalled)

    # 5. Recall @5 cap
    recall_denom = min(5, len(gold_doc_ids))
    weighted_correct_capped = min(weighted_correct, recall_denom)
    recall = weighted_correct_capped / recall_denom

    # 6. Precision (weighted denominator to match weighted numerator)
    precision_denom = len(recalled_cited) + 0.5 * len(non_recalled_cited)
    precision = weighted_correct / precision_denom if precision_denom > 0 else 1.0

    # 7. F1
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * recall * precision / (recall + precision)

    return {
        "citation_f1": f1,
        "citation_recall": recall,
        "citation_precision": precision,
        "n_cited": len(cited_doc_ids),
        "n_recalled": len(recalled_doc_ids),
    }


def _parse_qampari_answer_variants(
    settings: dict,
    ground_truth: dict | None,
) -> list[list[str]]:
    import json as _json

    answer_variants = settings.get("answer_variants")
    if isinstance(answer_variants, list):
        parsed = []
        for variants in answer_variants:
            if not isinstance(variants, list):
                continue
            cleaned = []
            seen = set()
            for variant in variants:
                if not isinstance(variant, str):
                    continue
                stripped = variant.strip()
                if not stripped:
                    continue
                lowered = stripped.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                cleaned.append(stripped)
            parsed.append(cleaned)
        if parsed:
            return parsed

    answer_field = ground_truth.get("answer", "[]") if ground_truth else "[]"
    if isinstance(answer_field, str):
        try:
            parsed_answers = _json.loads(answer_field)
        except (ValueError, TypeError):
            parsed_answers = []
    elif isinstance(answer_field, list):
        parsed_answers = answer_field
    else:
        parsed_answers = []

    return [[str(answer).strip()] for answer in parsed_answers if str(answer).strip()]


def compute_qampari_answer_coverage(
    model_answer: str,
    extra_info: dict,
    ground_truth: dict | None,
) -> dict:
    import json as _json

    _zero = {
        "qampari_answer_coverage": 0.0,
        "qampari_covered_answers": 0,
        "qampari_available_answers": 0,
    }
    if not model_answer:
        return _zero

    settings = extra_info.get("settings", "{}") if extra_info else "{}"
    if isinstance(settings, str):
        try:
            settings = _json.loads(settings)
        except (ValueError, TypeError):
            settings = {}
    elif not isinstance(settings, dict):
        settings = {}

    answer_to_doc_ids = settings.get("answer_to_doc_ids", {})
    if not isinstance(answer_to_doc_ids, dict):
        return _zero

    available_answer_indices = []
    for answer_idx, doc_ids in answer_to_doc_ids.items():
        if not isinstance(doc_ids, list) or not doc_ids:
            continue
        try:
            available_answer_indices.append(int(answer_idx))
        except (TypeError, ValueError):
            continue
    if not available_answer_indices:
        return _zero

    answer_variants = _parse_qampari_answer_variants(settings, ground_truth)
    if not answer_variants:
        return _zero

    return compute_qampari_answer_coverage_stats(
        model_answer=model_answer,
        answer_variants=answer_variants,
        available_answer_indices=available_answer_indices,
    )


def score_answer(model_answer: str, data_source: str, ground_truth: dict) -> float:
    """
    Given a model answer, data source (dataset name), and the ground truth dict,
    returns the answer score as a float.
    """
    if data_source in ['hotpotqa', 'musique', '2wikimultihopqa']:
        aliases = ground_truth['answer'].split('|||')
        score = max(two_way_subEM(model_answer, a) for a in aliases)
    elif data_source in ['math_retrieval', 'retrieval_math']:
        if ground_truth['answer'] in model_answer:
            score = 1.0
        elif f"{ground_truth['math_answer']}" in model_answer:
            score = 0.25
        else:
            score = 0.0
    elif data_source in ['retrieval']:
        score = 1.0 if ground_truth['answer'] in model_answer else 0.0
    elif data_source in ['multi_niah']:
        positive_answers = ground_truth['answer'].split()
        negative_answers = ground_truth['neg_answer'].split()
        num_pos_answers = sum(1 for ans in positive_answers if ans in model_answer)
        num_neg_answers = sum(1 for ans in negative_answers if ans in model_answer)
        score = max(0.0, (num_pos_answers - num_neg_answers) / len(positive_answers))
    elif data_source in ['mcqa_math']:
        boxed_answer = extract_boxed_answer(model_answer)
        if boxed_answer is not None and (ground_truth['answer'] in boxed_answer or ground_truth['math_answer'] in boxed_answer):
            score = 1.0
        else:
            score = 0.0
    elif data_source in ['dapo_math']:
        boxed_answer = extract_boxed_answer(model_answer)
        score = 1.0 if boxed_answer is not None and ground_truth['answer'] in boxed_answer else 0.0
    elif data_source in ['nq', 'triviaqa']:
        aliases = ground_truth['answer'].split('|||')
        score = max(two_way_subEM(model_answer, a) for a in aliases)
    elif data_source in ['banking77', 'massive']:
        score = 1.0 if normalize_answer(model_answer) == normalize_answer(ground_truth['answer']) else 0.0
    elif data_source in ['quality']:
        mc_match = re.search(r'[A-D]', model_answer)
        if mc_match is not None:
            score = 1.0 if mc_match.group(0) == ground_truth['answer'] else 0.0
        else:
            score = 0.0
    elif data_source in ['majority_vote', 'threshold_filter']:
        # Four-tier matching: exact, identifier-only, word-boundary substring, identifier at first/last word.
        # Does NOT use normalize_answer (which strips article "A" from "Proposal A").
        gt = ' '.join(ground_truth['answer'].strip().lower().split())
        ma = ' '.join(model_answer.strip().lower().rstrip('.,;:!?').split())
        gt_id = gt.split()[-1]
        ma_words = ma.split()
        if gt == ma:
            score = 1.0
        elif ma == gt_id:
            score = 1.0
        elif re.search(r'(?<!\w)' + re.escape(gt) + r'(?!\w)', ma):
            score = 1.0
        elif ma_words and (ma_words[0] == gt_id or ma_words[-1] == gt_id):
            score = 1.0
        else:
            score = 0.0
    elif data_source == 'top_n_vote':
        # Multi-answer set matching with spam penalty.
        gt_options = ground_truth['answer'].split('|||')
        n_top = len(gt_options)
        ma_lower = model_answer.strip().lower()
        ma_original = model_answer.strip()

        def _top_n_vote_match(opt: str) -> bool:
            opt_lower = opt.strip().lower()
            if re.search(r'(?<!\w)' + re.escape(opt_lower) + r'(?!\w)', ma_lower):
                return True
            opt_id = opt.strip().split()[-1]
            if re.search(r'(?<!\w)' + re.escape(opt_id) + r'(?!\w)', ma_original):
                return True
            return False

        n_correct = sum(1 for opt in gt_options if _top_n_vote_match(opt))
        neg_options_str = ground_truth.get('neg_answer', '')
        neg_options = neg_options_str.split('|||') if neg_options_str else []
        n_incorrect = sum(1 for opt in neg_options if _top_n_vote_match(opt))
        score = max(0.0, (n_correct - n_incorrect) / n_top)
    elif data_source in ['msmarco_v2']:
        score = _ndcg_at_k(model_answer, ground_truth, k=10)
    elif data_source in ['qampari']:
        # Citation F1 is computed separately in final_reward() after recall spans
        # are available (needs recall-backed citation filtering).
        score = 0.0
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    return score


def fractional_correct_recall_usage(
    completion: str,
    recall_start: str = "<recall>",
    recall_end: str = "</recall>",
    min_recall_length: int = None,
) -> float:
    """
    Return a score between 0.0 and 1.0 based on the relative number of errors
    in recall usage to the number of recall uses.
    """
    recall_spans = get_recall_spans(completion, recall_start, recall_end)
    n_recall_spans = len(recall_spans)
    if n_recall_spans == 0:
        return 1.0

    # check number of recall starts vs ends
    n_recall_starts = completion.count(recall_start)
    n_recall_ends = completion.count(recall_end)
    tag_diff = abs(n_recall_starts - n_recall_ends)

    # check the number of too-short recall spans
    n_too_short = 0
    if min_recall_length is not None:
        n_too_short = sum(1 for span in recall_spans if len(span) < min_recall_length)

    # number of allowed errors is sqrt of number of recall spans
    n_errors = tag_diff + n_too_short
    n_allowed_errors = math.sqrt(n_recall_spans)

    return max(0.0, 1.0 - n_errors / n_allowed_errors)


def smoothed_geometric_mean(*values, eps: float = 0.01) -> float:
    """
    Computes a smoothed geometric mean of the given values.
    Adds eps before computing the geometric mean, then subtracts eps from the result.
    """
    prod = 1.0
    n = len(values)
    for v in values:
        prod *= (v + eps)
    return (prod ** (1.0 / n)) - eps


def density_recall_use_v2(
    completion: str,
    response_length: int,
    num_gold_docs: int,
    threshold: float = 4.0,
    half_life: float = 4.0,
    free_first_spans: int | None = None,
    free_first_spans_min: int | None = 2,
    free_first_spans_max: int | None = 8,
    recall_start: str = "<recall>",
    recall_end: str = "</recall>",
) -> float:
    """
    Density penalty with free_first_spans.

    The first N recall spans are free -- not counted toward density. Beyond that,
    apply exponential decay when density exceeds threshold spans per 1024 tokens.
    """
    recall_spans = get_recall_spans(completion, recall_start, recall_end)
    n_recall_spans = len(recall_spans)
    if response_length == 0 or n_recall_spans == 0:
        return 1.0

    if free_first_spans is None:
        assert free_first_spans_min is not None and free_first_spans_max is not None
        free_first_spans = max(free_first_spans_min, min(num_gold_docs, free_first_spans_max))
    n_counted = max(0, n_recall_spans - free_first_spans)
    if n_counted == 0:
        return 1.0

    density = n_counted / (response_length / 1024.0)
    if density <= threshold:
        return 1.0
    n_half_lives_over = (density - threshold) / half_life
    return 1.0 / (2.0 ** n_half_lives_over)


# =============================================================================
# Master reward function
# =============================================================================


def final_reward(
        data_source: str,
        solution_str: str,
        ground_truth: dict = None,
        extra_info: dict = None,
        recall_start_str: str = "<recall>",
        recall_end_str: str = "</recall>",
        # Composite reward weights
        hybrid_format_frac: float = 0.2,
        hybrid_additive_frac: float = 0.4,
        hybrid_multiplicative_frac: float = 0.4,
        additive_answer_weight: float = 0.5,
        # Task-specific format blending (citation_qa + reranking only)
        task_specific_format_frac: float = 0.5,
        qampari_answer_coverage_weight: float = 0.0,
        geom_smoothing_eps: float = 0.01,
        # Density penalty params
        density_threshold: float = 6.0,
        density_half_life: float = 4.0,
        free_first_spans_min: int | None = None,
        free_first_spans_max: int | None = None,
        # ICL-specific ablation knob
        icl_gold_doc_mode: str = "always_1",
        icl_gold_doc_top_k: int = 2,
        icl_label_penalty: float = 0.5,
        # No-mask ablation: bypass gold_doc_overlap (force 1.0 for all categories)
        disable_gold_doc_overlap: bool = False,
        # No-gold-doc ablation: bypass correct_recall_usage and density_multiplier
        disable_recall_penalties: bool = False,
        **kwargs,
) -> dict:
    """
    Master reward function that routes by category. Combines format, answer
    quality, and in-context retrieval quality into a composite reward.

    R = 0.2 * R_format + 0.4 * R_add + 0.4 * R_mult
    R_add = 0.5 * R_ans + 0.5 * R_ret
    R_mult = smoothed_geometric_mean(R_ans, R_ret)

    See CATEGORY_CONFIG for per-category settings (hit thresholds, gold doc
    modes, free spans, etc.) and the paper for formal definitions.
    """
    # 1. Look up category
    category = DATASET_TO_CATEGORY[data_source]
    cat_cfg = CATEGORY_CONFIG[category]

    # 2. Resolve min_recall_span_length (with per-dataset override support)
    min_recall_length = cat_cfg.get("min_recall_span_length_overrides", {}).get(
        data_source, cat_cfg["min_recall_span_length"]
    )

    # 3. Extract answer from completion (after </think>)
    completion = solution_str.split("</think>")[-1]
    match = ANSWER_RE.search(completion)
    has_answer = match is not None

    # 4. Compute answer_score (deferred for citation_qa -- needs recall spans first)
    model_answer = match.group(1) if has_answer else None
    if has_answer and category != "citation_qa":
        answer_score = score_answer(model_answer, data_source, ground_truth)
    else:
        answer_score = 0.0

    # 5. Get recall spans from full solution and check format
    recall_spans = get_recall_spans(solution_str, recall_start_str, recall_end_str)
    used_recall = len(recall_spans) > 0
    correct_format = is_correct_format(solution_str)

    # 5b. Deferred citation F1 for citation_qa (needs recall_spans from step 5)
    #     Always populate citation_metrics for ALL categories (NaN for non-citation)
    #     so that every sample in a batch has identical keys.
    _nan = float('nan')
    qampari_metrics = {
        "qampari_answer_coverage": _nan,
        "qampari_covered_answers": _nan,
        "qampari_available_answers": _nan,
        "qampari_answer_score_blended": _nan,
    }
    if category == "citation_qa":
        if has_answer:
            citation_metrics = compute_citation_f1(model_answer, recall_spans, extra_info)
            qampari_coverage = compute_qampari_answer_coverage(model_answer, extra_info, ground_truth)
            answer_score = (
                (1.0 - qampari_answer_coverage_weight) * citation_metrics["citation_f1"]
                + qampari_answer_coverage_weight * qampari_coverage["qampari_answer_coverage"]
            )
            qampari_metrics = {
                **qampari_coverage,
                "qampari_answer_score_blended": answer_score,
            }
        else:
            citation_metrics = {
                "citation_f1": 0.0, "citation_recall": 0.0,
                "citation_precision": 0.0, "n_cited": 0, "n_recalled": 0,
            }
            qampari_metrics = {
                "qampari_answer_coverage": 0.0,
                "qampari_covered_answers": 0,
                "qampari_available_answers": 0,
                "qampari_answer_score_blended": 0.0,
            }
    else:
        citation_metrics = {
            "citation_f1": _nan, "citation_recall": _nan,
            "citation_precision": _nan, "n_cited": _nan, "n_recalled": _nan,
        }

    # 5c. Task-specific format blending (citation_qa + reranking)
    task_specific_format = _compute_task_specific_format(
        category, model_answer, extra_info, ground_truth,
    )
    if task_specific_format is not None:
        correct_format = (
            (1 - task_specific_format_frac) * float(correct_format)
            + task_specific_format_frac * task_specific_format
        )

    # 6. Compute fractional correct recall usage
    if disable_recall_penalties:
        correct_recall_usage = 1.0
    else:
        correct_recall_usage = fractional_correct_recall_usage(
            solution_str,
            recall_start=recall_start_str,
            recall_end=recall_end_str,
            min_recall_length=min_recall_length,
        )

    # 7. Compute density_multiplier (v2 with free_first_spans)
    response_length = extra_info['response_length']
    gold_docs = ground_truth.get('pos_docs', []) if ground_truth else []
    num_gold_docs = len(gold_docs)
    if disable_recall_penalties:
        density_multiplier = 1.0
    else:
        free_first_spans = cat_cfg.get("free_first_spans_overrides", {}).get(
            data_source, cat_cfg.get("free_first_spans")
        )
        if free_first_spans_min is not None or free_first_spans_max is not None:
            assert free_first_spans_min is not None and free_first_spans_max is not None, (
                "free_first_spans_min and free_first_spans_max must be provided together"
            )
            free_first_spans = None

        density_multiplier = density_recall_use_v2(
            solution_str,
            response_length,
            num_gold_docs=num_gold_docs,
            threshold=density_threshold,
            half_life=density_half_life,
            free_first_spans=free_first_spans,
            free_first_spans_min=free_first_spans_min,
            free_first_spans_max=free_first_spans_max,
            recall_start=recall_start_str,
            recall_end=recall_end_str,
        )

    # 8. Determine gold_doc_mode and top_k for this category
    gold_doc_mode = cat_cfg["gold_doc_mode"]
    gold_doc_top_k = cat_cfg.get("gold_doc_top_k", icl_gold_doc_top_k)
    if category == "icl":
        gold_doc_mode = icl_gold_doc_mode  # override from kwarg
        gold_doc_top_k = icl_gold_doc_top_k
    hit_threshold = cat_cfg["hit_threshold"]

    # 9. Compute gold_doc_overlap
    if disable_gold_doc_overlap:
        gold_doc_overlap = 1.0
    else:
        eff_hit_threshold = hit_threshold if hit_threshold is not None else 1.0
        gold_doc_overlap = compute_gold_doc_score_for_category(
            recall_spans=recall_spans,
            gold_docs=gold_docs,
            hit_threshold=eff_hit_threshold,
            gold_doc_mode=gold_doc_mode,
            used_recall=used_recall,
            icl_gold_doc_top_k=gold_doc_top_k,
        )

    # 9b. Label-presence penalty (ICL + citation_qa + reranking)
    _label_penalty_categories = {
        "icl": "label: ",
        "citation_qa": "",
        "reranking": "",
    }
    if not disable_gold_doc_overlap and category in _label_penalty_categories and (
        category != "icl" or gold_doc_mode in ("standard", "top_k_gold")
    ):
        import json as _json
        settings_str = extra_info.get("settings", "") if extra_info else ""
        if isinstance(settings_str, str) and settings_str:
            try:
                settings = _json.loads(settings_str)
            except (ValueError, TypeError):
                settings = {}
        else:
            settings = {}
        pos_doc_labels = settings.get("pos_doc_labels", [])
        if pos_doc_labels and len(pos_doc_labels) == len(gold_docs):
            label_prefix = _label_penalty_categories[category]
            gold_doc_overlap = _apply_label_presence_penalty(
                recall_spans=recall_spans,
                gold_docs=gold_docs,
                pos_doc_labels=pos_doc_labels,
                hit_threshold=eff_hit_threshold,
                gold_doc_mode=gold_doc_mode,
                gold_doc_top_k=gold_doc_top_k,
                label_penalty=icl_label_penalty,
                label_prefix=label_prefix,
            )

    # 10. Combine components
    gold_doc_score = gold_doc_overlap * correct_recall_usage * density_multiplier

    additive = answer_score * additive_answer_weight + gold_doc_score * (1.0 - additive_answer_weight)
    multiplicative = smoothed_geometric_mean(answer_score, gold_doc_score, eps=geom_smoothing_eps)

    score = (
        additive * hybrid_additive_frac
        + multiplicative * hybrid_multiplicative_frac
        + correct_format * hybrid_format_frac
    )

    # 11. Return all components for logging
    result = {
        "score": score,
        "correct_format": correct_format,
        "answer_score": answer_score,
        "gold_doc_overlap": gold_doc_overlap,
        "gold_doc_score": gold_doc_score,
        "correct_recall_usage": correct_recall_usage,
        "recall_density_multiplier": density_multiplier,
        "additive_component": additive,
        "multiplicative_component": multiplicative,
        "has_answer": has_answer,
        "used_recall": used_recall,
        "n_recall_spans": len(recall_spans),
        "category": category,
        "num_gold_docs": num_gold_docs,
    }
    # Always include ALL optional keys so every sample has identical key sets.
    # VeRL's DataProto crashes if key sets differ across samples in a batch.
    result["task_specific_format"] = task_specific_format if task_specific_format is not None else _nan
    result["citation_recall"] = citation_metrics["citation_recall"]
    result["citation_precision"] = citation_metrics["citation_precision"]
    result["n_cited"] = citation_metrics["n_cited"]
    result["n_recalled"] = citation_metrics["n_recalled"]
    result["qampari_answer_coverage"] = qampari_metrics["qampari_answer_coverage"]
    result["qampari_covered_answers"] = qampari_metrics["qampari_covered_answers"]
    result["qampari_available_answers"] = qampari_metrics["qampari_available_answers"]
    result["qampari_answer_score_blended"] = qampari_metrics["qampari_answer_score_blended"]
    return result
