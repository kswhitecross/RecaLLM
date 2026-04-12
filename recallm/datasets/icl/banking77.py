"""
Raw Banking77 dataset for ICL training.

Loads Banking77 from HuggingFace, selects test examples, and builds
balanced demonstration pools via round-robin across label groups.
Returns ICLExample instances with raw integer labels — label formatting
(numeric permutation vs text names) is handled by the prompt-building layer.
"""

import random

import numpy as np
from datasets import load_dataset
from torch.utils.data import Dataset

from recallm.datasets.icl import ICLExample


class Banking77Dataset(Dataset):
    """Raw Banking77 dataset. Loads from HF, provides ICLExample instances."""

    def __init__(self, n_examples: int, seed: int = 0, max_demos: int = 2000):
        super().__init__()
        self.n_examples = int(n_examples)
        self.max_demos = int(max_demos)

        # Load dataset from HuggingFace (parquet revision for datasets>=4.0 compat)
        ds = load_dataset("PolyAI/banking77", split="train", revision="refs/convert/parquet")
        self.texts: list[str] = ds["text"]
        self.labels: list[int] = ds["label"]
        self.label_names: list[str] = ds.features["label"].names
        self.num_labels = len(self.label_names)

        # Group source indices by label for balanced demo selection
        self._examples_by_label: dict[int, list[int]] = {}
        for src_idx, label in enumerate(self.labels):
            self._examples_by_label.setdefault(label, []).append(src_idx)

        # Pre-shuffle test indices deterministically
        rng = np.random.default_rng(seed)
        all_indices = rng.permutation(len(self.texts)).tolist()
        self._test_indices = all_indices[:self.n_examples]

        # Pre-compute per-example seeds
        ss = np.random.SeedSequence(int(seed))
        children = ss.spawn(self.n_examples)
        self.base_seeds: list[int] = [
            int(cs.generate_state(1, dtype=np.uint32)[0]) for cs in children
        ]

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx: int) -> ICLExample:
        if idx < 0 or idx >= self.n_examples:
            raise IndexError(idx)

        test_src_idx = self._test_indices[idx]
        test_text = self.texts[test_src_idx]
        test_label = self.labels[test_src_idx]

        py_rng = random.Random(self.base_seeds[idx])

        # Build balanced demonstration pool via round-robin
        demo_texts, demo_labels = self._build_balanced_demos(
            test_src_idx, py_rng, max_demos=self.max_demos
        )

        return ICLExample(
            id=idx,
            test_text=test_text,
            test_label=test_label,
            demo_texts=demo_texts,
            demo_labels=demo_labels,
            label_names=self.label_names,
            num_labels=self.num_labels,
            dataset_type="banking77",
        )

    def _build_balanced_demos(
        self, exclude_idx: int, rng: random.Random,
        max_demos: int = 2000,
    ) -> tuple[list[str], list[int]]:
        """Build a balanced round-robin demonstration pool excluding one example.

        Groups all source examples by label, shuffles within each group,
        then round-robins across groups. The first k elements of the returned
        lists are always approximately balanced across labels.

        Stops once max_demos are collected (no need to exhaust all pools since
        set_largest_k() will select a subset that fits the target context).
        """
        # Build per-label pools, excluding the test example
        label_pools: dict[int, list[int]] = {}
        for label_id, src_indices in self._examples_by_label.items():
            pool = [i for i in src_indices if i != exclude_idx]
            rng.shuffle(pool)
            label_pools[label_id] = pool

        # Round-robin across label groups
        label_ids = sorted(label_pools.keys())
        rng.shuffle(label_ids)  # randomize which label comes first each round

        demo_texts: list[str] = []
        demo_labels: list[int] = []

        # Round-robin, stopping once we have enough demos
        max_rounds = max(len(p) for p in label_pools.values())
        for round_idx in range(max_rounds):
            for label_id in label_ids:
                pool = label_pools[label_id]
                if round_idx < len(pool):
                    src_idx = pool[round_idx]
                    demo_texts.append(self.texts[src_idx])
                    demo_labels.append(self.labels[src_idx])
                    if len(demo_texts) >= max_demos:
                        return demo_texts, demo_labels

        return demo_texts, demo_labels
