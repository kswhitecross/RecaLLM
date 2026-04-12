from dataclasses import dataclass


@dataclass
class ICLExample:
    """Raw ICL example: a test query + ordered pool of demonstrations.

    Bridge between raw dataset classes (Banking77Dataset, MassiveDataset)
    and the prompt-building layer (ICLPromptDataset).
    """

    id: int
    test_text: str
    test_label: int  # original integer label
    demo_texts: list[str]  # pre-ordered in balanced round-robin
    demo_labels: list[int]  # corresponding original integer labels
    label_names: list[str]  # human-readable names for all labels
    num_labels: int
    dataset_type: str  # "banking77" or "massive"
