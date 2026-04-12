from transformers import TrainerCallback
from datasets import Dataset, concatenate_datasets


class EvalOnFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0:
            control.should_evaluate = True


def sample_by_dataset(
        dataset: Dataset,
        sample_fraction: float,
        seed: int = 42
) -> Dataset:
    """
    Stratified sample of the dataset by 'dataset' field.
    """
    dataset_types = dataset.unique('dataset')
    datasets_by_type = {dt: dataset.filter(lambda x: x['dataset'] == dt) for dt in dataset_types}
    sampled_datasets = []
    for dt, dt_dataset in datasets_by_type.items():
        sample_size = int(len(dt_dataset) * sample_fraction)
        sampled_dt_dataset = dt_dataset.shuffle(seed=seed).select(range(sample_size))
        sampled_datasets.append(sampled_dt_dataset)
    sampled_dataset = concatenate_datasets(sampled_datasets)
    return sampled_dataset
