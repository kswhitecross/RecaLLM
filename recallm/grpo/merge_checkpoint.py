"""
Merge FSDP-sharded VeRL checkpoints into a bf16 HuggingFace model.

Takes a global_step_X directory (or its actor/ subdirectory) and produces a
bf16/ folder containing a standard HF model loadable with:
    AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)

Also copies modelling_recallm.py into both the huggingface/ config dir (needed
by the merger) and the output bf16/ dir (needed for downstream loading).

Usage:
    python -m recallm.grpo.merge_checkpoint /path/to/global_step_150
    python -m recallm.grpo.merge_checkpoint /path/to/global_step_150/actor
    python -m recallm.grpo.merge_checkpoint /path/to/global_step_150 --output_dir /path/to/output
    python -m recallm.grpo.merge_checkpoint /path/to/global_step_150 --cpu
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

MODELLING_RECALLM_SRC = Path(__file__).resolve().parent.parent / "modelling_recallm.py"


def resolve_actor_dir(checkpoint_path: Path) -> Path:
    """Given a global_step_X dir or its actor/ subdir, return the actor/ dir."""
    if checkpoint_path.name == "actor":
        return checkpoint_path
    actor_dir = checkpoint_path / "actor"
    if actor_dir.is_dir():
        return actor_dir
    raise FileNotFoundError(
        f"Cannot find actor/ directory. Expected either:\n"
        f"  {checkpoint_path}/actor/\n"
        f"  or pass the actor/ directory directly."
    )


def ensure_modelling_file(directory: Path) -> bool:
    """Copy modelling_recallm.py to directory if not already present. Returns True if copied."""
    target = directory / "modelling_recallm.py"
    if target.exists():
        return False
    if not MODELLING_RECALLM_SRC.exists():
        raise FileNotFoundError(f"Source modelling_recallm.py not found at {MODELLING_RECALLM_SRC}")
    shutil.copy2(MODELLING_RECALLM_SRC, target)
    return True


def main():
    parser = argparse.ArgumentParser(description="Merge FSDP checkpoint to bf16 HuggingFace model")
    parser.add_argument("checkpoint_path", type=Path, help="Path to global_step_X dir or its actor/ subdir")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output directory (default: <actor_dir>/bf16)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU-only execution (hides all GPUs)")
    args = parser.parse_args()

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    actor_dir = resolve_actor_dir(args.checkpoint_path.resolve())
    hf_config_dir = actor_dir / "huggingface"
    output_dir = args.output_dir or (actor_dir.parent / "bf16")

    # Validate
    if not (actor_dir / "fsdp_config.json").exists():
        print(f"ERROR: {actor_dir}/fsdp_config.json not found -- is this an FSDP checkpoint?", file=sys.stderr)
        sys.exit(1)
    if not hf_config_dir.exists():
        print(f"ERROR: {hf_config_dir} not found -- checkpoint missing HF config/tokenizer", file=sys.stderr)
        sys.exit(1)
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"ERROR: Output directory {output_dir} already exists and is non-empty. "
              f"Remove it first or choose a different --output_dir.", file=sys.stderr)
        sys.exit(1)

    # Step 1: Ensure modelling_recallm.py is in huggingface/ dir so AutoConfig can find it
    if ensure_modelling_file(hf_config_dir):
        print(f"Copied modelling_recallm.py -> {hf_config_dir}/")
    else:
        print(f"modelling_recallm.py already present in {hf_config_dir}/")

    # Step 2: Run the VeRL model merger
    print(f"\nMerging FSDP shards from {actor_dir}")
    print(f"Output: {output_dir}\n")

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir=str(actor_dir),
        target_dir=str(output_dir),
        hf_model_config_path=str(hf_config_dir),
        trust_remote_code=True,
    )

    merger = FSDPModelMerger(config)
    merger.merge_and_save()
    merger.cleanup()

    # Step 3: Copy modelling_recallm.py to output dir for trust_remote_code loading
    ensure_modelling_file(output_dir)
    print(f"Copied modelling_recallm.py -> {output_dir}/")

    print(f"\nDone! Model saved to {output_dir}")
    print(f"Load with: AutoModelForCausalLM.from_pretrained('{output_dir}', trust_remote_code=True)")


if __name__ == "__main__":
    main()
