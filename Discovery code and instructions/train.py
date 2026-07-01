"""
discovery_ft.train
==================

Fine-tune a SMALL embedding model on the triplets we built.

Model choice: Qwen3-Embedding-0.6B. It is the smallest of the Qwen3 embedding
family, it is instruction-aware (built for the query/document asymmetry this
project lives on), it is Apache-2.0, and at 0.6B it FULLY fine-tunes inside a
single 12 GB GPU. That makes it the right fast-iteration model for the local
server. (Move to the 4B model with LoRA, or to Anvil, only once the 0.6B loop
proves the approach works.)

Loss: MultipleNegativesRankingLoss (MNRL). Given (anchor, positive, negative)
rows it pulls the anchor toward its positive and pushes it away from its own
negative AND from every other positive in the batch (in-batch negatives). It is
the standard, strong starting point. Bigger batch = more in-batch negatives =
better signal, so we push batch size as high as 12 GB allows and use gradient
checkpointing to buy headroom.

What this script does NOT do: MarginMSE / cross-encoder distillation (the full
GPL recipe). Add that only if MNRL plateaus -- it is a second-stage refinement,
not a starting point.

Usage (single GPU):
    python -m discovery_ft.train \
        --train dataset/train.jsonl --eval dataset/eval.jsonl \
        --model Qwen/Qwen3-Embedding-0.6B \
        --out runs/qwen06b-ft --epochs 3 --batch 16

Usage (both GPUs, data-parallel):
    torchrun --nproc_per_node=2 -m discovery_ft.train ...same args...
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_jsonl_dataset(path: str):
    """Load a triplet JSONL into a HF Dataset with columns anchor/positive/negative."""
    from datasets import load_dataset
    ds = load_dataset("json", data_files=path, split="train")
    # sentence-transformers maps columns positionally: 1st=anchor, 2nd=positive,
    # 3rd=negative. Make sure the order is right and no extra columns linger.
    keep = ["anchor", "positive", "negative"]
    ds = ds.select_columns([c for c in keep if c in ds.column_names])
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="dataset/train.jsonl")
    ap.add_argument("--eval", default="dataset/eval.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out", default="runs/qwen06b-ft")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16,
                    help="per-device batch; lower to 8 or 4 if you hit OOM")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-seq", type=int, default=512,
                    help="truncation length; 512 fits comfortably on 12GB. "
                         "Raise toward 1024+ only on Anvil.")
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="raise to simulate a bigger batch without more VRAM")
    ap.add_argument("--lora", action="store_true",
                    help="use LoRA instead of full fine-tuning (needed for 4B+)")
    args = ap.parse_args()

    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    from sentence_transformers.training_args import BatchSamplers

    # ---- model -------------------------------------------------------------- #
    model = SentenceTransformer(args.model, trust_remote_code=True)
    model.max_seq_length = args.max_seq

    # Optional LoRA path for bigger models that won't fully fit.
    if args.lora:
        from peft import LoraConfig
        peft_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        model.add_adapter(peft_cfg)
        print("LoRA adapters attached (only adapters will train).")

    # ---- data --------------------------------------------------------------- #
    train_ds = load_jsonl_dataset(args.train)
    eval_ds = load_jsonl_dataset(args.eval) if Path(args.eval).exists() else None
    print(f"train rows: {len(train_ds)}"
          + (f" | eval rows: {len(eval_ds)}" if eval_ds else ""))

    # ---- loss --------------------------------------------------------------- #
    loss = MultipleNegativesRankingLoss(model)

    # ---- training args ------------------------------------------------------ #
    targs = SentenceTransformerTrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        # memory savers for 12 GB cards:
        fp16=True,                         # half precision; use bf16=True on Ampere+
        gradient_checkpointing=True,       # trade compute for VRAM
        # MNRL needs no duplicate texts within a batch sharing a positive:
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch" if eval_ds else "no",
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to="none",                  # set to "wandb" if you use it
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
    )

    trainer.train()

    final = Path(args.out) / "final"
    model.save_pretrained(str(final))
    print(f"\nSaved fine-tuned model to {final}")
    print("Load it later with: SentenceTransformer(r'%s', trust_remote_code=True)"
          % final)


if __name__ == "__main__":
    main()
