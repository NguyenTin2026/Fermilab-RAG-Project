#!/usr/bin/env python3
"""
03_finetune_qlora.py  --  DSSI Fermilab LLM Project, Stage 3
QLoRA supervised fine-tuning of a small open model on the synthetic Q&A set.

Run this on the GPU machine (>= ~12 GB VRAM for the 4B model):
    pip install torch transformers trl peft bitsandbytes accelerate datasets
    python 03_finetune_qlora.py --data train_qa.jsonl --out ./ft-qwen3-fermilab

After training:
    python 03_finetune_qlora.py --chat ./ft-qwen3-fermilab     # quick before/after test
    python 03_finetune_qlora.py --merge ./ft-qwen3-fermilab    # bake adapter into full weights

Hyperparameters follow the project methodology doc:
QLoRA 4-bit (nf4) | r=32, alpha=64, dropout 0.05, all linear layers
lr 2e-4 cosine + 3% warmup | effective batch 16 | 2 epochs | max_len 1024
"""

import argparse, json, random, shutil
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"   # swap for SmolLM3-3B for the second run


# ---------------------------------------------------------------- data
def load_messages(path, val_frac=0.05, seed=13):
    """Train/val split of the SFT pairs. NOTE: this val split is for watching
    training loss only -- it is NOT the gold test set, which never enters here."""
    rows = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(rows)
    n_val = max(8, int(len(rows) * val_frac))
    keep = lambda r: {"messages": r["messages"]}
    return [keep(r) for r in rows[n_val:]], [keep(r) for r in rows[:n_val]]


# --------------------------------------------------------------- train
def train(args):
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    train_rows, val_rows = load_messages(args.data)
    print(f"train pairs: {len(train_rows)}  val pairs: {len(val_rows)}")

    torch.cuda.set_device(0)
    bnb = BitsAndBytesConfig(                      # the "Q" in QLoRA
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map={"": 0},
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    peft_cfg = LoraConfig(                         # the "LoRA"
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,             # effective batch = 16
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=1024,
        packing=False,
        bf16=True,
        gradient_checkpointing=True,               # trade speed for VRAM
        logging_steps=5,
        # Bat eval dinh ky: truoc day eval_strategy="no" -> eval_dataset khong bao gio
        # duoc dung va loi khuyen "theo doi eval_loss" ben duoi khong the thuc hien.
        # Gio eval moi 25 step de ra eval_loss/eval_token_acc (khop bang trong report).
        eval_strategy="steps",
        eval_steps=25,
        per_device_eval_batch_size=1,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=Dataset.from_list(train_rows),   # "messages" field ->
        eval_dataset=Dataset.from_list(val_rows),      # chat template applied by TRL
        peft_config=peft_cfg,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.out)                       # saves the LoRA adapter
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")
    print("watch for: eval_loss rising while train loss falls = overfitting "
          "-> stop at the earlier checkpoint or cut to 1 epoch.")


# ------------------------------------------------- quick sanity chat
QUESTIONS = [
    "What does Fermilab's NOvA experiment study?",
    "What is the goal of the DUNE experiment at LBNF?",
    "What is Fermilab's primary mission as America's particle physics and accelerator laboratory?",
]

def chat(adapter_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    torch.cuda.set_device(0)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(adapter_dir)

    def ask(model, q):
        msgs = [{"role": "user", "content": q}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        out = model.generate(**ids, max_new_tokens=160, do_sample=False)
        return tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
              quantization_config=bnb, device_map={"": 0})
    print("=" * 30, "BASE MODEL", "=" * 30)
    for q in QUESTIONS:
        print(f"\nQ: {q}\nA: {ask(base, q)}")

    tuned = PeftModel.from_pretrained(base, adapter_dir)       # adapter on top
    print("\n" + "=" * 30, "FINE-TUNED", "=" * 30)
    for q in QUESTIONS:
        print(f"\nQ: {q}\nA: {ask(tuned, q)}")


# ------------------------------------------------------------- merge
def merge(adapter_dir):
    """Bake adapter into full bf16 weights (for vLLM / GGUF export)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
              torch_dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    out = Path(adapter_dir) / "merged"
    if out.exists():
        shutil.rmtree(out)
    merged.save_pretrained(out)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out)
    print(f"merged model -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train_qa.jsonl")
    ap.add_argument("--out", default="./ft-qwen3-fermilab")
    ap.add_argument("--chat", metavar="ADAPTER_DIR")
    ap.add_argument("--merge", metavar="ADAPTER_DIR")
    a = ap.parse_args()
    if a.chat:    chat(a.chat)
    elif a.merge: merge(a.merge)
    else:         train(a)
