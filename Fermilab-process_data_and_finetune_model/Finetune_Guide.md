# Fermilab Q&A Assistant — RAG + QLoRA Fine-tuned LLM (Qwen3-4B)

End-to-end pipeline for building a question-answering model about Fermilab research:
raw corpus → synthetic Q&A generation → QLoRA fine-tuning → evaluation → merged production model.

> **Run in order: Step 1 → 2 → 3 → 4 → 5.** Each step produces the input for the next.

---

## 0. Environment setup

Activate the `ndo1` virtual environment before doing anything:

```bash
cd ~/.vscode-server/Fermilab_Project/Fermilab-process_data_and_finetune_model
source ndo1/bin/activate      # the (ndo1) prefix in your prompt means it's active
```

Install the required libraries:

```bash
python3 -m pip install PyMuPDF        # for Step 1 (import fitz). NOTE: do NOT install the "fitz" package
python3 -m pip install google-genai   # for Step 2 (from google import genai)
# Step 3 also needs: torch, transformers, peft, trl, bitsandbytes, datasets, accelerate
```

Set the Gemini API key (get it from Google AI Studio — aistudio.google.com):

```bash
export GEMINI_API_KEY=AIza...your_key...
```

To avoid re-setting it on every new SSH session, append it to `~/.bashrc`:

```bash
echo 'export GEMINI_API_KEY=AIza...your_key...' >> ~/.bashrc
source ~/.bashrc
```

**Never commit the API key to GitHub.** Google automatically revokes leaked keys.

---

## Step 1 — Process the corpus

**File:** `01_process_corpus.py`
Reads raw documents (PDF, etc.) from `./data`, cleans them, splits them into chunks, assigns a train/test split → outputs `corpus.jsonl`.

```bash
python3 01_process_corpus.py --in_dir ./data --out corpus.jsonl
```

Verify the output:

```bash
wc -l corpus.jsonl        # number of chunks
head -n 1 corpus.jsonl    # inspect one chunk
```

---

## Step 2 — Generate Q&A data (distillation from Gemini)

**File:** `02_generate_qa.py`
Uses Gemini as the "teacher" to generate question-answer pairs from chunks in the `train` split. Built-in leakage guards (only reads `train` chunks, applies n-gram decontamination). **Resume-safe:** if interrupted, rerunning skips already-processed chunks.

Run the offline self-test first (verifies the filtering logic, no API cost):

```bash
python3 02_generate_qa.py --selftest      # prints "selftest OK" when healthy
```

Run for real:

```bash
python3 02_generate_qa.py --corpus corpus.jsonl --out train_qa.jsonl
```

Verify the output:

```bash
wc -l train_qa.jsonl        # number of Q&A pairs
head -n 1 train_qa.jsonl    # inspect one pair
```

→ Produces `train_qa.jsonl` (standard conversational JSONL format).

---

## Step 3 — Fine-tune with QLoRA (the heaviest step)

**File:** `03_finetune_qlora.py`
Fine-tunes the base model (default `Qwen/Qwen3-4B-Instruct-2507`, change it on line 22) using QLoRA:
- **4-bit NF4 (BitsAndBytes)** to minimize VRAM usage (requires a GPU with ≥ 12GB VRAM)
- **LoRA adapter (PEFT)** trains only a small set of added parameters attached to the Linear layers (`q_proj`, `v_proj`, ...)
- **SFTTrainer (TRL)** automatically applies the tokenizer's chat template

Check GPU availability first:

```bash
nvidia-smi
```

Run training (use `tmux` so it survives an SSH disconnect):

```bash
tmux new -s train
python3 03_finetune_qlora.py --data train_qa.jsonl --out ./ft-qwen3-fermilab
# Press Ctrl+B then D to detach; run "tmux attach -t train" to come back
```

**Monitor** `train_loss` and `eval_loss`:
- Both decreasing → good.
- `eval_loss` rising while `train_loss` keeps falling → **overfitting**: stop early at the previous checkpoint or reduce epochs to 1.

→ Adapter weights are saved in `./ft-qwen3-fermilab`.

---

## Step 4 — Quick before/after check (Sanity Chat)

```bash
python3 03_finetune_qlora.py --chat ./ft-qwen3-fermilab
```

Prints the base model's and fine-tuned model's answers side by side for visual comparison. If the fine-tuned model answers Fermilab questions more accurately → success.

---

## Step 5 — Merge weights (only when satisfied)

```bash
python3 03_finetune_qlora.py --merge ./ft-qwen3-fermilab
```

Merges the LoRA adapter back into the base model to produce a single, complete set of weights → saved to `./ft-qwen3-fermilab/merged`. Use it for high-speed deployment (vLLM, GGUF). **No need to merge while still experimenting.**

---

## Data flow summary

```
./data  →[Step 1]→  corpus.jsonl  →[Step 2]→  train_qa.jsonl
        →[Step 3]→  ./ft-qwen3-fermilab (adapter)
        →[Step 4]→  compare results
        →[Step 5]→  ./ft-qwen3-fermilab/merged (complete model)
```

---

## Troubleshooting — issues encountered & fixes

**`ModuleNotFoundError: No module named 'fitz'`** (Step 1)
→ Install `python3 -m pip install PyMuPDF`. Do not install the package named `fitz`.

**`ModuleNotFoundError: No module named 'google'`** (Step 2)
→ Install `python3 -m pip install google-genai`.

**`KeyError: 'GEMINI_API_KEY'`** (Step 2)
→ API key not set. Run `export GEMINI_API_KEY=...`.

**`AttributeError` / `KeyError: 'shape'` in the `ask` function** (Step 4)
The tokenizer returns a `BatchEncoding` (a "box" holding `input_ids`, `attention_mask`), not a tensor. You must "open the box" in two places inside `ask`:

```python
# generate line — add ** to spread input_ids/attention_mask as keyword args
out = model.generate(**ids, max_new_tokens=160, do_sample=False)

# decode line — read the prompt length from input_ids to strip the question part
return tok.decode(out[0][ids["input_ids"].shape[-1]:], skip_special_tokens=True)
```

**`FutureWarning` about `bitsandbytes`**
→ Just a warning, not an error. Ignore it.