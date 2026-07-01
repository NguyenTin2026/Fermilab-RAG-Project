# Assignment 9 — Process Data & Fine-Tune an LLM (Fermilab Project)

A complete, copy-paste guide. Anyone with the server login can reproduce every
stage. *Hướng dẫn đầy đủ — ai có tài khoản server cũng chạy lại được.*

---

## 1. What this assignment does

Build a 3-stage pipeline that turns crawled Fermilab web pages into a fine-tuned
question-answering model:

```
crawled web pages (.txt)
        │
        ▼
[Stage 1] process  ─►  corpus_st7.jsonl     (clean, chunked, train/test split)
        │
        ▼
[Stage 2] generate ─►  train_qa.jsonl       (Gemini writes Q&A from TRAIN chunks)
        │
        ▼
[Stage 3] fine-tune ─► ft-qwen3-fermilab/   (QLoRA adapter on Qwen3-4B)
```

Key ideas:
- **RAG vs fine-tuning are different.** This assignment is the *fine-tuning*
  (task/style adaptation) half. Factual grounding is RAG's job in later stages.
- **No evaluation leakage.** The train/test split is decided in Stage 1, before
  any Q&A is generated. Q&A is generated only from TRAIN chunks.

---

## 2. Prerequisites

- SSH access to the server (`st7@aiserver`), with the project at
  `~/.vscode-server/Fermilab_Project/`.
- Python 3.12 (already on the server).
- A **GPU** with ≥12 GB VRAM for Stage 3 (`nvidia-smi` to check).
- A free **Gemini API key** from <https://aistudio.google.com> ("Get API key").

All commands below are run from the materials folder:

```bash
cd ~/.vscode-server/Fermilab_Project/Fermilab-process_data_and_finetune_model
```

---

## 3. Input data

Your crawl lives in `../crawled_data/`:

| Path | What |
|---|---|
| `text_pages/*.txt` | 2,233 cleaned fnal.gov pages (the input we use) |
| `html_pages/*.html` | raw HTML (not needed for this assignment) |
| `download_report_1000.csv` | maps each file → real URL + page title (provenance) |

---

## 4. Stage 1 — Process data

The provided `01_process_corpus.py` only reads **PDFs**. Our data is **text**,
so we use the text-ingest variant `01b_process_text_corpus.py` (same cleaning,
chunking, dedup, and leakage-safe split, but for `.txt`).

```bash
python3 01b_process_text_corpus.py \
    --text_dir ../crawled_data/text_pages \
    --report   ../crawled_data/download_report_1000.csv \
    --out      corpus_st7.jsonl
```

**Expected output (screenshot this):**

```
Found 2233 .txt files ...
2233 usable docs; 68 boilerplate lines removed
chunks written        : 8847
  train chunks        : 7100
  test  chunks        : 1747
Wrote 8847 chunks -> corpus_st7.jsonl
```

Each line of `corpus_st7.jsonl` carries provenance:
`id, source, source_type, title, origin (url), split, chunk_index, n_words,
content_hash, tags, text`.

---

## 5. Stage 2 — Generate Q&A with Gemini

```bash
# install the client (server is "externally managed", so use this flag)
pip install --break-system-packages google-genai

# offline check — no API used (screenshot the "selftest OK")
python3 02_generate_qa.py --selftest

# real run — set YOUR key first
export GEMINI_API_KEY=AIza...your_key
python3 02_generate_qa.py --corpus corpus_st7.jsonl --out train_qa.jsonl
```

It prints progress like `10/7100 chunks | kept 28 | dropped 3`. It is
**resume-safe**: press **Ctrl+C** after a few hundred pairs; `train_qa.jsonl`
keeps what was generated. **Screenshot** the running progress.

Quick check:
```bash
wc -l train_qa.jsonl
head -1 train_qa.jsonl
```

---

## 6. Stage 3 — Fine-tune (QLoRA)

```bash
# heavy install (~2-3 min)
pip install --break-system-packages torch transformers trl peft bitsandbytes accelerate datasets

# confirm GPU + data
nvidia-smi
wc -l train_qa.jsonl

# fine-tune (downloads Qwen3-4B the first time, then trains)
python3 03_finetune_qlora.py --data train_qa.jsonl --out ./ft-qwen3-fermilab
```

**Expected output (screenshot this — the most important proof):**

```
train pairs: 229  val pairs: 12
{'loss': '2.748', ... 'epoch': '0.35'}
{'loss': '1.535', ... 'epoch': '0.70'}
...
{'loss': '1.101', ... 'epoch': '2'}
adapter saved to ./ft-qwen3-fermilab
```

The falling loss = the model is learning from the Fermilab Q&A.

---

## 7. Results from the reference run

| Stage | Output | Key numbers |
|---|---|---|
| 1 Process | `corpus_st7.jsonl` | 2,233 pages → 8,847 chunks (7,100 train / 1,747 test) |
| 2 Q&A | `train_qa.jsonl` | ~241 Q&A pairs (selftest passed; leakage guard on) |
| 3 Fine-tune | `ft-qwen3-fermilab/` | loss 2.75 → 1.10, token acc 0.58 → 0.74, eval_loss ~1.34 (no overfit) |

---

## 8. Troubleshooting (real errors we hit)

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'fitz'` | ran `01_process_corpus.py` (PDF-only) on text data | use **`01b_process_text_corpus.py`** instead |
| `error: externally-managed-environment` | server blocks system pip | add **`--break-system-packages`** to `pip install` |
| `No module named 'google'` | `google-genai` not installed | `pip install --break-system-packages google-genai` |
| Stage 2 asks for a key / `KeyError: GEMINI_API_KEY` | key not set | `export GEMINI_API_KEY=...` before running |
| `--chat` → `AttributeError` at `model.generate` | script's chat demo is incompatible with `transformers 5.x` (passes the tokenizer dict, not `input_ids`) | optional; does **not** affect the fine-tune. Skip, or patch `ask()` to `model.generate(**ids, ...)` |

---

## 9. How to submit (Moodle)

1. Attach 3 screenshots: Stage 1 output, Stage 2 running, Stage 3 training loss.
2. Attach `Submission.md` (results write-up) and/or fill the
   `Fermilab_Processing_FineTuning_Worksheet.docx` if the professor requires it.
3. Confirm the code and outputs live on the server under
   `~/.vscode-server/Fermilab_Project/Fermilab-process_data_and_finetune_model/`.

---

## 10. Files

```
Fermilab-process_data_and_finetune_model/
├── 01_process_corpus.py          # provided (PDF input)
├── 01b_process_text_corpus.py    # text-input variant used here
├── 02_generate_qa.py             # Gemini Q&A generation
├── 03_finetune_qlora.py          # QLoRA fine-tune
├── corpus_st7.jsonl              # Stage 1 output
├── train_qa.jsonl                # Stage 2 output
├── ft-qwen3-fermilab/            # Stage 3 LoRA adapter
├── audit_report_st7.txt          # Stage 1 audit
├── Submission.md                 # results write-up
└── README_Assignment9.md         # this file
```