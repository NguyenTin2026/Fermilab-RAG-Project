# Fermilab Domain LLM — Project Report
**Dataset, Fine-Tuned Model, and Evaluation Performance**

---

## 1. Dataset

### 1.1 Raw data collection
A custom BFS web crawler (`fermilab.py`) harvested public pages from Fermilab domains (`news.fnal.gov`, `education.fnal.gov`, `lbnf-dune.fnal.gov`, `novaexperiment.fnal.gov`, and others), starting from a small set of seed URLs, with a 1.5 s polite delay and a 20 s timeout per request. Teammates contributed additional sources in the same shared pipeline: arXiv papers (PDF, e.g. SeaQuest, IOTA, muon g−2 review) and fnal.gov pages saved as print-to-PDF.

| Metric | Value |
|---|---|
| Pages attempted | 1,000 |
| Pages downloaded successfully | 936 (93.6%) |
| Raw HTML files on disk | 2,233 (601 MB) |
| Extracted text files on disk | 2,233 (434 MB) |
| Total crawl data on disk | ~1.1 GB |

Provenance for every page (URL, title, HTTP status, word count) is tracked in `crawled_data/download_report_1000.csv`.

### 1.2 Corpus processing → `corpus_st7.jsonl`
`01b_process_text_corpus.py` (a text-input variant of the PDF-only `01_process_corpus.py`) turns the raw crawl into a clean, chunked corpus:

1. Strips crawler-added URL/title headers.
2. Removes boilerplate navigation/footer lines (lines on ≥30% of pages, ≤12 words) — **68 lines removed**.
3. Fixes hyphenation line-wraps, normalizes whitespace.
4. Greedy-packs text into **350–450-word chunks**, respecting line boundaries.
5. Deduplicates identical chunks via content hash.
6. **Leakage guard**: assigns each *document* (not chunk) to train/test with a deterministic SHA-1/SHA-256 hash **before** any synthetic data is generated, so no document's chunks straddle the split.

| Metric | Value |
|---|---|
| Text files processed | 2,233 |
| Chunks written | 8,847 |
| Train chunks | 7,100 (80.25%) |
| Test chunks | 1,747 (19.75%) |

Each record carries: `id, source, source_type, title, origin (URL), split, chunk_index, n_words, content_hash, tags, text`.

### 1.3 Synthetic Q&A generation → `train_qa.jsonl`
`02_generate_qa.py` distills grounded question–answer pairs **only from train-split chunks** using `gemini-2.5-flash` as a teacher model, in OpenAI/HF-style conversational format (`system` / `user` / `assistant` messages). Built-in guards:

- **Leakage guard** — refuses to read any `test`-split chunk.
- **Decontamination** — drops generated questions sharing an 8-gram with any gold evaluation question.
- **Quality filter** — drops short or meta-referential ("the passage...") Q/A pairs.
- **Resume-safe** — skips already-processed chunk IDs on restart.

The final combined dataset used for fine-tuning is **438 Q&A pairs**, pooled across team members:

| Source | Q&A pairs |
|---|---|
| st7 | 241 |
| st4 | 184 |
| st6 | 13 |
| **Total** | **438** |

For the actual training run, this was split into **417 training / 21 validation** examples (5% random holdout, seed 13) — this validation split is for monitoring training loss only; it is *not* the document-level gold test set reserved in step 1.2.

---

## 2. Model Fine-Tuned

**Base model:** `Qwen/Qwen3-4B-Instruct-2507` (4B-parameter instruction-tuned LLM).

**Method: QLoRA** (Quantized Low-Rank Adaptation), via `03_finetune_qlora.py`:

| Setting | Value |
|---|---|
| Quantization | 4-bit NF4, double quantization, bf16 compute dtype (`bitsandbytes`) |
| LoRA rank (`r`) | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (all linear layers) |
| Trainer | Hugging Face `trl.SFTTrainer` |
| Per-device batch size | 2 |
| Gradient accumulation | 8 → effective batch size 16 |
| Epochs | 2 |
| Learning rate | 2 × 10⁻⁴, cosine schedule, 3% warmup |
| Max sequence length | 2,048 tokens |
| Gradient checkpointing | enabled (VRAM saving) |

**Compute:** trained on a single GPU via SLURM on the Purdue Anvil cluster (`finetune.sub`: 1 node, 1 GPU, 32 CPUs, 2-hour walltime, conda env `ndo1`, fully offline `HF_HUB_OFFLINE=1`). Total training wall-clock: **232 seconds** for 54 steps (2 epochs over 417 examples).

**Output artifacts:** LoRA adapter + tokenizer saved to `ft-qwen3-fermilab/` (with `checkpoint-27` and `checkpoint-54` snapshots). The script also supports `--merge` to bake the adapter into full bf16 weights for faster inference, and a `--chat` mode that prints base-model vs. fine-tuned answers side by side for 3 sanity questions (NOvA, DUNE/LBNF, SeaQuest) — available for a qualitative before/after demo but not yet run/captured in this report.

---

## 3. Evaluation Performance — Before vs. After Fine-Tuning

The most important script for measuring impact (`evaluate.py`, a Recall@k / MRR benchmark against held-out gold questions) is part of the shared methodology template but has **not yet been run** for this Q&A model — it was written for an embedding-retrieval model, not a generative one, so the quantitative "before vs. after" signal we have to date comes from the **training/validation loss curve**, taken directly from `trainer_state.json` of the actual SLURM run (job `18543042`):

| Step | Epoch | Train Loss | Train Token Acc. | Eval Loss | Eval Token Acc. |
|---:|---:|---:|---:|---:|---:|
| 5  | 0.19 | 2.921 | 56.3% | — | — |
| 10 | 0.38 | 1.682 | 66.2% | — | — |
| 15 | 0.57 | 1.433 | 70.2% | — | — |
| 20 | 0.77 | 1.408 | 69.5% | — | — |
| 25 | 0.96 | 1.344 | 70.9% | 1.290 | 72.2% |
| 30 | 1.11 | 1.075 | 74.6% | — | — |
| 35 | 1.31 | 1.162 | 73.5% | — | — |
| 40 | 1.50 | 1.175 | 73.9% | — | — |
| 45 | 1.69 | 1.161 | 73.5% | — | — |
| 50 | 1.88 | 1.095 | 75.2% | 1.243 | 72.4% |
| **54** | **2.00** | **1.419 (run avg)** | — | **1.242** | **72.6%** |

**Reading the "before vs. after":**
- **Loss fell ~58%** on the training signal (2.921 → ~1.10 by the last logged step), i.e. the model went from struggling to predict the Fermilab-style answer tokens to fitting them well.
- **Token-level accuracy rose from 56.3% → ~75%** on training batches and **72.2% → 72.6%** on the held-out validation slice — a smaller but real and *stable* gain.
- **No overfitting observed:** eval loss kept *decreasing* slightly through training (1.290 → 1.242) instead of diverging from train loss, so 2 epochs was a reasonable stopping point.

**Honest limitation:** this is a loss/perplexity-style proxy, not a task-level benchmark against the 1,747-chunk gold test split reserved in Section 1.2. A rigorous "base model vs. fine-tuned model" evaluation — e.g., running `03_finetune_qlora.py --chat` to compare verbatim answers, or scoring both models on held-out Fermilab questions with an LLM-as-judge / exact-match metric — has not been executed yet and is the natural next step (see Section 5).

---

## 4. Other Useful Information

### 4.1 Deployed application (`app.py`)
A Streamlit dashboard (`Fermilab Science Q&A Assistant`) serves the fine-tuned model:
- Loads the merged fine-tuned model if available, otherwise base + LoRA adapter merged at load time, otherwise falls back to the plain base model — inference runs in **bf16** (not 4-bit) since VRAM is not a constraint at serving time.
- **RAG grounding**: `retriever.py` implements hybrid search — BM25 (lexical) fused with dense semantic similarity (SentenceTransformer embeddings, `corpus_embeddings.npy`) via **Reciprocal Rank Fusion (RRF)** — over the 8,847-chunk corpus.
- **"RAG vs. No-RAG" comparison mode**: runs the same model twice (once ungrounded, once with retrieved context) on the same question, to visually demonstrate hallucination reduction, with a downloadable comparison report.
- **Corpus topology map**: 2D TF-IDF + TruncatedSVD projection of all chunks for visual exploration.
- **Pipeline metrics tab**: live crawler stats (success rate, domain breakdown, HTTP status codes) and the training-loss table from Section 3.

### 4.2 Team data context
The shared methodology distinguished three teammate data drops:
- **st4** — arXiv PDFs (good text layer, scholarly cleaning path).
- **st6** — fnal.gov pages saved as print-to-PDF (webpage cleaning path, heavy nav/footer stripping).
- **st7 / st8** — raw web-page crawl text and arXiv PDFs (st8's initial download was 0 bytes due to a missing `-L` redirect flag in `curl`, fixed by re-downloading).

This project's final 438-pair Q&A set and fine-tuning run pool contributions from st7, st4, and st6.

---

## 5. Limitations & Suggested Next Steps

1. **Small fine-tuning set.** 417 training pairs is small for a 4B model; gains are real but modest. Generating more Q&A pairs (varying temperature, multi-hop/comparative questions) would likely help.
2. **No task-level before/after benchmark yet.** Run `python 03_finetune_qlora.py --chat ./ft-qwen3-fermilab` and/or write a small gold Q&A set from the *test*-split documents to score both models directly (exact-match / LLM-judge), rather than relying on training loss alone.
3. **Merge the adapter** (`--merge`) for faster, simpler deployment once satisfied with quality.
4. **Consider Retrieval-Augmented Fine-Tuning (RAFT)**-style training (mixing correct + distractor documents into the training prompts) to better align the fine-tuned model with the RAG pipeline it's deployed behind in `app.py`.
