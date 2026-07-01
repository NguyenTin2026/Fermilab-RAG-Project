# Discovery Hub — Processing Data & Fine-Tuning a Small Model

A step-by-step guide for st1, st2, and st5. By the end you will have: turned your
downloaded JSON into a clean, unified dataset; built training pairs even though
nobody labeled any; fine-tuned a small embedding model on the local GPU server;
and **measured whether the fine-tuning actually helped**.

Read this once top to bottom before running anything. The scripts referenced
live in the `discovery_ft/` package; every command is also wired into the
`Makefile`.

---

## The one idea that makes this project hard

We are matching two kinds of text that describe the same science in *different
languages*:

- **Document side** — patents (st5) and AUTM invention listings (st1). Written in
  legal / technical language: *"a genetically modified zebrafish stably
  expressing FLT3-ITD…"*
- **Query side** — research interests. A pharma scout writes in clinical /
  business language: *"in vivo model for screening leukemia drug candidates."*

A clinical trial (st2) reads like a research interest, so we treat trials as
**query-side** material. Everything we build below exists to bridge that
register gap. Keep it in mind — it explains why we generate queries the way we
do and why we evaluate the two sides separately.

---

## Step 0 — Set up (once)

On the local Ubuntu server (the one with the two 12 GB GPUs):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # installs torch, sentence-transformers, etc.
# Install the torch build that matches the server's CUDA driver if needed.
nvidia-smi                               # confirm both GPUs are visible
```

Put all three students' folders under one directory. The structure from the zip
is fine — the parser walks subfolders automatically:

```
data/
├── st1/autm_data/2026/*.json     # AUTM pages          -> document side
├── st2/*.json                    # ClinicalTrials.gov  -> query side
└── st5/*.json                    # USPTO patents       -> document side
```

---

## Step 1 — Look at what you have (no GPU, no API needed)

Before transforming anything, see that all three schemas parse:

```bash
make inspect            # == python -m discovery_ft.parse data
```

You'll get a count by source and by **side**, plus one example per source. This
is your first checkpoint: if a source shows 0 records, something's wrong with the
folder path or the JSON, and you fix it here before going further.

### What "processing" actually means

Each source has a totally different JSON shape. `discovery_ft/parse.py` converts
all of them into ONE record type with a single field that matters:
`embedding_text` — the exact string that becomes a vector. Nothing downstream
needs to know which source a record came from.

How each source is assembled into `embedding_text`:

| Source | What goes into `embedding_text` |
|---|---|
| **Patent (st5)** | title + abstract + **first 3 claims** + a plain-English CPC field gloss |
| **Trial (st2)** | brief title + brief summary + conditions + keywords + interventions |
| **AUTM (st1)** | title + excerpt + body (capped) |

Two cleaning steps matter for patents specifically, and the parser does them for
you:

1. **Boilerplate stripping.** Phrases like *"the present invention relates to"*
   and *"in one embodiment"* appear in nearly every patent and carry no
   discriminating signal. Left in, they make every patent look similar for the
   wrong reason. They're removed.
2. **Notation cleanup.** USPTO dumps superscripts as `.sup.` (so `FLT3-ITD`
   arrives as `FLT3.sup.ITD`). The parser collapses these so gene and variant
   names read normally — which is exactly the vocabulary a query needs to match.

Only the **first few claims** are kept. Independent claims come first and carry
the core invention; later dependent claims add narrow legal limitations and just
inflate length.

> **AUTM noise warning (st1):** not every AUTM page is an invention. Many are
> newsletters and press releases. Each record carries a `topic_score`; the
> pipeline drops low-scoring pages (`--min-autm-score`, default 5) before
> generating training data. Skim your AUTM folder and sanity-check that the
> high-score pages are the substantive ones.

---

## Step 2 — Manufacture training pairs (the key transformation)

**The problem:** fine-tuning a retriever needs `(query, matching-document)`
pairs. Nobody labeled any. We have documents but no queries.

**The fix (InPars / Promptagator recipe):** for each document, ask an LLM to
write the research-interest query a pharma scout would use to find it. This is
where your Phase-1 Gemini infrastructure earns its keep.

```bash
export GEMINI_API_KEY=...                # use YOUR OWN key, not the shared one
make pairs                               # 2 queries per document -> pairs.jsonl
```

The critical instruction in the prompt (in `generate_queries.py`): the LLM must
write in **business/clinical register and must NOT copy the patent's technical
phrases**. That deliberately recreates the cross-vocabulary gap our real task
has. If the synthetic query just echoes the document's jargon, the model learns
nothing useful from it.

Before spending API calls, dry-run to read the prompts:

```bash
python -m discovery_ft.generate_queries --in data --dry-run | head -40
```

Each output line in `pairs.jsonl` is one `(anchor, positive_id, positive)`
triple — the synthetic query, the document id it matches, and the document text.

> **Where to spend your effort:** the few-shot examples at the top of the prompt
> control the entire style of what you generate. Improving those 2–3 hand-written
> examples does more for quality than any other single change. Write a few real
> ones that sound like how an actual scout phrases an interest.

---

## Step 3 — Mine hard negatives & split (one command)

A retriever learns by **contrast** — it needs wrong documents to push away from.
Random wrong documents are useless (a battery patent is obviously unrelated to a
leukemia query). We want **hard negatives**: documents that look similar to the
right one but are actually wrong.

```bash
make dataset             # mines negatives with the base model, writes splits
```

What `build_dataset.py` does:

1. Embeds every document once with the base model.
2. For each synthetic query, retrieves the most-similar documents and takes the
   near-misses (skipping the labeled positive) as hard negatives.
3. Skips the very top 1–2 hits — they might be *true* matches we just don't have
   labels for (false negatives), and training on those as "wrong" hurts.
4. Splits into `train.jsonl` / `eval.jsonl` **by anchor**, so the same query
   never appears in both train and eval.

Outputs (in `dataset/`):

| File | Contents |
|---|---|
| `train.jsonl` | `(anchor, positive, negative)` triplets for training |
| `eval.jsonl` | held-out triplets |
| `corpus.jsonl` | every document — used to build the eval search index |
| `eval_qrels.jsonl` | the correct doc id for each eval query — used for scoring |

---

## Step 4 — Measure the BASELINE *before* you train

This is the step people skip and regret. You cannot tell whether fine-tuning
helped without a before number.

```bash
make baseline            # evaluates the untrained base model
```

You'll see `Recall@1/5/10` and `MRR@10` on the held-out queries. **Write these
numbers down.** They are the bar fine-tuning has to clear.

> Be ready for the honest outcome: modern embedding models are strong
> out-of-the-box, and on a corpus this size fine-tuning sometimes barely moves
> the baseline. That is *information*, not failure — it tells you to invest in
> more/better data rather than more training.

---

## Step 5 — Fine-tune the small model

We fine-tune **Qwen3-Embedding-0.6B**. Why this one:

- **Small enough to fully fine-tune on a single 12 GB GPU** — fast iteration.
- **Instruction-aware** — built for the query/document asymmetry this project is
  about.
- **Apache-2.0** — no licensing headache if you demo it to companies.

```bash
make train               # full fine-tune, 3 epochs, batch 16
```

The loss is **MultipleNegativesRankingLoss (MNRL)**: it pulls each query toward
its correct document and pushes it away from its own hard negative *and* every
other document in the batch. Bigger batches give more of these "in-batch
negatives," so push the batch size as high as 12 GB allows.

**Memory knobs** (in `train.py` / overridable on the command line):

- Hitting out-of-memory? Lower `--batch` to 8 or 4, and raise `--grad-accum` to
  keep the effective batch size up without using more VRAM.
- `gradient_checkpointing` and `fp16` are already on — they trade a little speed
  for a lot of headroom. (On these GPUs, try `bf16` instead of `fp16` if
  supported; it's more numerically stable.)
- `--max-seq 512` keeps memory in check and fits patents comfortably. Only raise
  it toward 1024+ when you move to **Anvil** (see below).

### Using both GPUs

```bash
torchrun --nproc_per_node=2 -m discovery_ft.train \
    --train dataset/train.jsonl --eval dataset/eval.jsonl \
    --model Qwen/Qwen3-Embedding-0.6B --out runs/qwen06b-ft --batch 16
```

### Bigger model (4B) — LoRA path

The 4B model won't fully fine-tune on 12 GB. Use LoRA (trains small adapters,
not the whole model):

```bash
python -m discovery_ft.train --model Qwen/Qwen3-Embedding-4B --lora \
    --batch 4 --grad-accum 4 --max-seq 512 --out runs/qwen4b-lora
```

For full-text (long claims) fine-tuning of the 4B/8B model, that's an **Anvil**
job — request a GPU node, raise `--max-seq`, and copy the resulting
`runs/.../final` checkpoint back to the local server for inference. Nothing in
the serving code changes.

---

## Step 6 — Did it work?

```bash
make eval                # evaluates the fine-tuned model the same way
```

Put the two printouts side by side:

```
=== base       ===                  === finetuned  ===
  Recall@1 : 0.41                      Recall@1 : 0.55
  Recall@5 : 0.68                      Recall@5 : 0.79
  MRR@10   : 0.52                      MRR@10   : 0.64
```

If `finetuned` is clearly above `base`, ship it. If it's flat or worse, **don't
ship it** — instead generate more/better synthetic pairs (Step 2), improve the
few-shot prompt, or add more documents, then retry.

---

## The whole thing in one shot

Once you've eyeballed each step at least once:

```bash
make all       # pairs -> dataset -> baseline -> train -> eval
```

---

## What to report back to the team

1. The baseline vs. fine-tuned table from Steps 4 and 6.
2. How many pairs you generated and a handful of example synthetic queries
   (paste 5–6 so we can judge register quality).
3. Anything weird in your source's data (st1: how many AUTM pages were real
   inventions vs. noise? st5: any patents where claim parsing looked off? st2:
   any trials that don't read like a research interest?).

---

## Two things NOT to do yet

- **Don't reach for the cross-encoder / GPL soft-label recipe** on the first
  pass. MNRL is the right starting point. Add MarginMSE distillation only if MNRL
  plateaus — it's a refinement, not a starting point.
- **Don't fine-tune the reranker at the same time.** Get retrieval (Recall@k)
  right first; the reranker can only re-order what retrieval already found.

---

## File map

```
discovery_finetune/
├── README.md                    # this guide
├── requirements.txt
├── Makefile                     # every step wired up
└── discovery_ft/
    ├── parse.py                 # Step 1: all 3 schemas -> unified embedding_text
    ├── generate_queries.py      # Step 2: synthetic query/doc pairs via Gemini
    ├── build_dataset.py         # Step 3: hard-negative mining + train/eval split
    ├── train.py                 # Step 5: fine-tune Qwen3-Embedding-0.6B
    └── evaluate.py              # Steps 4 & 6: Recall@k / MRR, base vs fine-tuned
```
