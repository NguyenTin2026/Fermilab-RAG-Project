"""
discovery_ft.build_dataset
==========================

Turn (anchor, positive) pairs into (anchor, positive, NEGATIVE) triplets and
split into train / eval.

Why negatives matter: a retriever learns by contrast. If you only ever show it
correct pairs, it has nothing to push away from. RANDOM negatives are too easy --
a random patent about batteries is obviously unrelated to a leukemia query, so
the model learns nothing from it. We want HARD negatives: documents that look
similar to the positive but are actually wrong. We find them by embedding every
document with the base model and, for each anchor, retrieving the most similar
documents that are NOT the labeled positive. Those near-misses are the hard
negatives.

This is the same idea GPL uses (mine negatives with a dense retriever), minus
the cross-encoder soft-labelling step, which we add later only if simple
contrastive training plateaus.

Output: train.jsonl / eval.jsonl, each line:
    {"anchor": "...", "positive": "...", "negative": "..."}

Plus corpus.jsonl (every document, for building the eval index) and
eval_qrels.jsonl (which doc_id is correct for each eval anchor), used by
evaluate.py.

Usage:
    python -m discovery_ft.build_dataset \
        --pairs pairs.jsonl --in data \
        --base-model Qwen/Qwen3-Embedding-0.6B \
        --out-dir dataset --negs-per-pair 1 --eval-frac 0.2
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .parse import load_folder


def _load_pairs(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.jsonl")
    ap.add_argument("--in", dest="inp", default="data",
                    help="raw json folder, to rebuild the document corpus")
    ap.add_argument("--base-model", default="Qwen/Qwen3-Embedding-0.6B")
    ap.add_argument("--out-dir", default="dataset")
    ap.add_argument("--negs-per-pair", type=int, default=1)
    ap.add_argument("--top-k", type=int, default=20,
                    help="search depth for mining; negatives drawn from here")
    ap.add_argument("--skip-top", type=int, default=2,
                    help="skip the very top hits -- they may be true positives "
                         "we just don't have labels for (false negatives)")
    ap.add_argument("--eval-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Build the document corpus (document-side records only) ---------- #
    corpus = [d for d in load_folder(args.inp) if d.side == "document"]
    id2text = {d.doc_id: d.embedding_text for d in corpus}
    doc_ids = [d.doc_id for d in corpus]
    doc_texts = [d.embedding_text for d in corpus]
    print(f"Corpus: {len(corpus)} document-side records")

    with open(out / "corpus.jsonl", "w", encoding="utf-8") as f:
        for d in corpus:
            f.write(json.dumps(
                {"doc_id": d.doc_id, "text": d.embedding_text,
                 "source": d.source}, ensure_ascii=False) + "\n")

    # ---- 2. Embed the corpus once with the base model ----------------------- #
    # Imported here so `--help` works without torch installed.
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print(f"Loading base model {args.base_model} to mine hard negatives...")
    model = SentenceTransformer(args.base_model, trust_remote_code=True)

    # Qwen3-Embedding is instruction-aware: the DOCUMENT side takes no special
    # instruction, the QUERY side does. Match how we'll query at eval time.
    doc_emb = model.encode(
        doc_texts, batch_size=16, normalize_embeddings=True,
        show_progress_bar=True, convert_to_numpy=True,
    )

    pairs = _load_pairs(args.pairs)
    anchors = [p["anchor"] for p in pairs]
    q_instr = "Instruct: Given a research interest, retrieve relevant patents " \
              "and inventions.\nQuery: "
    anc_emb = model.encode(
        [q_instr + a for a in anchors], batch_size=16,
        normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True,
    )

    # ---- 3. For each pair, mine hard negatives ------------------------------ #
    # cosine sim = dot product (vectors are normalized).
    sims = anc_emb @ doc_emb.T                      # [n_pairs, n_docs]
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    triplets = []
    for p, row in zip(pairs, sims):
        pos_id = p["positive_id"]
        order = np.argsort(-row)                    # most similar first
        negs = []
        for rank, idx in enumerate(order):
            if rank < args.skip_top:                # skip possible false negs
                continue
            cand_id = doc_ids[idx]
            if cand_id == pos_id:
                continue
            negs.append(id2text[cand_id])
            if len(negs) >= args.negs_per_pair:
                break
            if rank >= args.top_k:
                break
        for neg in negs:
            triplets.append({
                "anchor": p["anchor"],
                "positive": p["positive"],
                "negative": neg,
                "positive_id": pos_id,
            })

    print(f"Built {len(triplets)} triplets from {len(pairs)} pairs")

    # ---- 4. Split train / eval BY ANCHOR ------------------------------------ #
    # Split on the anchor text so the same query never lands in both sets.
    uniq_anchors = sorted({t["anchor"] for t in triplets})
    random.shuffle(uniq_anchors)
    n_eval = max(1, int(len(uniq_anchors) * args.eval_frac))
    eval_anchors = set(uniq_anchors[:n_eval])

    train = [t for t in triplets if t["anchor"] not in eval_anchors]
    eval_ = [t for t in triplets if t["anchor"] in eval_anchors]

    def dump(rows, name):
        with open(out / name, "w", encoding="utf-8") as f:
            for t in rows:
                f.write(json.dumps(
                    {"anchor": t["anchor"], "positive": t["positive"],
                     "negative": t["negative"]}, ensure_ascii=False) + "\n")

    dump(train, "train.jsonl")
    dump(eval_, "eval.jsonl")

    # qrels: for each eval anchor, the correct doc_id (for ranking metrics)
    seen = set()
    with open(out / "eval_qrels.jsonl", "w", encoding="utf-8") as f:
        for t in eval_:
            key = (t["anchor"], t["positive_id"])
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(
                {"anchor": t["anchor"], "positive_id": t["positive_id"]},
                ensure_ascii=False) + "\n")

    print(f"train={len(train)}  eval={len(eval_)}  "
          f"(eval anchors={len(eval_anchors)})")
    print(f"Wrote {out}/train.jsonl, eval.jsonl, corpus.jsonl, eval_qrels.jsonl")


if __name__ == "__main__":
    main()
