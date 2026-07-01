"""
discovery_ft.evaluate
=====================

Measure whether fine-tuning actually helped. This is the most important script
in the package and the one teams skip and regret skipping.

It builds a tiny retrieval benchmark from the held-out eval anchors: each anchor
has one known-correct document (its positive). We embed the whole corpus, embed
each anchor, rank documents by cosine similarity, and check where the correct
document landed. We report:

    Recall@k  -- fraction of anchors whose correct doc is in the top k
    MRR@10    -- mean reciprocal rank of the correct doc (1.0 = always first)

Run it TWICE -- once with the base model, once with your fine-tuned model -- and
compare. If the fine-tuned numbers aren't clearly higher, fine-tuning didn't
help and you should not ship it. (That is a real and common outcome with strong
modern base models; it's information, not failure.)

Usage:
    # baseline
    python -m discovery_ft.evaluate \
        --corpus dataset/corpus.jsonl --qrels dataset/eval_qrels.jsonl \
        --model Qwen/Qwen3-Embedding-0.6B --tag base

    # after training
    python -m discovery_ft.evaluate \
        --corpus dataset/corpus.jsonl --qrels dataset/eval_qrels.jsonl \
        --model runs/qwen06b-ft/final --tag finetuned
"""

from __future__ import annotations

import argparse
import json


Q_INSTR = ("Instruct: Given a research interest, retrieve relevant patents "
           "and inventions.\nQuery: ")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="dataset/corpus.jsonl")
    ap.add_argument("--qrels", default="dataset/eval_qrels.jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default="model", help="label for the printout")
    ap.add_argument("--ks", default="1,5,10")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",")]

    from sentence_transformers import SentenceTransformer
    import numpy as np

    corpus = _read_jsonl(args.corpus)
    qrels = _read_jsonl(args.qrels)
    doc_ids = [c["doc_id"] for c in corpus]
    doc_texts = [c["text"] for c in corpus]
    id_to_idx = {d: i for i, d in enumerate(doc_ids)}

    model = SentenceTransformer(args.model, trust_remote_code=True)

    doc_emb = model.encode(doc_texts, batch_size=16, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True)
    q_emb = model.encode([Q_INSTR + q["anchor"] for q in qrels], batch_size=16,
                         normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=True)

    sims = q_emb @ doc_emb.T
    ranks = []
    for row, q in zip(sims, qrels):
        gold = q["positive_id"]
        if gold not in id_to_idx:
            continue
        order = np.argsort(-row)
        # position (1-indexed) of the gold doc
        pos = int(np.where(order == id_to_idx[gold])[0][0]) + 1
        ranks.append(pos)

    n = len(ranks)
    if n == 0:
        print("No evaluable anchors (gold docs not in corpus?).")
        return

    print(f"\n=== {args.tag}  (model={args.model}) ===")
    print(f"eval anchors: {n}")
    for k in ks:
        rec = sum(1 for r in ranks if r <= k) / n
        print(f"  Recall@{k:<2d}: {rec:.3f}")
    mrr10 = sum(1.0 / r for r in ranks if r <= 10) / n
    print(f"  MRR@10  : {mrr10:.3f}")
    print(f"  median rank of correct doc: {int(np.median(ranks))}")


if __name__ == "__main__":
    main()
