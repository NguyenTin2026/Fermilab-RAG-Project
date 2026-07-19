#!/usr/bin/env python3
"""
eval_retrieval.py  --  Do chat luong RETRIEVAL cua hybrid retriever (BM25 + dense + RRF).

Tra loi cau hoi: "khi hoi 1 cau, retriever co keo dung tai lieu lien quan len top khong?"
Bao cao:  Recall@k  va  MRR@10.

CACH CHAM (relevance proxy):
  Vi du an CHUA co qrels thu cong (question -> gold doc id), script danh gia bang
  "keyword-grounded relevance": mot chunk duoc coi la LIEN QUAN neu text/title cua no
  chua it nhat 1 tu khoa trong truong "keywords" cua cau hoi gold. Day la PROXY hop ly
  khi chua co nhan tay -- muon nghiem tuc hon, hay them qrels that va sua ham `is_relevant`.

Usage (chay tren may co corpus + sentence-transformers, offline HF OK sau khi cache model):
    python eval/eval_retrieval.py \
        --corpus Fermilab-process_data_and_finetune_model/corpus_st7.jsonl \
        --gold   eval/gold_qa.jsonl \
        --k 1,3,5 --split test

LUU Y: can `git lfs pull` de co corpus_st7.jsonl that (khong phai con tro LFS).
"""
import argparse
import json
import sys
from pathlib import Path

# retriever.py nam trong thu muc con -> them vao sys.path de import duoc.
REPO_ROOT = Path(__file__).resolve().parent.parent
RETR_DIR = REPO_ROOT / "Fermilab-process_data_and_finetune_model"
sys.path.insert(0, str(RETR_DIR))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def is_relevant(chunk, keywords):
    """Chunk lien quan neu title/text chua BAT KY tu khoa nao (khong phan biet hoa/thuong)."""
    hay = (str(chunk.get("title", "")) + " " + str(chunk.get("text", ""))).lower()
    return any(kw.lower() in hay for kw in keywords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(RETR_DIR / "corpus_st7.jsonl"))
    ap.add_argument("--gold", default=str(REPO_ROOT / "eval" / "gold_qa.jsonl"))
    ap.add_argument("--k", default="1,3,5", help="danh sach k, vi du 1,3,5")
    ap.add_argument("--split", default=None, help="loc chunk theo split (vd 'test'); bo trong = tat ca")
    args = ap.parse_args()

    ks = sorted(int(x) for x in args.k.split(","))
    top_k = max(ks)

    from retriever import Retriever  # import sau khi da them sys.path

    gold = read_jsonl(args.gold)
    print(f"Loading retriever from {args.corpus} ...")
    ret = Retriever(args.corpus)

    hits_at = {k: 0 for k in ks}
    recip_ranks = []
    n = 0

    for q in gold:
        kws = q.get("keywords") or []
        if not kws:
            continue  # bo cau khong co keyword de cham
        n += 1
        results = ret.retrieve(q["question"], top_k=top_k, split=args.split)

        # Vi tri (1-indexed) cua chunk lien quan DAU TIEN trong top-k
        first_rel = None
        for rank, r in enumerate(results, start=1):
            if is_relevant(r, kws):
                first_rel = rank
                break

        if first_rel is not None:
            for k in ks:
                if first_rel <= k:
                    hits_at[k] += 1
            recip_ranks.append(1.0 / first_rel)
        else:
            recip_ranks.append(0.0)

    if n == 0:
        print("Khong co cau hoi nao co 'keywords' de cham. Them keywords vao gold_qa.jsonl.")
        return

    print(f"\n=== RETRIEVAL EVAL  (n={n} cau hoi, split={args.split or 'all'}) ===")
    for k in ks:
        print(f"  Recall@{k:<2d}: {hits_at[k] / n:.3f}")
    print(f"  MRR@{top_k:<3d}: {sum(recip_ranks) / n:.3f}")
    print("\n(Relevance = keyword-grounded proxy. Thay bang qrels tay de nghiem tuc hon.)")


if __name__ == "__main__":
    main()
