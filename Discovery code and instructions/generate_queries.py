"""
discovery_ft.generate_queries
=============================

Manufacture the training signal we don't have.

We have documents (patents, AUTM listings) but NO labeled "this research interest
matches this patent" pairs. Fine-tuning a retriever needs pairs. So for each
document we ask an LLM to write the research-interest query a pharma tech-scout
would type to find it. This is the InPars / Promptagator recipe: use a strong
LLM to generate synthetic queries for unlabeled passages, then train on the
resulting (query, document) pairs.

The single most important trick for THIS project: we instruct the LLM to write
in the *business / clinical register*, deliberately NOT reusing the patent's
legal/technical phrasing. That manufactures the exact cross-vocabulary gap our
real task has (a scout writes "treatment for AML", the patent says "genetically
modified zebrafish expressing FLT3-ITD"). If the synthetic query just echoes the
patent's words, the model learns nothing useful.

Output: a JSONL of pairs, one per line:
    {"anchor": "<synthetic research interest>",
     "positive_id": "<doc_id>",
     "positive": "<document embedding_text>",
     "source": "uspto_patent"}

This file uses the Gemini API (the project's Phase-1 infra). The call is isolated
in `generate_one()` so you can swap in a local Qwen model later without touching
the rest of the pipeline.

Usage:
    export GEMINI_API_KEY=...        # students use their own key
    python -m discovery_ft.generate_queries \
        --in data --out pairs.jsonl --n-per-doc 2

Add --dry-run to print prompts without calling the API (free, good for testing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .parse import load_folder, DiscoveryDoc


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
# Few-shot examples teach the register we want. These are hand-written and are
# the highest-leverage thing in this file -- spend effort here. Each shows a
# technical document followed by the kind of short, business-language query a
# pharma scout would actually use. Note the queries never copy the document's
# jargon.
_FEWSHOT = """\
Document: Genetically modified zebrafish stably expressing combinations of human \
AML driver mutations (FLT3-ITD, NPMc+, DNMT3A-R882) used as an in vivo platform \
for high-throughput screening of candidate anti-leukemia compounds.
Queries:
- in vivo model for screening acute myeloid leukemia drug candidates
- animal platform to test personalized therapies for AML mutation subtypes

Document: A monoclonal antibody that binds PD-L1 with high affinity, formulated \
for intravenous administration, shown to restore T-cell activity in solid tumor \
models.
Queries:
- checkpoint inhibitor antibody targeting PD-L1 for solid tumors
- immunotherapy to reactivate exhausted T cells in cancer patients
"""

_PROMPT_TEMPLATE = """\
You are helping build a search system that connects pharmaceutical companies to \
university inventions and patents. Pharma scouts search using short, plain \
business or clinical language describing a research interest or unmet need. They \
do NOT use the legal or highly technical wording found in patents.

Given the document below, write {n} DISTINCT search queries a pharma scout might \
type to find it. Rules:
- Use everyday clinical/business register, not patent or legalese phrasing.
- Do NOT copy distinctive technical phrases, gene symbols, or chemical names \
verbatim from the document; describe the underlying purpose instead.
- Each query is one line, 6-16 words, no numbering, no quotes.
- Vary angle across queries (e.g. mechanism vs. disease vs. application).

Here are examples of the style we want:
{fewshot}

Now do the same for this document.
Document: {doc}
Queries:"""


def build_prompt(doc: DiscoveryDoc, n: int) -> str:
    # Cap document length in the prompt to control token cost.
    body = doc.embedding_text[:1200]
    return _PROMPT_TEMPLATE.format(n=n, fewshot=_FEWSHOT, doc=body)


# --------------------------------------------------------------------------- #
# LLM call -- swap this one function to change provider
# --------------------------------------------------------------------------- #
def generate_one(prompt: str, model: str = "gemini-2.0-flash") -> list[str]:
    """
    Call Gemini and return a list of query strings. Isolated so the rest of the
    pipeline is provider-agnostic. To use a local model instead, replace the body
    with a call to your vLLM/Ollama endpoint and keep the return shape.
    """
    from google import genai  # pip install google-genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    resp = client.models.generate_content(model=model, contents=prompt)
    text = resp.text or ""
    return _parse_lines(text)


def _parse_lines(text: str) -> list[str]:
    """Pull clean query lines out of the model's response."""
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        # Drop empties, headers, and anything suspiciously long/short.
        if not line or line.lower().startswith(("queries", "document")):
            continue
        if 3 <= len(line.split()) <= 30:
            out.append(line.strip('"').strip())
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data", help="folder of raw json")
    ap.add_argument("--out", default="pairs.jsonl", help="output JSONL of pairs")
    ap.add_argument("--n-per-doc", type=int, default=2)
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--sides", default="document",
                    help="comma list of sides to generate FROM (usually 'document')")
    ap.add_argument("--min-autm-score", type=int, default=5,
                    help="drop AUTM pages below this topic_score (noise filter)")
    ap.add_argument("--sleep", type=float, default=0.5, help="seconds between calls")
    ap.add_argument("--dry-run", action="store_true",
                    help="print prompts, do not call the API")
    args = ap.parse_args()

    wanted_sides = set(args.sides.split(","))
    docs = []
    for d in load_folder(args.inp):
        if d.side not in wanted_sides:
            continue
        # Noise filter: skip low-signal AUTM newsletters/press releases.
        if d.source == "autm" and d.extra.get("topic_score", 0) < args.min_autm_score:
            continue
        docs.append(d)

    print(f"Generating queries for {len(docs)} documents "
          f"({args.n_per_doc} each) -> {args.out}", file=sys.stderr)

    n_pairs = 0
    with open(args.out, "w", encoding="utf-8") as fout:
        for i, doc in enumerate(docs, 1):
            prompt = build_prompt(doc, args.n_per_doc)
            if args.dry_run:
                print(f"\n===== DRY RUN doc {i}: {doc.doc_id} =====")
                print(prompt[:900], "...")
                continue
            try:
                queries = generate_one(prompt, model=args.model)
            except Exception as e:  # keep going on transient errors
                print(f"  [warn] {doc.doc_id}: {e}", file=sys.stderr)
                time.sleep(2.0)
                continue
            for q in queries[: args.n_per_doc]:
                rec = {
                    "anchor": q,
                    "positive_id": doc.doc_id,
                    "positive": doc.embedding_text,
                    "source": doc.source,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_pairs += 1
            if i % 25 == 0:
                print(f"  ...{i}/{len(docs)} docs, {n_pairs} pairs", file=sys.stderr)
            time.sleep(args.sleep)

    if not args.dry_run:
        print(f"Done. Wrote {n_pairs} pairs to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
