#!/usr/bin/env python3
"""
02_generate_qa.py  --  DSSI Fermilab LLM Project, Stage 2
Generate grounded Q&A training pairs from TRAIN-split corpus chunks,
using Gemini as the teacher (distillation).

Usage:
    pip install google-genai
    export GEMINI_API_KEY=...        # from Google AI Studio
    python 02_generate_qa.py --corpus corpus.jsonl --out train_qa.jsonl
    python 02_generate_qa.py --selftest          # runs filter/decontam tests, no API

Leakage guards built in:
  * refuses to read any chunk whose split != "train"
  * optional --gold gold_questions.jsonl -> drops generated questions that
    share an 8-gram with any gold question (n-gram decontamination)
Resume-safe: already-processed chunk ids in --out are skipped.
"""

import argparse, hashlib, json, os, re, sys, time
from pathlib import Path
from google import genai

MODEL = "gemini-2.5-flash"   # teacher; flash is cheap+fast, use pro for harder chunks

SYSTEM_RULES = """You write training data for a question-answering model about Fermilab.
You will be given one passage. Produce question-answer pairs that satisfy ALL rules:
1. Every answer must be fully supported by the passage. Do not add outside facts.
2. Questions must be SELF-CONTAINED: name the experiment/topic explicitly
   (e.g. "What does the DUNE experiment study?"), never "this paper",
   "the passage", "the authors", or "the text".
3. Answers are 1-4 sentences, direct, in plain language a physics student understands.
4. Vary the types: factual lookup, definition, explanation/why, comparison.
5. If the passage is too thin/numeric to support a question type, skip that type.
Return ONLY a JSON array: [{"question": "...", "answer": "...", "type": "..."}]"""


def n_pairs_for(words: int) -> int:
    return 2 if words < 200 else 3 if words < 350 else 4


def build_prompt(chunk: dict) -> str:
    return (f"{SYSTEM_RULES}\n\nPassage (from {chunk['title'][:90]}):\n"
            f"\"\"\"\n{chunk['text']}\n\"\"\"\n\n"
            f"Produce exactly {n_pairs_for(chunk['n_words'])} pairs.")


# ----------------------- quality filters ------------------------------
BAD_PHRASES = re.compile(
    r"\b(the (passage|text|article|paper|document|excerpt)|according to the "
    r"(passage|text)|as (mentioned|stated) (above|in the))\b", re.I)

def pair_ok(q: str, a: str) -> bool:
    if not q or not a:                          return False
    if len(q.split()) < 5 or len(a.split()) < 3: return False
    if BAD_PHRASES.search(q) or BAD_PHRASES.search(a): return False
    if not q.strip().endswith("?"):             return False
    return True


def ngrams(text: str, n: int = 8):
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def load_gold_ngrams(path):
    grams = set()
    if path and Path(path).exists():
        for line in open(path):
            grams |= ngrams(json.loads(line)["question"])
    return grams


def contaminated(question: str, gold_grams: set) -> bool:
    return bool(gold_grams) and not ngrams(question).isdisjoint(gold_grams)


# ----------------------- teacher call ---------------------------------
def call_teacher(client, prompt: str):
    """One generation call; returns parsed list or raises."""
    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={"response_mime_type": "application/json", "temperature": 0.8},
    )
    return json.loads(resp.text)


def generate(args):
    from google import genai                                    # lazy import
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    chunks = [json.loads(l) for l in open(args.corpus)]
    train  = [c for c in chunks if c["split"] == "train"]       # GUARD
    assert all(c["split"] == "train" for c in train)
    print(f"{len(train)} train chunks ({len(chunks)-len(train)} test chunks untouched)")

    done = set()
    if Path(args.out).exists():                                 # resume
        done = {json.loads(l)["meta"]["chunk_id"] for l in open(args.out)}
        print(f"resuming: {len(done)} chunks already processed")

    gold_grams = load_gold_ngrams(args.gold)
    seen_q, kept, dropped = set(), 0, 0

    with open(args.out, "a") as f:
        for i, c in enumerate(train):
            if c["id"] in done:
                continue
            try:
                pairs = call_teacher(client, build_prompt(c))
            except Exception as e:                              # one retry
                print(f"  retry {c['id']}: {e}", file=sys.stderr)
                time.sleep(5)
                try:
                    pairs = call_teacher(client, build_prompt(c))
                except Exception as e2:
                    print(f"  SKIP {c['id']}: {e2}", file=sys.stderr)
                    continue

            for p in pairs if isinstance(pairs, list) else []:
                q, a = p.get("question", "").strip(), p.get("answer", "").strip()
                qh = hashlib.sha256(q.lower().encode()).hexdigest()
                if not pair_ok(q, a) or qh in seen_q or contaminated(q, gold_grams):
                    dropped += 1; continue
                seen_q.add(qh)
                f.write(json.dumps({
                    "messages": [
                        {"role": "system",
                         "content": "You are a helpful assistant that answers questions about Fermilab research."},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "meta": {"chunk_id": c["id"], "source": c["source"],
                             "type": p.get("type", ""), "origin": c.get("origin")},
                }, ensure_ascii=False) + "\n")
                kept += 1
            f.flush()
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(train)} chunks | kept {kept} | dropped {dropped}")
            time.sleep(args.sleep)                              # stay under rate limits

    print(f"done: kept {kept} pairs, dropped {dropped} -> {args.out}")


# ----------------------- offline self-test ----------------------------
def selftest():
    assert pair_ok("What does the DUNE experiment study?", "It studies neutrino oscillations.")
    assert not pair_ok("What does the passage say about DUNE?", "Things.")      # bad phrase
    assert not pair_ok("Why", "Because reasons exist here.")                    # too short
    g = ngrams("what is the flavor asymmetry of the proton light quark sea measured by seaquest")
    assert contaminated("Explain the flavor asymmetry of the proton light quark sea measured by SeaQuest.", g)
    assert not contaminated("What is a neutrino?", g)
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus.jsonl")
    ap.add_argument("--out", default="train_qa.jsonl")
    ap.add_argument("--gold", default=None, help="gold_questions.jsonl for decontamination")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    selftest() if args.selftest else generate(args)
