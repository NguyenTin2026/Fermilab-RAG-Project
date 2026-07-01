#!/usr/bin/env python3
"""
01b_process_text_corpus.py  --  DSSI Fermilab LLM Project, Stage 1 (text variant)

The provided 01_process_corpus.py reads *PDFs* (PyMuPDF) from st4/st6/st8 folders.
st7's data is different: ~2233 already-extracted .txt pages crawled from fnal.gov
(in crawled_data/text_pages/). This script ingests THOSE into the SAME corpus
schema 02_generate_qa.py expects, so the rest of the pipeline runs unchanged.

What it does, in order (mirrors the official Stage-1 steps):
  1. Reads crawled_data/download_report_1000.csv for authoritative url + title
     (only rows with status == SUCCESS).
  2. Strips the 3-line file header (URL:/Title:/separator) the crawler added.
  3. Removes boilerplate: nav/footer lines that repeat across many pages.
  4. Fixes hyphenation + whitespace.
  5. Chunks into ~350-450 word passages on line boundaries.
  6. Deduplicates chunks by content hash.
  7. Assigns a DETERMINISTIC doc-level train/test split BEFORE any Q&A
     generation -- this is the leakage guard (a whole page is train OR test,
     never split across both).
  8. Writes one JSON object per chunk to the output JSONL + an audit report.

Pure standard library -- no pip install needed.

Usage (run from the 'Fermilab - process data and finetune model' folder):
    python3 01b_process_text_corpus.py \
        --text_dir ../crawled_data/text_pages \
        --report   ../crawled_data/download_report_1000.csv \
        --out      corpus_st7.jsonl
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

SOURCE = "st7"
SOURCE_TYPE = "webpage_text"
TEST_PCT = 15            # ~15% of documents held out for test
MIN_CHUNK_WORDS = 50     # drop tiny junk chunks
CHUNK_LO, CHUNK_HI = 350, 450
BOILERPLATE_DOC_FRAC = 0.30   # a line on >=30% of pages is nav/footer -> drop

_WS = re.compile(r"[ \t]+")
_HYPHEN = re.compile(r"(\w)-\n(\w)")       # join words split across a line break
_SEP = re.compile(r"^[─-╿\-_=]{6,}$")  # the box-drawing separator line


def load_report(report_path: Path) -> dict[str, dict]:
    """Map txt_filename -> {url, title, status} from the crawl report CSV."""
    meta: dict[str, dict] = {}
    if not report_path.exists():
        print(f"[warn] report not found: {report_path} (will fall back to filenames)",
              file=sys.stderr)
        return meta
    with open(report_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            fn = (row.get("txt_filename") or "").strip()
            if fn:
                meta[fn] = {
                    "url": (row.get("url") or "").strip(),
                    "title": (row.get("page_title") or "").strip(),
                    "status": (row.get("status") or "").strip().upper(),
                }
    return meta


def strip_header(lines: list[str]) -> list[str]:
    """Drop the leading 'URL:' / 'Title:' / separator header the crawler wrote."""
    out, body_started = [], False
    for ln in lines:
        if not body_started:
            if ln.startswith(("URL:", "Title:")):
                continue
            if _SEP.match(ln.strip()):
                body_started = True
                continue
            # No recognizable header -> treat everything as body.
            body_started = True
        out.append(ln)
    return out


def clean_line(ln: str) -> str:
    return _WS.sub(" ", ln).strip()


def read_doc(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    raw = _HYPHEN.sub(r"\1\2", raw)
    lines = [clean_line(l) for l in raw.splitlines()]
    lines = strip_header(lines)
    return [l for l in lines if l]            # drop blank lines


def chunk_lines(lines: list[str]) -> list[str]:
    """Greedily pack lines into ~350-450 word chunks on line boundaries."""
    chunks, buf, words = [], [], 0
    for ln in lines:
        w = len(ln.split())
        if words + w > CHUNK_HI and words >= CHUNK_LO:
            chunks.append(" ".join(buf))
            buf, words = [], 0
        buf.append(ln)
        words += w
    if buf and words >= MIN_CHUNK_WORDS:
        chunks.append(" ".join(buf))
    return chunks


def doc_split(stem: str) -> str:
    h = int(hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8], 16)
    return "test" if (h % 100) < TEST_PCT else "train"


def reconstruct_url(stem: str) -> str:
    # "0003_education.fnal.gov_about" -> https://education.fnal.gov/about
    body = re.sub(r"^\d+_", "", stem)
    return "https://" + body.replace("_", "/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text_dir", default="../crawled_data/text_pages")
    ap.add_argument("--report", default="../crawled_data/download_report_1000.csv")
    ap.add_argument("--out", default="corpus_st7.jsonl")
    ap.add_argument("--audit", default="audit_report_st7.txt")
    args = ap.parse_args()

    text_dir = Path(args.text_dir)
    if not text_dir.is_dir():
        raise SystemExit(f"text_dir not found: {text_dir}")

    meta = load_report(Path(args.report))
    files = sorted(text_dir.glob("*.txt"))
    print(f"Found {len(files)} .txt files in {text_dir}", file=sys.stderr)

    # ---- pass 1: read all docs, count line frequency for boilerplate ---------
    docs: dict[str, list[str]] = {}
    line_doc_freq: Counter = Counter()
    skipped_empty = 0
    for p in files:
        fn = p.name
        info = meta.get(fn, {})
        if info and info.get("status") and info["status"] != "SUCCESS":
            continue
        lines = read_doc(p)
        if not lines:
            skipped_empty += 1
            continue
        docs[fn] = lines
        for ln in set(lines):                 # count each line once per doc
            line_doc_freq[ln] += 1

    n_docs = len(docs)
    cutoff = max(2, int(n_docs * BOILERPLATE_DOC_FRAC))
    boiler = {ln for ln, c in line_doc_freq.items()
              if c >= cutoff and len(ln.split()) <= 12}
    print(f"{n_docs} usable docs; {len(boiler)} boilerplate lines removed "
          f"(appeared on >= {cutoff} pages)", file=sys.stderr)

    # ---- pass 2: clean, chunk, dedup, split, write ---------------------------
    seen_hashes: set[str] = set()
    n_chunks = 0
    by_split: Counter = Counter()
    with open(args.out, "w", encoding="utf-8") as out_f:
        for fn, lines in docs.items():
            stem = Path(fn).stem
            info = meta.get(fn, {})
            url = info.get("url") or reconstruct_url(stem)
            title = info.get("title") or stem
            split = doc_split(stem)            # whole doc -> one split (leakage guard)

            body_lines = [l for l in lines if l not in boiler]
            for idx, text in enumerate(chunk_lines(body_lines)):
                n_words = len(text.split())
                if n_words < MIN_CHUNK_WORDS:
                    continue
                chash = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
                if chash in seen_hashes:       # dedup identical chunks
                    continue
                seen_hashes.add(chash)
                rec = {
                    "id": f"{SOURCE}/{stem}#c{idx:03d}",
                    "source": SOURCE,
                    "source_type": SOURCE_TYPE,
                    "title": title,
                    "origin": url,
                    "split": split,
                    "chunk_index": idx,
                    "n_words": n_words,
                    "content_hash": chash,
                    "tags": [SOURCE_TYPE, SOURCE],
                    "text": text,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_chunks += 1
                by_split[split] += 1

    # ---- audit report --------------------------------------------------------
    lines_out = [
        "DSSI Fermilab -- Stage 1 (st7 text) audit",
        f"text files found      : {len(files)}",
        f"usable documents      : {n_docs}",
        f"skipped (empty/failed): {skipped_empty + (len(files) - n_docs - skipped_empty)}",
        f"boilerplate lines cut : {len(boiler)}",
        f"chunks written        : {n_chunks}",
        f"  train chunks        : {by_split['train']}",
        f"  test  chunks        : {by_split['test']}",
        f"output file           : {args.out}",
    ]
    Path(args.audit).write_text("\n".join(lines_out) + "\n", encoding="utf-8")
    print("\n".join(lines_out))
    print(f"\nWrote {n_chunks} chunks -> {args.out} (audit: {args.audit})")


if __name__ == "__main__":
    main()
