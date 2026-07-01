#!/usr/bin/env python3
"""
01_process_corpus.py  --  DSSI Fermilab LLM Project, Stage 1
Turn the raw student downloads (st4 / st6 / st8) into clean, chunked,
deduplicated JSONL records in the shared schema.

Usage:
    pip install pymupdf
    python 01_process_corpus.py --in_dir ./data --out corpus.jsonl

What it does, in order:
  1. Audits every file (catches zero-byte / failed downloads -> audit_report.txt)
  2. Extracts text per page with PyMuPDF, as layout blocks
  3. Removes boilerplate:
       - blocks repeated on many pages (nav menus, running headers)
       - known fnal.gov footer/nav lines (webpage-PDFs only)
       - page numbers and tiny junk blocks
       - reference sections (scholarly papers only)
  4. Fixes hyphenation and whitespace
  5. Chunks into ~350-450 word passages on block boundaries
  6. Deduplicates chunks by content hash
  7. Assigns a deterministic doc-level train/test split  (BEFORE any Q&A
     generation may happen -- this is the leakage guard)
  8. Writes one JSON object per chunk to corpus.jsonl
"""

import argparse, hashlib, json, re, sys, unicodedata
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

# ----------------------------------------------------------------------
# Folder -> source-type map.  Extend as more students contribute folders.
SOURCE_TYPE = {
    "st4": "arxiv_paper",
    "st6": "webpage_pdf",
    "st8": "arxiv_paper",
}

# Lines that mark fnal.gov site chrome (footers / nav) in webpage-PDFs.
NAV_MARKERS = {
    "subscribe to our newsletters", "about", "community", "newsroom",
    "contact", "resources", "for employees", "for industry", "jobs",
    "science", "home", "phone book", "people", "news", "how it works",
    "for neighbors and businesses", "for the teams", "particle physics 101",
    "dune at lbnf", "particle physics", "particle accelerators",
    "security, privacy, legal | use of cookies",
    "detectors, computing,", "quantum",
}
NAV_FOOTER_RE = re.compile(
    r"managed by fermi forward|fermi national accelerator laboratory$"
    r"|we bring the world together", re.I)

REF_HEAD_RE = re.compile(
    r"^\s*(references|bibliography|literature\s+cited)\s*$", re.I)
REF_INLINE_RE = re.compile(
    r"\b(References|REFERENCES|Bibliography|Literature\s+Cited|LITERATURE\s+CITED)\b")
CITE_RE = re.compile(r"\[\d{1,3}\]\s|arXiv:\d|\(\d{4}\)[,.]|Phys\.\s?Rev|JHEP\s")
PAGENUM_RE  = re.compile(r"^\s*(page\s*)?\d{1,4}\s*$", re.I)

TARGET_WORDS, MIN_WORDS = 420, 120     # ~550-650 tokens per chunk


# ----------------------------------------------------------------------
def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")                  # soft hyphens
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text) # de-hyphenate line wraps
    text = re.sub(r"\s*\n\s*", " ", text)              # unwrap lines in a block
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def origin_from_name(folder: str, stem: str):
    """Best-effort reconstruction of where a file came from."""
    if SOURCE_TYPE[folder] == "arxiv_paper":
        m = re.search(r"(\d{4}\.\d{4,5}(v\d+)?)", stem)
        return f"https://arxiv.org/abs/{m.group(1)}" if m else None
    # webpage_pdf:  02_lbnf_dune_fnal_gov_about_overview -> lbnf-dune.fnal.gov/about/overview
    toks = re.sub(r"^\d+_", "", stem).split("_")
    if "gov" in toks:
        g = toks.index("gov")
        domain = "-".join(toks[:g - 1]) + "." + toks[g - 1] + ".gov"
        path = "/".join(toks[g + 1:])
        return f"https://{domain}/{path}".rstrip("/")
    return None


def extract_blocks(doc: fitz.Document):
    """Page-ordered text blocks; boilerplate removed at LINE level.

    Site nav and running headers often arrive merged inside larger blocks,
    so per-block matching misses them. We therefore filter individual lines:
      - lines repeated on >=40% of pages (nav menus, running headers)
      - known fnal.gov nav/footer lines
      - bare page numbers and pipe-separator strips
    then rejoin the surviving lines of each block.
    """
    raw_pages = []
    for page in doc:
        blocks = [b[4] for b in page.get_text("blocks") if b[6] == 0]
        raw_pages.append([b.splitlines() for b in blocks if b.strip()])

    # line-frequency across pages -> repeated chrome
    if len(raw_pages) > 1:
        freq = Counter(
            ln.strip().lower()
            for pg in raw_pages
            for ln in {l for blk in pg for l in blk if l.strip()}
        )
        cutoff = max(2, round(0.4 * len(raw_pages)))
        repeated = {k for k, v in freq.items() if v >= cutoff and len(k) > 2}
    else:
        repeated = set()

    def keep_line(ln: str) -> bool:
        low = ln.strip().lower()
        if not low:                                   return False
        if low in repeated:                           return False
        if low in NAV_MARKERS:                        return False
        if NAV_FOOTER_RE.search(low):                 return False
        if PAGENUM_RE.match(ln):                      return False
        if low.replace("|", "").strip() == "":        return False
        return True

    kept = []
    for pg in raw_pages:
        for blk in pg:
            lines = [ln for ln in blk if keep_line(ln)]
            if not lines:
                continue
            b = normalize("\n".join(lines))
            if len(b) < 3:
                continue
            if alpha_ratio(b) < 0.45 and len(b) > 60:  # numeric tables/axis dumps
                continue
            kept.append(b)
    return kept


def alpha_ratio(s: str) -> float:
    return sum(c.isalpha() for c in s) / max(1, len(s))


def clean_blocks(blocks, source_type):
    out, in_refs, n = [], False, len(blocks)
    for idx, b in enumerate(blocks):
        if in_refs:
            continue
        if source_type == "arxiv_paper":
            # heading as its own block, or merged at the start of a larger one
            if REF_HEAD_RE.match(b):
                in_refs = True; continue
            m = REF_INLINE_RE.search(b)
            if m and idx > 0.4 * n:               # ignore "references" used in prose early on
                head = b[:m.start()].strip()
                if len(head.split()) > 15:        # keep real text before the heading
                    out.append(head)
                in_refs = True; continue
            # safety net: blocks that *look like* a bibliography in the back matter
            if idx > 0.6 * n and len(CITE_RE.findall(b)) >= 3:
                continue
        out.append(b)
    return out


def chunk(blocks, source_type):
    """Greedy packing of blocks into ~TARGET_WORDS chunks; merge tiny tails."""
    chunks, cur, n = [], [], 0
    for b in blocks:
        w = len(b.split())
        if n + w > TARGET_WORDS and n >= MIN_WORDS:
            chunks.append(" ".join(cur)); cur, n = [], 0
        cur.append(b); n += w
    if cur:
        if chunks and n < MIN_WORDS:
            chunks[-1] += " " + " ".join(cur)    # merge short tail into previous
        else:
            chunks.append(" ".join(cur))

    def is_junk(c):
        if alpha_ratio(c) < 0.35:                # table assembled from tiny blocks
            return True
        if source_type == "arxiv_paper":         # bibliography assembled from tiny blocks
            hits = len(CITE_RE.findall(c))
            return hits >= 5 and hits / max(1, len(c.split())) * 100 >= 2
        return False

    return [c for c in chunks if not is_junk(c)]


def doc_split(doc_id: str) -> str:
    """Deterministic 80/20 doc-level split. Same input -> same answer, forever."""
    return "test" if int(hashlib.sha256(doc_id.encode()).hexdigest(), 16) % 5 == 0 else "train"


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="./data")
    ap.add_argument("--out", default="corpus.jsonl")
    args = ap.parse_args()
    root = Path(args.in_dir)

    audit, records, seen_hashes = [], [], set()

    for folder in sorted(SOURCE_TYPE):
        fdir = root / folder
        if not fdir.exists():
            audit.append(f"[WARN] missing folder: {folder}")
            continue
        for pdf in sorted(fdir.glob("*.pdf")):
            size = pdf.stat().st_size
            if size == 0:
                audit.append(f"[FAIL] {folder}/{pdf.name}: 0 bytes - download failed, re-download")
                continue
            try:
                doc = fitz.open(pdf)
            except Exception as e:
                audit.append(f"[FAIL] {folder}/{pdf.name}: unreadable ({e})")
                continue

            stype  = SOURCE_TYPE[folder]
            blocks = clean_blocks(extract_blocks(doc), stype)
            chunks = chunk(blocks, stype)
            if not chunks:
                audit.append(f"[WARN] {folder}/{pdf.name}: no text extracted")
                continue

            doc_id = f"{folder}/{pdf.stem}"
            split  = doc_split(doc_id)
            title  = blocks[0][:120] if blocks else pdf.stem
            origin = origin_from_name(folder, pdf.stem)

            kept = dup = 0
            for i, text in enumerate(chunks):
                h = sha(text)
                if h in seen_hashes:
                    dup += 1; continue
                seen_hashes.add(h)
                records.append({
                    "id": f"{doc_id}#c{i:03d}",
                    "source": folder,
                    "source_type": stype,
                    "title": title,
                    "origin": origin,
                    "split": split,
                    "chunk_index": i,
                    "n_words": len(text.split()),
                    "content_hash": h,
                    "tags": [stype, folder],
                    "text": text,
                })
                kept += 1
            audit.append(f"[ OK ] {folder}/{pdf.name}: {len(doc)}p -> "
                         f"{kept} chunks ({dup} dups dropped) [{split}]")

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    Path("audit_report.txt").write_text("\n".join(audit) + "\n")

    n_tr = sum(r["split"] == "train" for r in records)
    print("\n".join(audit))
    print(f"\nWrote {len(records)} chunks -> {args.out}  "
          f"(train: {n_tr}, test: {len(records) - n_tr})")
    if any(a.startswith("[FAIL]") for a in audit):
        print("!! Some files FAILED - see audit_report.txt", file=sys.stderr)


if __name__ == "__main__":
    main()
