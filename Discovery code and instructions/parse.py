"""
discovery_ft.parse
==================

Turn the raw JSON each student downloaded into ONE common record shape.

The three students pulled from three very different sources, so their JSON
looks nothing alike:

    st5  -> USPTO patents      (flat dict: title / abstract / claims / cpc_codes ...)
    st2  -> ClinicalTrials.gov (deeply nested protocolSection.* )
    st1  -> AUTM web pages     (flat dict: title / body / excerpt ...)

Downstream code (building training pairs, fine-tuning) must NOT have to know
about any of that. So every source gets converted into the same `DiscoveryDoc`
with a single canonical `embedding_text` field.

Two ideas matter here and are worth saying out loud, because they are the whole
reason this file exists:

1. SIDE. Every record is either a "document" (a patent / invention that can be
   licensed) or a "query" (a research-interest statement describing what someone
   is looking for). Patents and AUTM listings are documents. Clinical trials are
   treated as query-side material, because a trial describes an active research
   interest in business/clinical language -- exactly the kind of text a pharma
   scout would write. Getting this split right is the point of the whole project:
   we are matching query-language against document-language across two registers.

2. embedding_text. For each record we assemble the ONE string that will actually
   be turned into a vector. For patents that means title + abstract + the first
   few claims + the plain-English CPC descriptions, with legal boilerplate
   stripped. Storing this explicitly (instead of re-deriving it at train time)
   means the text is reproducible and reviewable -- a student can open the JSONL
   and see exactly what the model saw.

Run `python -m discovery_ft.parse <folder>` to see a summary, or import
`load_folder()` from another script.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator


# --------------------------------------------------------------------------- #
# The unified record
# --------------------------------------------------------------------------- #
@dataclass
class DiscoveryDoc:
    doc_id: str                 # stable unique id (patent no, NCT id, slug)
    source: str                 # "uspto_patent" | "clinicaltrials" | "autm"
    side: str                   # "document" | "query"
    title: str
    embedding_text: str         # the canonical string we embed
    # provenance / debugging -- never embedded, just carried along
    source_url: str = ""
    retrieved_date: str = ""
    cpc_codes: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #

# Phrases that show up in nearly every patent and carry no discriminating
# signal. Stripping them stops the model wasting context on legal scaffolding
# and stops every patent looking similar for the wrong reason.
_PATENT_BOILERPLATE = [
    r"the present invention relates to",
    r"in one embodiment",
    r"in another embodiment",
    r"in a preferred embodiment",
    r"according to (?:the|one|an) (?:aspect|embodiment)",
    r"cross[- ]reference to related applications?",
    r"field of the invention",
    r"background of the invention",
    r"brief description of the drawings?",
    r"detailed description of the (?:invention|embodiments?)",
    r"it (?:is|will be) (?:understood|appreciated) that",
    r"as used herein,?",
    r"without departing from the (?:scope|spirit)",
]
_BOILER_RE = re.compile("|".join(_PATENT_BOILERPLATE), flags=re.IGNORECASE)

# USPTO encodes sub/superscripts as ".sup." / ".sub." in the text dump
# (e.g. "FLT3.sup.ITD"). Collapse those so gene/variant names read normally.
_SUP_SUB_RE = re.compile(r"\.su[bp]\.")

# Claim numbering like "1 .", "12 ." at the start of a clause.
_CLAIM_NUM_RE = re.compile(r"(?m)^\s*\d+\s*\.\s*")

_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    if not text:
        return ""
    text = _SUP_SUB_RE.sub("", text)
    text = _BOILER_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _first_claims(claims_text: str, n: int = 3, max_chars: int = 1800) -> str:
    """
    Keep only the first `n` claims. Independent claims come first and carry the
    core invention; later dependent claims mostly add narrow limitations and
    inflate length. We also hard-cap characters so one runaway claim can't blow
    the budget.
    """
    if not claims_text:
        return ""
    # Split on claim numbers but keep the text after each.
    parts = _CLAIM_NUM_RE.split(claims_text)
    parts = [p.strip() for p in parts if p.strip()]
    kept = " ".join(parts[:n]) if parts else claims_text
    kept = _clean(kept)
    return kept[:max_chars]


def _join_nonempty(*chunks: str, sep: str = "\n\n") -> str:
    return sep.join(c for c in (chunk.strip() for chunk in chunks) if c)


# --------------------------------------------------------------------------- #
# Per-source parsers
# --------------------------------------------------------------------------- #
def parse_patent(d: dict) -> DiscoveryDoc:
    """st5: USPTO patent. Flat dict. This is a DOCUMENT-side record."""
    title = (d.get("title") or "").strip()
    abstract = _clean(d.get("abstract", ""))
    claims = _first_claims(d.get("claims") or d.get("claims_excerpt", ""))

    # CPC codes are like "A61K31/00". Turn them into a short readable hint using
    # the section map below. Even the one-line gloss adds standardized vocabulary
    # that helps categorization.
    cpc = d.get("cpc_codes", []) or []
    cpc_gloss = cpc_to_text(cpc)

    embedding_text = _join_nonempty(title, abstract, claims, cpc_gloss)

    return DiscoveryDoc(
        doc_id=d.get("document_id") or d.get("patent_number") or title[:40],
        source="uspto_patent",
        side="document",
        title=title,
        embedding_text=embedding_text,
        source_url=d.get("source_url", ""),
        retrieved_date=d.get("retrieved_date", ""),
        cpc_codes=cpc,
        extra={
            "assignees": d.get("assignees", []),
            "pharma_label_type": d.get("pharma_label_type", ""),
            "publication_date": d.get("publication_date", ""),
        },
    )


def parse_clinical_trial(d: dict) -> DiscoveryDoc:
    """
    st2: ClinicalTrials.gov v2 record. Deeply nested. This is a QUERY-side
    record -- a trial states a research interest in clinical/business language.
    """
    ps = d.get("protocolSection", {})
    ident = ps.get("identificationModule", {})
    desc = ps.get("descriptionModule", {})
    cond = ps.get("conditionsModule", {})
    arms = ps.get("armsInterventionsModule", {})

    nct = ident.get("nctId", "")
    title = (ident.get("briefTitle") or ident.get("officialTitle") or "").strip()
    brief = _clean(desc.get("briefSummary", ""))

    conditions = ", ".join(cond.get("conditions", []) or [])
    keywords = ", ".join(cond.get("keywords", []) or [])

    # Intervention names + types capture the "what is being tested" signal.
    interventions = arms.get("interventions", []) or []
    interv_str = "; ".join(
        f"{i.get('type','').title()}: {i.get('name','')}".strip(": ")
        for i in interventions
        if i.get("name")
    )

    cond_line = f"Conditions: {conditions}" if conditions else ""
    kw_line = f"Keywords: {keywords}" if keywords else ""
    interv_line = f"Interventions: {interv_str}" if interv_str else ""

    embedding_text = _join_nonempty(title, brief, cond_line, kw_line, interv_line)

    return DiscoveryDoc(
        doc_id=nct or title[:40],
        source="clinicaltrials",
        side="query",
        title=title,
        embedding_text=embedding_text,
        source_url=f"https://clinicaltrials.gov/study/{nct}" if nct else "",
        retrieved_date=d.get("derivedSection", {})
        .get("miscInfoModule", {})
        .get("versionHolder", ""),
        extra={
            "conditions": cond.get("conditions", []),
            "phase": ps.get("designModule", {}).get("phases", []),
        },
    )


def parse_autm(d: dict) -> DiscoveryDoc:
    """
    st1: AUTM web page. Flat dict. DOCUMENT-side, but be careful: not every AUTM
    page is an invention listing -- many are newsletters / press releases. We
    keep the record but flag low-signal ones via `topic_score` so a later filter
    can drop them.
    """
    title = (d.get("title") or "").strip()
    excerpt = _clean(d.get("excerpt", ""))
    body = _clean(d.get("body", ""))
    # Body can be long; cap it so a press release doesn't dominate.
    body = body[:1500]

    embedding_text = _join_nonempty(title, excerpt, body)

    return DiscoveryDoc(
        doc_id=d.get("url", title)[:80],
        source="autm",
        side="document",
        title=title,
        embedding_text=embedding_text,
        source_url=d.get("url", ""),
        retrieved_date=d.get("date", ""),
        extra={
            "content_type": d.get("content_type", ""),
            "topic_score": d.get("topic_score", 0),
        },
    )


# --------------------------------------------------------------------------- #
# CPC code -> readable hint
# --------------------------------------------------------------------------- #
# Minimal map covering the pharma-relevant sections the project targets. This is
# deliberately small; extend it as you meet new prefixes. The goal is a short
# natural-language gloss, not a full CPC dictionary.
_CPC_SECTION = {
    "A61K": "preparations for medical purposes",
    "A61P": "therapeutic activity of compounds",
    "A61B": "diagnosis, surgery, identification",
    "C07K": "peptides",
    "C07D": "heterocyclic compounds",
    "C12N": "microorganisms, enzymes, mutation, genetic engineering",
    "C12Q": "measuring or testing involving nucleic acids",
    "G01N": "investigating or analysing materials",
    "A01K": "animal husbandry; model organisms",
    "C12P": "fermentation or enzyme processes",
}


def cpc_to_text(codes: list[str]) -> str:
    if not codes:
        return ""
    seen = []
    for c in codes:
        prefix = re.match(r"[A-Z]\d{2}[A-Z]", c or "")
        if not prefix:
            continue
        key = prefix.group(0)
        gloss = _CPC_SECTION.get(key)
        if gloss and gloss not in seen:
            seen.append(gloss)
    if not seen:
        return ""
    return "Technical field: " + "; ".join(seen) + "."


# --------------------------------------------------------------------------- #
# Source detection + folder loading
# --------------------------------------------------------------------------- #
def detect_and_parse(d: dict) -> DiscoveryDoc | None:
    """Look at the keys and route to the right parser."""
    if "protocolSection" in d:                       # ClinicalTrials.gov
        return parse_clinical_trial(d)
    if "claims" in d or "cpc_codes" in d:            # USPTO patent
        return parse_patent(d)
    if "body" in d and "site" in d:                  # AUTM page
        return parse_autm(d)
    return None


def load_folder(root: str | Path) -> Iterator[DiscoveryDoc]:
    """
    Walk a folder, parse every .json into a DiscoveryDoc, skipping anything
    unrecognized or empty. Yields lazily so it scales to the full ~10k corpus.
    """
    root = Path(root)
    for path in sorted(root.rglob("*.json")):
        # Skip the raw/ mirror patents keep -- it duplicates the cleaned record.
        if "/raw/" in str(path).replace("\\", "/"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(d, dict):
            continue
        doc = detect_and_parse(d)
        if doc and doc.embedding_text.strip():
            yield doc


# --------------------------------------------------------------------------- #
# CLI: quick sanity summary
# --------------------------------------------------------------------------- #
def _summary(root: str) -> None:
    from collections import Counter

    by_source = Counter()
    by_side = Counter()
    lengths = []
    examples = {}
    for doc in load_folder(root):
        by_source[doc.source] += 1
        by_side[doc.side] += 1
        lengths.append(len(doc.embedding_text))
        examples.setdefault(doc.source, doc)

    total = sum(by_source.values())
    print(f"Parsed {total} usable records from {root}\n")
    print("By source:")
    for s, n in by_source.items():
        print(f"  {s:18s} {n}")
    print("\nBy side (this is the asymmetric-matching split):")
    for s, n in by_side.items():
        print(f"  {s:10s} {n}")
    if lengths:
        lengths.sort()
        print(
            f"\nembedding_text length (chars): "
            f"min={lengths[0]} median={lengths[len(lengths)//2]} max={lengths[-1]}"
        )
    print("\n--- one example per source ---")
    for s, doc in examples.items():
        print(f"\n[{s}] side={doc.side} id={doc.doc_id}")
        print(f"  title: {doc.title[:80]}")
        snippet = doc.embedding_text[:280].replace("\n", " ")
        print(f"  embedding_text: {snippet}...")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    _summary(folder)
