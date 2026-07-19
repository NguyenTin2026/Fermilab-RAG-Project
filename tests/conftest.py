"""Fixtures dung chung: dung mot corpus nho + embeddings gia de test retriever
offline (khong tai model, khong can GPU)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RETR_DIR = REPO_ROOT / "Fermilab-process_data_and_finetune_model"
sys.path.insert(0, str(RETR_DIR))

MINI_DOCS = [
    {"id": "d1", "text": "The DUNE experiment studies neutrino oscillations at LBNF.",
     "title": "DUNE neutrino", "origin": "http://x/dune", "split": "test"},
    {"id": "d2", "text": "The muon g-2 experiment measures the muon magnetic moment.",
     "title": "Muon g-2", "origin": "http://x/g2", "split": "test"},
    {"id": "d3", "text": "PIP-II is a superconducting linear accelerator at Fermilab.",
     "title": "PIP-II accelerator", "origin": "http://x/pip2", "split": "train"},
]


@pytest.fixture
def mini_retriever(tmp_path):
    """Retriever tren corpus nho, seed san corpus_embeddings.npy de bo qua model load."""
    corpus = tmp_path / "mini.jsonl"
    with open(corpus, "w", encoding="utf-8") as f:
        for d in MINI_DOCS:
            f.write(json.dumps(d) + "\n")
    # 384 chieu = so chieu cua all-MiniLM-L6-v2; noi dung khong quan trong voi test BM25.
    np.save(tmp_path / "corpus_embeddings.npy",
            np.random.RandomState(0).randn(len(MINI_DOCS), 384).astype("float32"))
    from retriever import Retriever
    return Retriever(str(corpus))
