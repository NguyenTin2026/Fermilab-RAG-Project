"""Test cac phan KHONG can model cua retriever (tokenize + BM25 ranking)."""


def test_tokenize_lowercases_and_strips_punct(mini_retriever):
    toks = mini_retriever._tokenize("DUNE, neutrino! oscillations?")
    assert toks == ["dune", "neutrino", "oscillations"]


def test_tokenize_drops_single_chars(mini_retriever):
    # regex \b[a-z0-9]{2,}\b -> bo token 1 ky tu
    assert "a" not in mini_retriever._tokenize("a big accelerator")


def test_bm25_ranks_keyword_doc_first(mini_retriever):
    scores = mini_retriever._get_bm25_scores(
        mini_retriever._tokenize("neutrino oscillations DUNE"))
    top = max(scores, key=scores.get)
    assert mini_retriever.documents[top]["id"] == "d1"


def test_bm25_split_filter(mini_retriever):
    # Chi tinh chunk split='train' -> d1/d2 (test) bi loai
    scores = mini_retriever._get_bm25_scores(
        mini_retriever._tokenize("accelerator"), split="train")
    for idx in scores:
        assert mini_retriever.documents[idx]["split"] == "train"


def test_corpus_loaded(mini_retriever):
    assert mini_retriever.n_docs == 3
    assert mini_retriever.avg_doc_len > 0
