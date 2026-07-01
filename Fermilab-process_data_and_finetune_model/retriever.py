#!/usr/bin/env python3
import json
import math
import re
import os
from pathlib import Path
import numpy as np

class Retriever:
    def __init__(self, corpus_path="corpus_st7.jsonl", k1=1.5, b=0.75, rrf_k=60):
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k
        self.documents = []
        self.doc_len = []
        self.doc_term_freqs = []
        self.df = {}
        self.avg_doc_len = 0.0
        
        # Paths setup
        self.base_dir = Path(__file__).parent
        self.corpus_file = self.base_dir / corpus_path if not Path(corpus_path).is_absolute() else Path(corpus_path)
        if not self.corpus_file.exists():
            # Fallback to local working directory
            self.corpus_file = Path(corpus_path)
            if not self.corpus_file.exists():
                raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

        print(f"Loading corpus from {self.corpus_file}...")
        with open(self.corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.documents.append(json.loads(line.strip()))
        
        self.n_docs = len(self.documents)
        print(f"Loaded {self.n_docs} chunks. Indexing BM25...")
        
        total_len = 0
        for doc in self.documents:
            tokens = self._tokenize(doc["text"])
            self.doc_len.append(len(tokens))
            total_len += len(tokens)
            
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)
            
            for t in tf.keys():
                self.df[t] = self.df.get(t, 0) + 1
                
        self.avg_doc_len = total_len / self.n_docs if self.n_docs > 0 else 1.0
        
        # Dense Retrieval Setup (CPU-friendly)
        self.embeddings_path = self.corpus_file.parent / "corpus_embeddings.npy"
        self.model = None
        self.doc_embeddings = None
        
        self._load_or_compute_embeddings()

    def _tokenize(self, text):
        return re.findall(r"\b[a-z0-9]{2,}\b", text.lower())

    def _load_or_compute_embeddings(self):
        """Loads pre-computed embeddings or computes them using SentenceTransformer on CPU."""
        if self.embeddings_path.exists():
            print(f"Loading pre-computed semantic embeddings from {self.embeddings_path}...")
            self.doc_embeddings = np.load(str(self.embeddings_path))
        else:
            print("No pre-computed embeddings found. Initializing SentenceTransformer on CPU...")
            from sentence_transformers import SentenceTransformer
            # Small, fast model: ~90MB, 384 dimensions
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            
            print("Encoding corpus chunks (this runs once and caches the result)...")
            texts = [doc["text"] for doc in self.documents]
            # Encode in batches
            self.doc_embeddings = self.model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
            
            print(f"Saving corpus embeddings to {self.embeddings_path}...")
            np.save(str(self.embeddings_path), self.doc_embeddings)
            
        # Normalize embeddings for fast cosine similarity via dot product
        norms = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # Avoid division by zero
        self.doc_embeddings = self.doc_embeddings / norms
        print(f"Semantic index ready. Shape: {self.doc_embeddings.shape}")

    def _get_bm25_scores(self, query_tokens, split=None):
        scores = {}
        for doc_idx, tf in enumerate(self.doc_term_freqs):
            doc = self.documents[doc_idx]
            if split and doc.get("split") != split:
                continue
                
            doc_len = self.doc_len[doc_idx]
            score = 0.0
            for t in query_tokens:
                if t in tf:
                    df_t = self.df.get(t, 0)
                    idf = math.log((self.n_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)
                    
                    tf_t = tf[t]
                    numerator = tf_t * (self.k1 + 1)
                    denominator = tf_t + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    score += idf * (numerator / denominator)
            if score > 0:
                scores[doc_idx] = score
        return scores

    def _get_semantic_scores(self, query, split=None):
        # Lazy load model if not loaded yet
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            
        # Embed query and normalize
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            query_embedding = query_embedding / q_norm
            
        # Cosine similarity is simply dot product now
        similarities = np.dot(self.doc_embeddings, query_embedding)
        
        scores = {}
        for doc_idx, sim in enumerate(similarities):
            doc = self.documents[doc_idx]
            if split and doc.get("split") != split:
                continue
            # Keep positive similarities
            if sim > 0:
                scores[doc_idx] = float(sim)
        return scores

    def retrieve(self, query: str, top_k: int = 3, split: str = None) -> list[dict]:
        """Hybrid Retrieval combining BM25 and Semantic Search using Reciprocal Rank Fusion (RRF)."""
        query_tokens = self._tokenize(query)
        
        # 1. Get BM25 rankings
        bm25_scores = self._get_bm25_scores(query_tokens, split)
        sorted_bm25 = sorted(bm25_scores.keys(), key=lambda x: bm25_scores[x], reverse=True)
        bm25_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(sorted_bm25)}
        
        # 2. Get Semantic rankings
        semantic_scores = self._get_semantic_scores(query, split)
        sorted_semantic = sorted(semantic_scores.keys(), key=lambda x: semantic_scores[x], reverse=True)
        semantic_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(sorted_semantic)}
        
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        all_candidates = set(bm25_ranks.keys()) | set(semantic_ranks.keys())
        
        for doc_idx in all_candidates:
            # BM25 rank score
            r_bm25 = bm25_ranks.get(doc_idx, len(self.documents))
            score_bm25 = 1.0 / (self.rrf_k + r_bm25)
            
            # Semantic rank score
            r_sem = semantic_ranks.get(doc_idx, len(self.documents))
            score_sem = 1.0 / (self.rrf_k + r_sem)
            
            rrf_scores[doc_idx] = score_bm25 + score_sem
            
        # 4. Title Reranking & Boosting
        # Boost document score if query terms match the document title
        query_words = set(query.lower().split())
        for doc_idx in rrf_scores.keys():
            doc = self.documents[doc_idx]
            title_words = set(doc.get("title", "").lower().split())
            matches = len(query_words & title_words)
            if matches > 0:
                # Add a boost proportional to matching words
                rrf_scores[doc_idx] += 0.02 * matches

        # Sort by RRF score
        sorted_indices = sorted(rrf_scores.keys(), key=lambda idx: rrf_scores[idx], reverse=True)[:top_k]
        
        results = []
        for idx in sorted_indices:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "title": doc["title"],
                "origin": doc["origin"],
                "split": doc.get("split", ""),
                # Show semantic similarity score if available, otherwise fallback to RRF score
                "score": round(semantic_scores.get(idx, bm25_scores.get(idx, rrf_scores[idx])), 4)
            })
            
        return results

if __name__ == "__main__":
    # Self-test the hybrid retriever
    try:
        ret = Retriever("corpus_st7.jsonl")
        test_queries = ["What did the SeaQuest experiment measure?", "DUNE experiment neutrino"]
        for q in test_queries:
            print(f"\nQuery: {q}")
            results = ret.retrieve(q, top_k=2)
            for r in results:
                print(f"  [{r['score']}] Title: {r['title']} | URL: {r['origin']}")
                print(f"    Text snippet: {r['text'][:150]}...")
    except Exception as e:
        print(f"Error during self-test: {e}")
