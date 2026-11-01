"""
RetrievalAgent:
High-speed candidate search using FAISS dense vector indices.
Retrieves the Top-K semantically closest patents (SD) from millions of candidates in milliseconds.
"""
import os
import numpy as np
import pandas as pd
import faiss
import torch
from typing import List, Dict, Tuple, Optional
from train_models.sbert import Sentence_Bert


class RetrievalAgent:
    def __init__(
        self,
        index_path: str = "patents_faiss.index",
        meta_path: str = "patents_meta.parquet",
        device: Optional[str] = None,
    ):
        self.name = "RetrievalAgent"
        self.role = "Vector Search & Candidate Retrieval"
        self.index_path = index_path
        self.meta_path = meta_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict] = []
        self._sbert: Optional[Sentence_Bert] = None

        # Lazy-loaded on demand
        self._index_initialized = False

    @property
    def sbert(self) -> Sentence_Bert:
        """Lazy loader for Patent-BERT encoder."""
        if self._sbert is None:
            print(f"[{self.name}] Initializing Patent-BERT embedding engine on {self.device}...")
            self._sbert = Sentence_Bert(device=self.device).eval()
        return self._sbert

    def ensure_index_ready(self, max_samples: Optional[int] = 50):
        """Ensures that the FAISS index and metadata are loaded into memory."""
        if self._index_initialized and self.index is not None:
            return

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            print(f"[{self.name}] Loading precomputed FAISS vector index: {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            meta_df = pd.read_parquet(self.meta_path)
            self.metadata = meta_df.to_dict(orient="records")
            self._normalize_metadata_codes()
            print(f"[{self.name}] Vector index active with {self.index.ntotal} indexed patents.")
        else:
            print(f"[{self.name}] Warning: No index found. Build one with the Colab notebook or build_google_patents_index.py.")
        self._index_initialized = True

    def _normalize_metadata_codes(self):
        """
        Converts parquet arrays to lists, and repairs indices built from ccdv/patent-classification
        by older notebook versions, which stored the CPC section label as a fake '<section>06F'
        subclass (A06F, H06F, ...). Those only carry section-level information.
        """
        for rec in self.metadata:
            # Array columns (codes, ipc_codes, cpc_codes, ...) load as numpy arrays, which aren't JSON serializable
            for key, value in rec.items():
                if isinstance(value, np.ndarray):
                    rec[key] = [v.item() if isinstance(v, np.generic) else v for v in value]
                elif isinstance(value, np.generic):
                    rec[key] = value.item()
            codes = rec.get("codes")
            rec["codes"] = [str(c) for c in codes] if codes is not None else []

        synthetic = {f"{s}06F" for s in "ABCDEFGHY"}
        all_codes = {c for rec in self.metadata for c in rec["codes"]}
        if all_codes and all_codes <= synthetic and len(all_codes) > 1:
            print(f"[{self.name}] Index uses section-only labels (ccdv); mapping '<X>06F' -> section '<X>'.")
            for rec in self.metadata:
                rec["codes"] = [c[0] for c in rec["codes"]]

    def encode_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        """Encodes texts into 1024-dimensional Patent-BERT embeddings."""
        all_embeds = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                tok = self.sbert.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.device)
                
                outputs = self.sbert.bert(**tok)
                # Mean pooling across tokens
                pooled = torch.mean(outputs.last_hidden_state, dim=1)
                all_embeds.append(pooled.cpu().numpy().astype(np.float32))

        return np.vstack(all_embeds)

    def retrieve(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Retrieves the Top-K semantically closest patents for a given query text.
        Returns candidates with raw semantic similarity (SD) score.
        """
        self.ensure_index_ready()
        if self.index is None or self.index.ntotal == 0:
            raise RuntimeError("FAISS index is not initialized or empty.")

        query_vec = self.encode_texts([query_text])
        faiss.normalize_L2(query_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["sd_score"] = float(score)  # Cosine similarity in range [-1, 1]
                candidates.append(item)

        return candidates
