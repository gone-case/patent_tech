"""
RetrievalAgent:
High-speed candidate search using FAISS dense vector indices.
Retrieves the Top-K semantically closest patents (SD) from millions of candidates in milliseconds.
"""
import os
import ast
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
        fallback_excel: str = "patent_dataset.xlsx",
        device: Optional[str] = None,
    ):
        self.name = "RetrievalAgent"
        self.role = "Vector Search & Candidate Retrieval"
        self.index_path = index_path
        self.meta_path = meta_path
        self.fallback_excel = fallback_excel
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
            print(f"[{self.name}] Vector index active with {self.index.ntotal} indexed patents.")
        elif os.path.exists(self.fallback_excel):
            print(f"[{self.name}] No cached FAISS index found. Building index from {self.fallback_excel} (limit={max_samples})...")
            self._build_index_from_excel(self.fallback_excel, max_records=max_samples)
        else:
            print(f"[{self.name}] Warning: No index or dataset found. Ready to ingest new embeddings.")
        self._index_initialized = True

    def _build_index_from_excel(self, excel_path: str, max_records: Optional[int] = None):
        """Builds a normalized Inner-Product FAISS index (Cosine Similarity) from local dataset."""
        df = pd.read_excel(excel_path)
        if max_records:
            df = df.head(max_records)
        
        # Deduplicate target patents to create a database of candidate patents
        records = []
        texts = []
        seen = set()

        # Extract both candidate sides (target_meta and meta)
        for _, row in df.iterrows():
            target_meta = str(row.get("target_meta", "")).strip()
            if target_meta and target_meta not in seen:
                seen.add(target_meta)
                raw_codes = row.get("target_ipc", "['G06F']")
                try:
                    codes = ast.literal_eval(raw_codes) if isinstance(raw_codes, str) else list(raw_codes)
                except Exception:
                    codes = [str(raw_codes)[:4].upper()]
                records.append({
                    "patent_id": f"PAT_{len(records):06d}",
                    "meta": target_meta,
                    "codes": codes,
                })
                texts.append(target_meta)

        if not texts:
            print(f"[{self.name}] No valid text records found to index.")
            return

        print(f"[{self.name}] Computing Patent-BERT dense vectors for {len(texts)} patents...")
        embeddings = self.encode_texts(texts)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.metadata = records

        # Persist index and metadata for instant future startup
        try:
            faiss.write_index(self.index, self.index_path)
            pd.DataFrame(records).to_parquet(self.meta_path, index=False)
            print(f"[{self.name}] Successfully cached FAISS index ({self.index.ntotal} vectors) to disk.")
        except Exception as e:
            print(f"[{self.name}] Cache notice: {e}")

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
