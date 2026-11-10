"""
Compares text encoders for prior-art retrieval on computer-science patents, against real citations.

Test set: CS patents (CPC G06 computing, G16 informatics, H04L networking / crypto). Every citing CS patent whose
cited patents are also CS, those cited patents, and random CS patents as distractors up to --corpus patents.
Every encoder searches the same corpus; metrics are reported for SD (cosine) and SDTD (pool re-ranked with
the AnalystAgent formula, using the query patent's own CPC codes).

Encoders:
  patent-bert   anferico/bert-for-patents, mean pooling (vectors taken from patents_faiss.index)
  any sentence-transformers model, e.g. AI-Growth-Lab/PatentSBERTa, sentence-transformers/all-mpnet-base-v2

Usage:
  python -m finetune.compare_encoders --models patent-bert AI-Growth-Lab/PatentSBERTa sentence-transformers/all-mpnet-base-v2
Embeddings are cached in data/processed/encoder_cache/, results appended to finetune/encoder_comparison.json.
"""
import argparse
import json
import os
import random
import time

import faiss
import numpy as np
import pandas as pd

from agents.analyst_agent import AnalystAgent
from finetune.evaluate import metrics

CS_PREFIXES = ("G06", "G16", "H04L")
CACHE_DIR = "data/processed/encoder_cache"
RESULTS_PATH = "finetune/encoder_comparison.json"


def build_test_set(corpus_size: int, seed: int):
    meta = pd.read_parquet("patents_meta.parquet", columns=["patent_id", "meta", "codes"])
    meta["codes"] = meta.codes.map(list)
    is_cs = meta.codes.map(lambda cs: any(str(c).startswith(CS_PREFIXES) for c in cs))
    cs_ids = set(meta.patent_id[is_cs])

    ev = pd.read_parquet("data/processed/eval_citations.parquet")
    queries = []
    for q, rel in zip(ev.query_id, ev.relevant_ids):
        rel = [r for r in rel if r in cs_ids]
        if q in cs_ids and rel:
            queries.append((q, set(rel)))

    needed = {q for q, _ in queries} | {r for _, rel in queries for r in rel}
    others = sorted(cs_ids - needed)
    random.Random(seed).shuffle(others)
    corpus_ids = sorted(needed) + others[: max(corpus_size - len(needed), 0)]

    sub = meta.set_index("patent_id").loc[corpus_ids].reset_index()
    return sub, queries, len(cs_ids)


def embed(model_name: str, sub: pd.DataFrame, tag: str) -> np.ndarray:
    slug = model_name.replace("/", "__")
    path = os.path.join(CACHE_DIR, f"{slug}_{tag}.npy")
    if os.path.exists(path):
        return np.load(path)

    start = time.time()
    if model_name == "patent-bert":
        # Same vectors the app searches with
        index = faiss.read_index("patents_faiss.index")
        row = {pid: i for i, pid in enumerate(pd.read_parquet("patents_meta.parquet", columns=["patent_id"]).patent_id)}
        vecs = np.vstack([index.reconstruct(row[pid]) for pid in sub.patent_id]).astype(np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device="cpu")
        texts = sub.meta.tolist()
        chunks, step = [], 500
        for i in range(0, len(texts), step):
            chunks.append(model.encode(texts[i:i + step], batch_size=32, convert_to_numpy=True))
            done = min(i + step, len(texts))
            rate = done / (time.time() - start)
            print(f"  [{model_name}] {done:,}/{len(texts):,} ({rate:.1f}/s, ~{(len(texts) - done) / rate / 60:.0f} min left)",
                  flush=True)
        vecs = np.vstack(chunks).astype(np.float32)

    faiss.normalize_L2(vecs)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(path, vecs)
    print(f"  [{model_name}] embedded {len(vecs):,} patents in {(time.time() - start) / 60:.1f} min", flush=True)
    return vecs


def evaluate(vecs: np.ndarray, sub: pd.DataFrame, queries, pool: int):
    ids = sub.patent_id.tolist()
    codes = sub.codes.tolist()
    row = {pid: i for i, pid in enumerate(ids)}
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    q_rows = np.array([row[q] for q, _ in queries])
    scores, nbrs = index.search(vecs[q_rows], pool + 1)
    analyst = AnalystAgent()
    sd_lists, sdtd_lists = [], []
    for qi, s_row, n_row in zip(q_rows, scores, nbrs):
        cands = [(int(n), float(s)) for n, s in zip(n_row, s_row) if n != qi][:pool]
        sd_lists.append([ids[n] for n, _ in cands])
        rescored = sorted(((n, (analyst.calculate_technological_distance(codes[qi], codes[n]) + 1.0) * s / 2.0)
                           for n, s in cands), key=lambda x: x[1], reverse=True)
        sdtd_lists.append([ids[n] for n, _ in rescored])
    relevant = [rel for _, rel in queries]
    ks = (1, 5, 10, 50)
    return {"SD": metrics(sd_lists, relevant, ks), "SDTD": metrics(sdtd_lists, relevant, ks)}


def print_table(results):
    cols = ["recall@1", "recall@5", "recall@10", "hit@10", "recall@50", "mrr"]
    print(f"\n{'encoder':<42}{'ranking':<7}" + "".join(f"{c:>11}" for c in cols))
    base = results[0]
    for r in results:
        for rank in ("SD", "SDTD"):
            print(f"{r['encoder']:<42}{rank:<7}" + "".join(f"{r[rank][c]:>11.4f}" for c in cols))
    if len(results) > 1:
        print(f"\nChange vs {base['encoder']} (same ranking):")
        for r in results[1:]:
            for rank in ("SD", "SDTD"):
                print(f"{r['encoder']:<42}{rank:<7}" + "".join(f"{r[rank][c] - base[rank][c]:>+11.4f}" for c in cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["patent-bert", "AI-Growth-Lab/PatentSBERTa"])
    parser.add_argument("--corpus", type=int, default=10000)
    parser.add_argument("--pool", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    sub, queries, n_cs = build_test_set(args.corpus, args.seed)
    tag = f"cs{len(sub)}_s{args.seed}"
    print(f"CS test set: {len(sub):,} of {n_cs:,} CS patents searched; {len(queries)} query patents, "
          f"{sum(len(r) for _, r in queries):,} citations", flush=True)

    results = []
    for name in args.models:
        vecs = embed(name, sub, tag)
        res = {"encoder": name, "dim": int(vecs.shape[1]), "corpus": len(sub), "queries": len(queries),
               "pool": args.pool, **evaluate(vecs, sub, queries, args.pool)}
        results.append(res)
        print_table(results)

    history = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else []
    history.append({"test_set": tag, "results": results})
    with open(RESULTS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nAppended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
