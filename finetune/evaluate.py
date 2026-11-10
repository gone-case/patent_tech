"""
Prior-art retrieval evaluation against real patent citations.

For every citing patent in eval_citations.parquet, its own title+abstract vector is taken from the index
(exactly what a search by patent ID does), the index is searched, and we check where its cited patents rank.

  Recall@k  share of the cited patents found in the top k (averaged over queries)
  Hit@k     share of queries with at least one cited patent in the top k
  MRR       1 / rank of the first cited patent within the candidate pool (0 if not found)

Two rankings are scored: SD (cosine only) and SDTD (the pool re-ranked with the AnalystAgent formula
using the query patent's own CPC codes).

Usage:
  python -m finetune.evaluate --label "frozen Patent-BERT"
  python -m finetune.evaluate --index finetuned/patents_faiss.index --meta finetuned/patents_meta.parquet --label "fine-tuned"
Results are appended to finetune/results.json.
"""
import argparse
import json
import os
import time

import faiss
import numpy as np
import pandas as pd

from agents.analyst_agent import AnalystAgent

EVAL_PATH = "data/processed/eval_citations.parquet"
RESULTS_PATH = "finetune/results.json"


def metrics(ranked_lists, relevant_sets, ks):
    out = {}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean([len(set(r[:k]) & rel) / len(rel) for r, rel in zip(ranked_lists, relevant_sets)]))
        out[f"hit@{k}"] = float(np.mean([bool(set(r[:k]) & rel) for r, rel in zip(ranked_lists, relevant_sets)]))
    rr = []
    for r, rel in zip(ranked_lists, relevant_sets):
        rank = next((i for i, pid in enumerate(r, 1) if pid in rel), None)
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr"] = float(np.mean(rr))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="patents_faiss.index")
    parser.add_argument("--meta", default="patents_meta.parquet")
    parser.add_argument("--eval", default=EVAL_PATH)
    parser.add_argument("--pool", type=int, default=100, help="candidates retrieved before SDTD re-ranking")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    start = time.time()
    index = faiss.read_index(args.index)
    meta = pd.read_parquet(args.meta, columns=["patent_id", "codes"])
    ids = meta.patent_id.tolist()
    codes = [list(c) for c in meta.codes]
    row = {pid: i for i, pid in enumerate(ids)}

    ev = pd.read_parquet(args.eval)
    ev = ev[ev.query_id.isin(row)]
    q_rows = np.array([row[q] for q in ev.query_id])
    relevant = [set(r) & set(row) for r in ev.relevant_ids]

    q_vecs = np.vstack([index.reconstruct(int(i)) for i in q_rows]).astype(np.float32)
    faiss.normalize_L2(q_vecs)
    scores, nbrs = index.search(q_vecs, args.pool + 1)

    analyst = AnalystAgent()
    sd_lists, sdtd_lists = [], []
    for qi, (s_row, n_row) in zip(q_rows, zip(scores, nbrs)):
        pool = [(int(n), float(s)) for n, s in zip(n_row, s_row) if n != -1 and n != qi][: args.pool]
        sd_lists.append([ids[n] for n, _ in pool])
        rescored = [
            (n, (analyst.calculate_technological_distance(codes[qi], codes[n]) + 1.0) * s / 2.0) for n, s in pool
        ]
        rescored.sort(key=lambda x: x[1], reverse=True)
        sdtd_lists.append([ids[n] for n, _ in rescored])

    ks = (1, 5, 10, 50)
    result = {
        "label": args.label or args.index,
        "index": args.index,
        "queries": len(q_rows),
        "relevant_pairs": int(sum(map(len, relevant))),
        "corpus": int(index.ntotal),
        "pool": args.pool,
        "SD": metrics(sd_lists, relevant, ks),
        "SDTD": metrics(sdtd_lists, relevant, ks),
        "seconds": round(time.time() - start, 1),
    }

    print(f"\n{result['label']}: {result['queries']:,} queries, {result['relevant_pairs']:,} cited patents, "
          f"corpus {result['corpus']:,}")
    print(f"{'ranking':<8}" + "".join(f"{m:>11}" for m in result["SD"]))
    for name in ("SD", "SDTD"):
        print(f"{name:<8}" + "".join(f"{v:>11.4f}" for v in result[name].values()))

    history = json.load(open(RESULTS_PATH)) if os.path.exists(RESULTS_PATH) else []
    history.append(result)
    with open(RESULTS_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Appended to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
