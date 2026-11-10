"""
Builds the fine-tuning and evaluation sets for Patent-BERT from the Google Patents pull.

  Evaluation: citation pairs where both the citing and the cited patent are in the index.
              A citation is an examiner/applicant judgement that the cited patent is prior art,
              so "does the cited patent show up in the top 10?" measures prior-art search quality.
  Training:   pairs of patents that share an exact CPC subgroup (weak supervision, as in PatentSBERTa).
              Every patent that appears in an evaluation citation is excluded from training, so the
              model never sees the evaluation relation or its texts.

Writes:
  data/processed/eval_citations.parquet         query_id, relevant_ids
  data/processed/finetune_train_pairs.parquet   anchor_id, positive_id, anchor, positive, code

Usage: python -m finetune.build_pairs [--n-pairs 60000] [--max-group 200] [--per-patent 3]
"""
import argparse
import random
from collections import defaultdict

import pandas as pd

META_PATH = "patents_meta.parquet"
FULL_PATH = "data/processed/patents_full.parquet"
EVAL_PATH = "data/processed/eval_citations.parquet"
TRAIN_PATH = "data/processed/finetune_train_pairs.parquet"


def citation_eval_set(ids: set) -> pd.DataFrame:
    full = pd.read_parquet(FULL_PATH, columns=["publication_number", "cites"])
    rows = []
    for pub, cites in zip(full.publication_number, full.cites):
        if pub not in ids:
            continue
        relevant = sorted({c for c in cites if c in ids and c != pub})
        if relevant:
            rows.append({"query_id": pub, "relevant_ids": relevant})
    return pd.DataFrame(rows)


def cpc_pairs(meta: pd.DataFrame, excluded: set, n_pairs: int, max_group: int, per_patent: int, seed: int):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for pid, codes in zip(meta.patent_id, meta.codes):
        if pid in excluded:
            continue
        # Exact subgroups only (e.g. H04L9/0852); Y-section tags are cross-cutting and too broad
        for code in dict.fromkeys(codes):
            if "/" in code and not code.startswith("Y"):
                groups[code].append(pid)

    usable = [(code, members) for code, members in groups.items() if 2 <= len(members) <= max_group]
    rng.shuffle(usable)

    used = defaultdict(int)
    seen, pairs = set(), []
    # Round-robin over groups so small, specific subgroups contribute as much as big ones
    while len(pairs) < n_pairs and usable:
        next_round = []
        for code, members in usable:
            a, b = rng.sample(members, 2)
            key = tuple(sorted((a, b)))
            if key not in seen and used[a] < per_patent and used[b] < per_patent:
                seen.add(key)
                used[a] += 1
                used[b] += 1
                pairs.append((a, b, code))
                if len(pairs) >= n_pairs:
                    break
            if sum(used[m] < per_patent for m in members) >= 2:
                next_round.append((code, members))
        if len(next_round) == len(usable) and len(pairs) == len(seen) and not next_round:
            break
        usable = next_round
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-pairs", type=int, default=60000)
    parser.add_argument("--max-group", type=int, default=200)
    parser.add_argument("--per-patent", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    meta = pd.read_parquet(META_PATH, columns=["patent_id", "meta", "codes"])
    ids = set(meta.patent_id)

    ev = citation_eval_set(ids)
    ev.to_parquet(EVAL_PATH, index=False)
    excluded = set(ev.query_id) | {r for rel in ev.relevant_ids for r in rel}
    n_rel = int(ev.relevant_ids.map(len).sum())
    print(f"Eval: {len(ev):,} citing patents, {n_rel:,} in-index citations -> {EVAL_PATH}")
    print(f"Excluded from training: {len(excluded):,} patents involved in eval citations")

    text = dict(zip(meta.patent_id, meta.meta))
    pairs = cpc_pairs(meta, excluded, args.n_pairs, args.max_group, args.per_patent, args.seed)
    train = pd.DataFrame(
        [{"anchor_id": a, "positive_id": b, "anchor": text[a], "positive": text[b], "code": c} for a, b, c in pairs]
    )
    train.to_parquet(TRAIN_PATH, index=False)
    print(f"Train: {len(train):,} CPC-subgroup pairs over {train.code.nunique():,} subgroups, "
          f"{len(set(train.anchor_id) | set(train.positive_id)):,} distinct patents -> {TRAIN_PATH}")


if __name__ == "__main__":
    main()
