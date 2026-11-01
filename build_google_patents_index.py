"""
Builds the FAISS index + metadata used by RetrievalAgent from the Google Patents BigQuery pull in data/raw.

  data/raw/text/*.parquet  -> title, abstract, url   (Patent-BERT input)
  data/raw/meta/*.parquet  -> full CPC codes          (Technological Distance input)
These are joined once into data/processed/index_records.parquet, which is reused if present.

Usage: python build_google_patents_index.py [--limit N] [--batch-size 16]
Writes patents_faiss.index and patents_meta.parquet (previous files are kept as *.bak).
"""
import argparse
import html
import os
import shutil
import time

import faiss
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from agents.retrieval_agent import RetrievalAgent

INDEX_PATH = "patents_faiss.index"
META_PATH = "patents_meta.parquet"
RECORDS_PATH = "data/processed/index_records.parquet"


def load_records(limit=None):
    if os.path.exists(RECORDS_PATH):
        df = pd.read_parquet(RECORDS_PATH)
        records = [dict(r, codes=list(r["codes"])) for r in df.to_dict(orient="records")]
        return records[:limit] if limit else records

    text = ds.dataset("data/raw/text", format="parquet").to_table(
        columns=["publication_number", "title", "abstract", "url"]
    ).to_pandas()
    pubs = text.publication_number.tolist()
    meta = ds.dataset("data/raw/meta", format="parquet").to_table(
        columns=["publication_number", "cpc"],
        filter=ds.field("publication_number").isin(pubs),
    ).to_pandas().drop_duplicates("publication_number")

    df = text.merge(meta, on="publication_number")
    records = []
    for row in df.itertuples():
        title = html.unescape(row.title or "").strip().strip('"')
        abstract = html.unescape(row.abstract or "").strip()
        codes = list(dict.fromkeys(str(c).replace(" ", "") for c in (row.cpc if row.cpc is not None else [])))
        if not abstract or not codes:
            continue
        records.append({
            "patent_id": row.publication_number,
            "title": title,
            "meta": f"{title}. {abstract}",
            "codes": codes,
            "url": row.url,
        })
        if limit and len(records) >= limit:
            break
    if not limit:
        pd.DataFrame(records).to_parquet(RECORDS_PATH, index=False)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    records = load_records(args.limit)
    print(f"Loaded {len(records):,} Google Patents records with CPC codes.")

    agent = RetrievalAgent()
    texts = [r["meta"] for r in records]
    chunks, start = [], time.time()
    step = args.batch_size * 20
    for i in range(0, len(texts), step):
        chunks.append(agent.encode_texts(texts[i:i + step], batch_size=args.batch_size))
        done = min(i + step, len(texts))
        rate = done / (time.time() - start)
        print(f"  embedded {done:,}/{len(texts):,} ({rate:.1f}/s, ~{(len(texts) - done) / rate / 60:.0f} min left)", flush=True)

    embeddings = np.vstack(chunks).astype(np.float32)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    for path in (INDEX_PATH, META_PATH):
        if os.path.exists(path):
            shutil.copy2(path, path + ".bak")
    faiss.write_index(index, INDEX_PATH)
    pd.DataFrame(records).to_parquet(META_PATH, index=False)
    print(f"Wrote {INDEX_PATH} ({index.ntotal:,} vectors) and {META_PATH}.")


if __name__ == "__main__":
    main()
