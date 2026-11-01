"""
Pulls patents from Google Patents public data on BigQuery with everything except the description:
title, abstract, claims, IPC + CPC codes, dates, assignees, inventors, citations, family.

Writes:
  data/processed/patents_full.parquet   all fields
  data/processed/index_records.parquet  what the Colab notebook / build_google_patents_index.py embed
                                        (patent_id, title, meta = title + abstract, codes, ipc_codes, cpc_codes, url)

Patents already in patents_full.parquet are kept and new ones are added. Records in an older
index_records.parquet without a patents_full.parquet are re-fetched by ID so every record has all fields.

Usage:
  python pull_google_patents.py --project fopnet-ma --dry-run     # GB that would be scanned, no cost
  python pull_google_patents.py --project fopnet-ma --n 100000
  python pull_google_patents.py --project fopnet-ma --n 100000 --reuse-job <job id>   # existing records, no re-scan
  python pull_google_patents.py --project fopnet-ma --n 100000 --skip-existing         # new sample only

Auth: gcloud application-default credentials. BigQuery bills by bytes scanned; TABLESAMPLE keeps the
random sample cheap, the by-ID lookup scans the needed columns of the whole table.
"""
import argparse
import html
import os

import pandas as pd
from google.cloud import bigquery

FULL_PATH = "data/processed/patents_full.parquet"
RECORDS_PATH = "data/processed/index_records.parquet"

FIELDS = """
    publication_number,
    application_number,
    country_code,
    kind_code,
    family_id,
    publication_date,
    filing_date,
    priority_date,
    grant_date,
    (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1) AS title,
    (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1) AS abstract,
    (SELECT text FROM UNNEST(claims_localized) WHERE language = 'en' LIMIT 1) AS claims,
    ARRAY(SELECT DISTINCT REPLACE(code, ' ', '') FROM UNNEST(ipc)) AS ipc_codes,
    ARRAY(SELECT DISTINCT REPLACE(code, ' ', '') FROM UNNEST(cpc)) AS cpc_codes,
    ARRAY(SELECT name FROM UNNEST(assignee_harmonized)) AS assignees,
    ARRAY(SELECT name FROM UNNEST(inventor_harmonized)) AS inventors,
    ARRAY(SELECT c.publication_number FROM UNNEST(citation) c WHERE c.publication_number != '') AS cites
"""

SAMPLE_SQL = """
SELECT * EXCEPT(rn) FROM (
  SELECT *, ROW_NUMBER() OVER (PARTITION BY family_id ORDER BY publication_number) AS rn
  FROM (
    SELECT {fields}
    FROM `patents-public-data.patents.publications` TABLESAMPLE SYSTEM ({sample_pct} PERCENT)
    WHERE publication_date >= @min_date AND ARRAY_LENGTH(ipc) > 0
  )
  WHERE title IS NOT NULL AND LENGTH(abstract) > 50
)
-- one publication per patent family, so the same invention isn't indexed several times
WHERE rn = 1
ORDER BY FARM_FINGERPRINT(publication_number)
LIMIT @n
"""

LOOKUP_SQL = """
SELECT {fields}
FROM `patents-public-data.patents.publications`
WHERE publication_number IN UNNEST(@ids)
"""


def run(client, sql, params, dry_run, label):
    cfg = bigquery.QueryJobConfig(query_parameters=params, dry_run=True, use_query_cache=False)
    gb = client.query(sql, job_config=cfg).total_bytes_processed / 1e9
    print(f"[{label}] would scan {gb:.1f} GB (~${max(gb / 1000, 0) * 6.25:.2f} beyond free tier)", flush=True)
    if dry_run:
        return None
    job = client.query(sql, job_config=bigquery.QueryJobConfig(query_parameters=params))
    df = job.to_dataframe(create_bqstorage_client=False, progress_bar_type="tqdm")
    print(f"[{label}] {len(df):,} rows, billed {(job.total_bytes_billed or 0) / 1e9:.1f} GB", flush=True)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("title", "abstract", "claims"):
        df[col] = df[col].fillna("").map(lambda s: html.unescape(s).strip().strip('"') if col == "title" else html.unescape(s).strip())
    for col in ("ipc_codes", "cpc_codes", "assignees", "inventors", "cites"):
        df[col] = df[col].map(lambda v: [] if v is None else list(dict.fromkeys(str(x) for x in v)))
    return df


def to_index_records(full: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "patent_id": full.publication_number,
        "title": full.title,
        "meta": full.title + ". " + full.abstract,
        # CPC first (what Google Patents pages show for live queries), then any IPC-only codes
        "codes": [list(dict.fromkeys(c + i)) for c, i in zip(full.cpc_codes, full.ipc_codes)],
        "ipc_codes": full.ipc_codes,
        "cpc_codes": full.cpc_codes,
        "url": "https://patents.google.com/patent/" + full.publication_number.str.replace("-", "", regex=False),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000, help="new patents to pull")
    parser.add_argument("--sample-pct", type=float, default=3.0, help="TABLESAMPLE percent (scan size)")
    parser.add_argument("--min-date", type=int, default=20100101)
    parser.add_argument("--project", default=None, help="GCP project to bill (default: gcloud config)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-job", default=None,
                        help="job ID of an already-run lookup for the existing records; its results are read for free")
    parser.add_argument("--skip-existing", action="store_true",
                        help="don't look up the existing index_records (no lookup cost); keep only the new sample")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)

    if os.path.exists(FULL_PATH):
        full = pd.read_parquet(FULL_PATH)
    elif args.reuse_job:
        # Reading a finished job's results is not billed (results are kept ~24h)
        full = client.get_job(args.reuse_job).to_dataframe(create_bqstorage_client=False, progress_bar_type="tqdm")
        print(f"[reused job {args.reuse_job}] {len(full):,} rows, no new scan", flush=True)
    elif os.path.exists(RECORDS_PATH) and not args.skip_existing:
        ids = pd.read_parquet(RECORDS_PATH, columns=["patent_id"]).patent_id.tolist()
        full = run(client, LOOKUP_SQL.format(fields=FIELDS),
                   [bigquery.ArrayQueryParameter("ids", "STRING", ids)],
                   args.dry_run, f"all fields for {len(ids):,} existing records")
    else:
        full = None

    new = run(client, SAMPLE_SQL.format(fields=FIELDS, sample_pct=args.sample_pct), [
        bigquery.ScalarQueryParameter("min_date", "INT64", args.min_date),
        bigquery.ScalarQueryParameter("n", "INT64", args.n),
    ], args.dry_run, f"sample of {args.n:,} new patents")

    if args.dry_run:
        return

    before = 0 if full is None else len(full)
    full = clean(pd.concat([full, new], ignore_index=True) if full is not None else new)
    full = full.drop_duplicates("publication_number")
    has_codes = full.ipc_codes.map(len).astype(bool) | full.cpc_codes.map(len).astype(bool)
    full = full[(full.abstract.str.len() > 0) & has_codes].reset_index(drop=True)

    os.makedirs(os.path.dirname(FULL_PATH), exist_ok=True)
    full.to_parquet(FULL_PATH, index=False)
    to_index_records(full).to_parquet(RECORDS_PATH, index=False)

    print(f"Added {len(full) - before:,} patents -> {len(full):,} total")
    print(f"  with IPC: {full.ipc_codes.map(len).astype(bool).sum():,} | with CPC: {full.cpc_codes.map(len).astype(bool).sum():,}"
          f" | with English claims: {(full.claims.str.len() > 0).sum():,}")
    print(f"  wrote {FULL_PATH} and {RECORDS_PATH}")


if __name__ == "__main__":
    main()
