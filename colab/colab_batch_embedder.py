"""
High-Speed Patent Embedding & FAISS Indexer (Cloud GPU Script)
Multi-Agent Patent Intelligence Project - Colab Cloud Worker

Run on Google Colab GPU:
    python colab_batch_embedder.py --max_patents 3270 --batch_size 64
"""
import os
import re
import ast
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Patent Embedder & FAISS Index Builder")
    parser.add_argument("--max_patents", type=int, default=50000, help="Max number of patents to embed (default: 50,000)")
    parser.add_argument("--batch_size", type=int, default=128, help="GPU inference batch size")
    parser.add_argument("--dataset", type=str, default="ccdv/patent-classification", help="HuggingFace dataset name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory where FAISS index and metadata will be saved")
    parser.add_argument("--drive_dir", type=str, default=None, help="Optional Google Drive path to also copy outputs to")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 65)
    print(">> PATENT-BERT GPU BATCH EMBEDDER & FAISS INDEX BUILDER <<")
    print("=" * 65)

    import torch
    from transformers import BertModel, BertTokenizer
    from datasets import load_dataset
    import faiss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute Device: {device}")
    if device == "cuda":
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"[*] Available VRAM: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")

    print(f"\n[1/4] Loading real USPTO patents from '{args.dataset}' ({args.split} split)...")
    ds = load_dataset(args.dataset, split=args.split)

    records = []
    texts = []

    print(f"[2/4] Parsing and standardizing up to {args.max_patents} patent documents...")
    limit = min(len(ds), args.max_patents) if args.max_patents > 0 else len(ds)

    for item in tqdm(ds, total=limit, desc="Cleaning patents"):
        title = (item.get("title") or "").strip()
        abstract = (item.get("abstract") or "").strip()
        abstract = re.sub(r'<[^>]+>', ' ', abstract).strip()

        raw_ipcr = item.get("ipcr_labels") or []
        if isinstance(raw_ipcr, str):
            try:
                raw_ipcr = ast.literal_eval(raw_ipcr)
            except Exception:
                raw_ipcr = [raw_ipcr]

        ipc_codes = list(dict.fromkeys([
            c[:4].upper() for c in raw_ipcr if len(c) >= 4 and c[:1].isalpha()
        ]))

        if not ipc_codes:
            main_ipc = (item.get("main_ipcr_label") or "").strip()
            if len(main_ipc) >= 4:
                ipc_codes = [main_ipc[:4].upper()]

        if len(title) > 10 and len(abstract) > 30 and len(ipc_codes) > 0:
            meta_text = f"{title}. {abstract}"
            app_num = str(item.get("application_number", "")).strip()
            patent_id = app_num if app_num else f"PAT_{len(records):06d}"
            records.append({
                "patent_id": patent_id,
                "meta": meta_text,
                "codes": ipc_codes,
            })
            texts.append(meta_text)

        if len(records) >= limit:
            break

    print(f"[*] Successfully prepared {len(records)} clean patent documents for vectorization.")

    print(f"\n[3/4] Initializing 'anferico/bert-for-patents' on {device}...")
    model_name = "anferico/bert-for-patents"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name).to(device)

    # Use FP16 on GPU for ~3x throughput improvement
    if device == "cuda":
        model = model.half()
    model.eval()

    embeddings = []
    batch_size = args.batch_size

    print(f"[*] Generating 1024-dimensional embeddings (Batch Size={batch_size})...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="BERT GPU Embedding"):
            batch = texts[i : i + batch_size]
            tok = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            out = model(**tok)
            # Mean pooling across token representations
            pooled = torch.mean(out.last_hidden_state, dim=1)
            embeddings.append(pooled.float().cpu().numpy())

    embeddings = np.vstack(embeddings).astype(np.float32)
    print(f"[*] Dense vectors computed: Shape = {embeddings.shape}")

    print("\n[4/4] Building normalized FAISS Inner-Product (Cosine Similarity) index...")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    index_path = os.path.join(args.output_dir, "patents_faiss.index")
    meta_path = os.path.join(args.output_dir, "patents_meta.parquet")

    faiss.write_index(index, index_path)
    pd.DataFrame(records).to_parquet(meta_path, index=False)

    print("\n" + "=" * 65)
    print("[SUCCESS] Indexing Pipeline Complete!")
    print(f"[*] Total Indexed Patents: {index.ntotal}")
    print(f"[*] FAISS Index file:     {os.path.abspath(index_path)} ({round(os.path.getsize(index_path)/(1024*1024), 2)} MB)")
    print(f"[*] Metadata Parquet file: {os.path.abspath(meta_path)} ({round(os.path.getsize(meta_path)/(1024*1024), 2)} MB)")
    print("=" * 65)

    if args.drive_dir:
        import shutil
        os.makedirs(args.drive_dir, exist_ok=True)
        drive_index = os.path.join(args.drive_dir, "patents_faiss.index")
        drive_meta = os.path.join(args.drive_dir, "patents_meta.parquet")
        shutil.copyfile(index_path, drive_index)
        shutil.copyfile(meta_path, drive_meta)
        print(f"[*] Also copied to Google Drive: {args.drive_dir}")


if __name__ == "__main__":
    main()
