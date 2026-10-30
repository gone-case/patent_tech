"""
ColabWorkerAgent:
Manages distributed cloud execution, Google Colab GPU embedding jobs,
and synchronization of FAISS indices built on Google Drive or HuggingFace.
"""
import os
import json
from typing import Dict, Optional


class ColabWorkerAgent:
    def __init__(self, colab_dir: str = "colab"):
        self.name = "ColabWorkerAgent"
        self.role = "Cloud GPU Worker & Colab Pipeline Coordinator"
        self.colab_dir = colab_dir
        os.makedirs(self.colab_dir, exist_ok=True)
        self.notebook_path = os.path.join(self.colab_dir, "patent_gpu_embedder.ipynb")
        self.script_path = os.path.join(self.colab_dir, "colab_batch_embedder.py")
        self._ensure_colab_artifacts()

    def _ensure_colab_artifacts(self):
        """Generates ready-to-run Google Colab artifacts if not already present."""
        if not os.path.exists(self.notebook_path):
            self.generate_colab_notebook()

    def generate_colab_notebook(self) -> str:
        """
        Creates a ready-to-upload Google Colab Jupyter Notebook (.ipynb)
        configured for free GPU (T4/V100/A100) to batch embed millions of patents.
        """
        cells = [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🚀 High-Speed Patent Embedding & FAISS Indexer (Google Colab GPU)\n",
                    "### Multi-Agent Patent Intelligence Project - Cloud Worker\n",
                    "This notebook offloads the heavy BERT embedding generation across thousands or millions of patents using Colab's free GPU.\n",
                    "1. Loads `anferico/bert-for-patents` with PyTorch FP16.\n",
                    "2. Streams patents from HUPD (Harvard USPTO Dataset) or uploads your custom dataset.\n",
                    "3. Builds a normalized **FAISS vector index** (`patents_faiss.index`) and metadata (`patents_meta.parquet`).\n",
                    "4. Saves directly to your Google Drive for download back into your local multi-agent project."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 1: Install dependencies & check GPU\n",
                    "!nvidia-smi\n",
                    "!pip install -q transformers torch faiss-gpu datasets pyarrow pandas tqdm"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 2: (Optional) Mount Google Drive to save the resulting index permanently\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n",
                    "OUTPUT_DIR = '/content/drive/MyDrive/patent_faiss_index'\n",
                    "import os; os.makedirs(OUTPUT_DIR, exist_ok=True)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 3: Stream Real Patent Dataset from HUPD (Harvard USPTO Dataset)\n",
                    "from datasets import load_dataset\n",
                    "import pandas as pd\n",
                    "import re, ast\n",
                    "from tqdm import tqdm\n",
                    "\n",
                    "print('Streaming patent records...')\n",
                    "# Loads the standard HUPD train split\n",
                    "ds = load_dataset('late-interacters/hupd-2007-v2-small', split='train')\n",
                    "\n",
                    "records = []\n",
                    "texts = []\n",
                    "MAX_PATENTS = 50000  # Adjust as needed (e.g. 10,000 to 1,000,000)\n",
                    "\n",
                    "for item in tqdm(ds, total=min(len(ds), MAX_PATENTS)):\n",
                    "    title = (item.get('title') or '').strip()\n",
                    "    abstract = (item.get('abstract') or '').strip()\n",
                    "    abstract = re.sub(r'<[^>]+>', ' ', abstract).strip()\n",
                    "    \n",
                    "    raw_ipcr = item.get('ipcr_labels') or []\n",
                    "    if isinstance(raw_ipcr, str):\n",
                    "        try:\n",
                    "            raw_ipcr = ast.literal_eval(raw_ipcr)\n",
                    "        except Exception:\n",
                    "            raw_ipcr = [raw_ipcr]\n",
                    "    ipc_codes = list(dict.fromkeys([c[:4].upper() for c in raw_ipcr if len(c) >= 4 and c[:1].isalpha()]))\n",
                    "    \n",
                    "    if len(title) > 10 and len(abstract) > 30:\n",
                    "        meta_text = f'{title}. {abstract}'\n",
                    "        records.append({\n",
                    "            'patent_id': str(item.get('application_number', f'PAT_{len(records):06d}')),\n",
                    "            'meta': meta_text,\n",
                    "            'codes': ipc_codes or ['G06F'],\n",
                    "        })\n",
                    "        texts.append(meta_text)\n",
                    "    if len(records) >= MAX_PATENTS:\n",
                    "        break\n",
                    "\n",
                    "print(f'Prepared {len(records)} patent documents for GPU embedding.')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 4: Batch GPU Inference with Patent-BERT (FP16)\n",
                    "import torch\n",
                    "from transformers import BertModel, BertTokenizer\n",
                    "import numpy as np\n",
                    "\n",
                    "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
                    "print(f'Using compute device: {device}')\n",
                    "\n",
                    "model_name = 'anferico/bert-for-patents'\n",
                    "tokenizer = BertTokenizer.from_pretrained(model_name)\n",
                    "model = BertModel.from_pretrained(model_name).to(device).half().eval()  # FP16 for 3x speedup\n",
                    "\n",
                    "BATCH_SIZE = 64\n",
                    "embeddings = []\n",
                    "\n",
                    "with torch.no_grad():\n",
                    "    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc='GPU Embedding'):\n",
                    "        batch = texts[i : i + BATCH_SIZE]\n",
                    "        tok = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)\n",
                    "        out = model(**tok)\n",
                    "        pooled = torch.mean(out.last_hidden_state, dim=1)\n",
                    "        embeddings.append(pooled.float().cpu().numpy())\n",
                    "\n",
                    "embeddings = np.vstack(embeddings)\n",
                    "print('Embeddings shape:', embeddings.shape)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Step 5: Build FAISS Vector Index & Save to Drive / Colab\n",
                    "import faiss\n",
                    "\n",
                    "faiss.normalize_L2(embeddings)\n",
                    "dim = embeddings.shape[1]\n",
                    "index = faiss.IndexFlatIP(dim)\n",
                    "index.add(embeddings)\n",
                    "\n",
                    "index_file = os.path.join(OUTPUT_DIR, 'patents_faiss.index')\n",
                    "meta_file = os.path.join(OUTPUT_DIR, 'patents_meta.parquet')\n",
                    "\n",
                    "faiss.write_index(index, index_file)\n",
                    "pd.DataFrame(records).to_parquet(meta_file, index=False)\n",
                    "\n",
                    "print(f'Successfully built and saved FAISS index with {index.ntotal} vectors!')\n",
                    "print(f'1. Index file: {index_file}')\n",
                    "print(f'2. Metadata file: {meta_file}')\n",
                    "print('Download these 2 files into your local project root folder to query millions of patents instantly!')"
                ]
            }
        ]

        notebook = {
            "cells": cells,
            "metadata": {
                "accelerator": "GPU",
                "colab": {"name": "patent_gpu_embedder.ipynb", "provenance": []},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 0,
        }

        with open(self.notebook_path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=2)

        print(f"[{self.name}] Generated Google Colab GPU notebook: {self.notebook_path}")
        return self.notebook_path

    def get_instructions(self) -> str:
        return (
            f"Colab Instructions:\n"
            f"1. Open https://colab.research.google.com\n"
            f"2. Upload '{self.notebook_path}'\n"
            f"3. Enable GPU (Runtime -> Change runtime type -> T4 GPU)\n"
            f"4. Run All. It will generate `patents_faiss.index` and `patents_meta.parquet`.\n"
            f"5. Place those 2 files in this project directory to search millions of patents locally in milliseconds."
        )
