# Patent Prior-Art Intelligence System (SDTD)

> Multi-agent prior art search engine using Patent-BERT + FAISS. Upload an invention idea and instantly find the most similar existing patents from a 50K USPTO database.

Based on the paper: **"A Novel Patent Similarity Measurement Methodology: Semantic Distance and Technological Distance (SDTD)"**
— `Yongmin Yoo`, `Cheonkam Jeong`, `Sanguk Gim`, `Junwon Lee`, `Deaho Seo`

---

## What It Does

1. Submit an **invention description** (text, document, or Google Patent ID)
2. The system embeds it using **Patent-BERT** (`anferico/bert-for-patents`)
3. Searches a **50,000-patent FAISS index** in milliseconds
4. Results are re-ranked using the **SDTD hybrid formula** (Semantic × Technological Distance)
5. Returns a ranked prior-art list with an infringement risk report

---

## Live Demo

Deployed on Render: *(add your Render URL here)*

---

## SDTD Hybrid Formula

| Component | Formula |
|---|---|
| Semantic Distance (SD) | Patent-BERT cosine similarity |
| Technological Distance (TD) | Hierarchical IPC/CPC overlap (section 0.25, class 0.5, subclass 0.75, main group 0.9, exact 1.0) |
| **Hybrid Score (SDTD)** | **(TD + 1) * SD / 2** |

---

## Benchmark Results

Evaluated on **200 real USPTO patent pairs** (HUPD 2007 dataset):

| Model | Pearson | Spearman |
| :---: | :---: | :---: |
| CNN | 0.0750 | 0.0352 |
| LSTM | 0.0541 | 0.0597 |
| Bi-LSTM | 0.1465 | 0.1855 |
| BERT | 0.5077 | 0.5422 |
| **SDTD (Proposed)** | **0.5694** | **0.6445** |

---

## Web App

### Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Visit `http://localhost:5000`

### Input Options
- Type an invention description
- Upload a `.pdf`, `.docx`, `.txt`, or `.md` document
- Paste a Google Patent ID (e.g. `US10123456B2`)

### Output
- Top-N prior-art patents ranked by SDTD hybrid score
- IPC classification overlap analysis
- Downloadable `prior_art_report.md`

---

## Multi-Agent Architecture

```
                               +-------------------------------+
                               |       OrchestratorAgent       |
                               |  (Workflow Director & Router) |
                               +---------------+---------------+
                                               |
         +-----------------+-----------------+------------------+
         v                 v                  v                  v
+----------------+ +------------------+ +---------------+ +---------------+
| HarvesterAgent | | ColabWorkerAgent | | RetrievalAgent| |  AnalystAgent |
| Scrapes Google | | GPU batch embed  | | FAISS <15ms   | | SDTD scoring  |
| Patents & docs | | via Colab T4 GPU | | cosine search | | & report gen  |
+----------------+ +------------------+ +---------------+ +---------------+
```

| Agent | File | Role |
|---|---|---|
| HarvesterAgent | `agents/harvester_agent.py` | Scrapes Google Patents or parses uploaded docs for title, abstract, IPC codes |
| ColabWorkerAgent | `agents/colab_worker.py` | Manages GPU embedding pipeline via Google Colab |
| RetrievalAgent | `agents/retrieval_agent.py` | Sub-15ms FAISS cosine similarity search across 50k patent vectors |
| AnalystAgent | `agents/analyst_agent.py` | Applies SDTD formula, generates prior-art report |

---

## Patent Search Database

50,000 real USPTO patents embedded as 1024-dim Patent-BERT vectors:

| File | Size | Description |
|---|---|---|
| `patents_faiss.index` | ~98 MB | FAISS cosine similarity index |
| `patents_meta.parquet` | ~15 MB | Metadata: ID, title, abstract, IPC codes |

### Rebuild the Index (larger scale)
1. Open `colab/patent_gpu_embedder.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**
3. Run all cells and choose scale (10k / 25k / 50k / 100k patents)
4. Download the two output files into the project root

> 50,000 patents takes ~15 minutes on a free Colab T4 GPU.

---

## CLI Usage

```bash
# Interactive menu
python run_multi_agent.py

# By Google Patent ID
python run_multi_agent.py --query "US10123456B2" --top_k 5

# By invention description
python run_multi_agent.py --query "Quantum key distribution using polarized photons" --ipc "H04L,G06F"

# By uploaded document
python run_multi_agent.py --file "my_invention.pdf" --top_k 5
```

---

## Project Structure

```
.
+-- app.py                          Flask web application
+-- run_multi_agent.py              CLI entrypoint
+-- build_google_patents_index.py   Builds the index locally from data/raw (CPU)
+-- requirements.txt
+-- Procfile                        Render.com config
+-- render.yaml
+-- agents/
|   +-- orchestrator.py
|   +-- harvester_agent.py
|   +-- retrieval_agent.py
|   +-- analyst_agent.py
|   +-- colab_worker.py
+-- train_models/
|   +-- sbert.py                    Patent-BERT encoder
+-- colab/
|   +-- patent_gpu_embedder.ipynb   GPU Colab notebook
+-- static/                         CSS and JS
+-- templates/                      HTML
+-- data/processed/index_records.parquet   Google Patents records (title, abstract, CPC) for Colab
+-- patents_faiss.index             Patent-BERT vector index (title + abstract)
+-- patents_meta.parquet            Patent metadata incl. CPC codes
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `flask`, `transformers`, `torch`, `faiss-cpu`, `pandas`, `datasets`, `python-docx`, `pypdf`
