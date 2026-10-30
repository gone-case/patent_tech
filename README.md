# A-Novel-Patent-Similarity-Measurement-Methodology

Document Similarity Prediction

  * [A-Novel-Patent-Similarity-Measurement-Methodology: Semantic Distance and Technological Distance](주소추가)
  
    * `Yongmin Yoo`, `Cheonkam Jeong`, `Sanguk Gim`, `Junwon Lee`, `Deaho Seo`

-------------------------------------------------

## Dataset

  * We used Google patent, randomly extract 420 pairs patents from 2019 to 2020, with which we perform expert validation
  * The expert panel whom has expertise in Data Analytics, Data Mining, and Artificial Intelligence assesses how semantically similar two patents
  * We used the score of law expert when there was a large difference in scores between experts.
  
  |    Expert    | Frequency  |
  | :------: | :---: |
  |    Expert panel    | 335  |
  |    Law Expert    | 85  |
  |    Total    | 420  |
  
  
-------------------------------------------------

## Model Structure

  * Semantic Similarity
    
    * We used [Patent Bert](https://huggingface.co/anferico/bert-for-patents)

  * Technical Similarity
    
    * $Intersection_{A,B}=Patent_{A} \cap Patent_{B}$
    * $Union_{A,B}=Patent_{A} \cup Patent_{B}$
    * $TD_{A,B}={Intersection_{A,B} \over Union_{A,B}}$

  * Hybrid Similarity

    * $SDTD_{A,B}={(TD+1) \cdot SD \over 2}$
  
 -------------------------------------------------
 
 ## Result
 
  |    Model    | Pearson  | Spearman  |
  | :------: | :---: | :-----: |
  |  CNN              |  0.074951      | 0.035229      |
  |  LSTM              | 0.05409      | 0.05967      |
  |  Bi-LSTM            | 0.14646      | 0.185534      |
  |  BERT         | 0.507742      | 0.542219      |
  |  **Proposed model** | **0.56935**  | **0.644496**  |

-------------------------------------------------

## Multi-Agent Architecture (Scaling to Millions of Patents)

To scale beyond small benchmark evaluation to **millions of patents** across Google Patents and USPTO, the project includes a production-grade **Multi-Agent System** with FAISS vector indexing and Google Colab GPU offloading.

```
                               ┌─────────────────────────────────┐
                               │       OrchestratorAgent         │
                               │  (Workflow Director & Router)   │
                               └────────────────┬────────────────┘
                                                │
         ┌─────────────────────────┬────────────┴────────────┬─────────────────────────┐
         ▼                         ▼                         ▼                         ▼
┌──────────────────┐    ┌────────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  HarvesterAgent  │    │  ColabWorkerAgent  │    │  RetrievalAgent  │    │   AnalystAgent   │
│                  │    │                    │    │                  │    │                  │
│ Live Google      │    │ Free Cloud GPU     │    │ FAISS sub-15ms   │    │ Evaluates SDTD   │
│ Patents scraper  │    │ batch embedding    │    │ vector search    │    │ formula, re-ranks│
│ & IPC parser     │    │ (1M+ patents)      │    │ dense index      │    │ prior-art report │
└──────────────────┘    └────────────────────┘    └──────────────────┘    └──────────────────┘
```

### 1. The 4 Specialized Agents
* **`HarvesterAgent`** (`agents/harvester_agent.py`): Scrapes patent title, abstract, and IPC/CPC classification trees directly from live Google Patents URLs (`patents.google.com/patent/{id}/en`) or parses custom user invention descriptions.
* **`ColabWorkerAgent`** (`agents/colab_worker.py`): Exports a ready-to-use Google Colab notebook (`colab/patent_gpu_embedder.ipynb`) to leverage cloud T4/A100 GPUs for FP16 batch embedding of 1,000,000+ patents into a FAISS index.
* **`RetrievalAgent`** (`agents/retrieval_agent.py`): Performs sub-15 millisecond vector similarity retrieval using a normalized inner-product FAISS index (`patents_faiss.index`).
* **`AnalystAgent`** (`agents/analyst_agent.py`): Calculates Technological Distance ($TD = \text{Intersection} / \text{Union}$) on IPC codes, applies the hybrid formula $SDTD = \frac{(TD+1)\cdot SD}{2}$, and generates an executive Prior-Art & Infringement Risk report (`prior_art_report.md`).

---

## How to Run the Multi-Agent System

### Interactive CLI Menu:
```powershell
python run_multi_agent.py
```

### Search by Google Patent ID (e.g., US10123456B2):
```powershell
python run_multi_agent.py --query "US10123456B2" --top_k 5
```

### Search by Custom Invention Description:
```powershell
python run_multi_agent.py --query "Quantum key distribution using polarized photons for cryptographic communication" --ipc "H04L,G06F"
```

### Search by Uploaded Document (.pdf, .docx, .txt, .md, .json):
```powershell
python run_multi_agent.py --file "my_invention.pdf" --top_k 5
# Or with a Word DOCX or text draft:
python run_multi_agent.py --file "patent_draft.docx"
```
*`HarvesterAgent` automatically extracts title, description, and auto-detects any mentioned IPC classification codes directly from the document.*

### Export Google Colab Cloud GPU Worker:
```powershell
python run_multi_agent.py --colab
```
Open [Google Colab](https://colab.research.google.com), upload `colab/patent_gpu_embedder.ipynb`, enable GPU, and run all to embed millions of patents. Download the generated `patents_faiss.index` and `patents_meta.parquet` files directly into this project root.

