# Patent Prior-Art Intelligence System (SDTD)

> Multi-agent prior-art search engine: LangGraph agents on Gemini (Vertex AI) plan, search, critique and claim-chart prior art across **118,840 Google Patents**, scored with Patent-BERT + FAISS and the SDTD formula.

Based on the paper: **"A Novel Patent Similarity Measurement Methodology: Semantic Distance and Technological Distance (SDTD)"**
— `Yongmin Yoo`, `Cheonkam Jeong`, `Sanguk Gim`, `Junwon Lee`, `Deaho Seo`

Full write-up: [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md)

---

## What It Does

1. Submit an **invention** (text, a document, or a Google Patent ID)
2. The **Intake agent** extracts its technical features and predicts specific CPC codes
3. The **Planner** writes one search query per feature; the **Retrieval agent** embeds them with **Patent-BERT** (`anferico/bert-for-patents`) and searches the FAISS index, then follows the citations of the top hits
4. Candidates are re-ranked with the **SDTD hybrid formula** (Semantic x Technological Distance)
5. The **Critic** sends weakly covered features back for another search round
6. **Claim Mappers** (in parallel) build a feature-by-feature claim chart for each top candidate
7. The **Reporter** writes a novelty verdict, the closest prior art, and claim-drafting recommendations

A **classic pipeline** (no LLM, ~5 s) is kept as a fast mode.

---

## SDTD Hybrid Formula

| Component | Formula |
|---|---|
| Semantic Distance (SD) | Patent-BERT cosine similarity (mean-pooled, title + abstract) |
| Technological Distance (TD) | Hierarchical IPC/CPC overlap (section 0.25, class 0.5, subclass 0.75, main group 0.9, exact 1.0) |
| **Hybrid Score (SDTD)** | **(TD + 1) * SD / 2** |

---

## Evaluation: citation retrieval

`finetune/evaluate.py` measures prior-art search against real citations: for each of the **796 patents** that cite
another indexed patent (**1,549 citations**), its own vector is searched against all 118,840 patents and we check
where the cited patents rank.

| Ranking | Recall@10 | Hit@10 | Recall@50 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| SD only (Patent-BERT cosine) | 0.216 | 0.270 | 0.351 | 0.168 |
| SDTD (Patent-BERT + hierarchical CPC) | 0.219 | 0.275 | 0.345 | 0.164 |

The index uses mean pooling over real tokens only. An earlier index also averaged padding tokens, which scored
lower (SD Recall@10 0.184, SDTD 0.200); fixing it improved SD by about 17%.

Encoder comparison on 10,000 computer-science patents (CPC G06/G16/H04L, 527 queries, 1,255 citations; recorded in
`finetune/encoder_comparison*.json`, kept for future evaluation; the app uses Patent-BERT):

| Encoder | SD Recall@10 | SDTD Recall@10 |
| :--- | :---: | :---: |
| Patent-BERT (`anferico/bert-for-patents`) | 0.270 | 0.229 |
| PatentSBERTa (SBERT) | 0.270 | 0.296 |
| all-mpnet-base-v2 (SBERT) | 0.346 | 0.349 |

```bash
python -m finetune.build_pairs      # eval citations (+ optional fine-tuning pairs)
python -m finetune.evaluate --label "frozen Patent-BERT"
```

The encoder is the pretrained Patent-BERT, not fine-tuned. `finetune/build_pairs.py` also writes 60,000
CPC-subgroup training pairs (patents in eval citations excluded) for optional contrastive fine-tuning later; a
fine-tuned model directory can be used via `PATENT_ENCODER=<dir>` or a `patents_faiss.index.json` sidecar.

---

## Multi-Agent Architecture (LangGraph)

```
              START
                |
            Harvester          tool agent: routes file / patent ID / text to HarvesterAgent
                |
              Intake           LLM: features, problem, solution, CPC codes
                |  \
                |   Human Review   (optional interrupt: edit features / codes)
                |  /
             Planner  <------+     LLM: one patent-style query per feature (+ the original text)
                |            |
            Retrieval        |     tool agent: Patent-BERT + FAISS per query, citation expansion
                |            |
           SDTD Analyst      |     tool agent: AnalystAgent re-ranking, per-feature coverage
                |            |
              Critic --------+     LLM: weak features go back to the Planner (max 2 extra rounds)
                |
       Send x top_k (parallel)
                |
          Claim Mappers            LLM: claim chart per candidate; evidence quotes are verified
                |
             Reporter              LLM: novelty verdict + recommendations, markdown report
                |
               END
```

| Agent | File | Kind |
|---|---|---|
| Harvester | `graph/agents/harvester.py` → `agents/harvester_agent.py` | tool (scraper, PDF/DOCX parser) |
| Intake | `graph/agents/intake.py` | Gemini, structured output |
| Human Review | `graph/agents/review.py` | LangGraph `interrupt()` + checkpointer |
| Planner | `graph/agents/planner.py` | Gemini, structured output |
| Retrieval | `graph/agents/retrieval.py` → `agents/retrieval_agent.py` | tool (Patent-BERT, FAISS, citations) |
| SDTD Analyst | `graph/agents/analyst.py` → `agents/analyst_agent.py` | tool (deterministic scoring) |
| Critic | `graph/agents/critic.py` | Gemini, conditional edge / loop |
| Claim Mapper | `graph/agents/claim_mapper.py` | Gemini, parallel map step (`Send`) |
| Reporter | `graph/agents/reporter.py` | Gemini |

Every agent logs what it did; the web UI streams these steps live (NDJSON) into an **Agent Trace** panel.

### LLM setup (Vertex AI)
Uses gcloud application-default credentials (`gcloud auth application-default login`) and needs the Vertex AI API.

| Env var | Default |
|---|---|
| `VERTEX_FAST_MODEL` (intake, planner, critic) | `gemini-3.5-flash-lite` |
| `VERTEX_SMART_MODEL` (claim mapping, report) | `gemini-3.8-flash` |
| `VERTEX_LOCATION` | `global` |
| `GOOGLE_CLOUD_PROJECT` | gcloud default project |

A typical agent search takes ~50-80 s (about 8-12 Gemini calls); the classic pipeline ~5 s.

---

## Web App

### Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Visit `http://localhost:5000` and pick the engine: **LangGraph agents** or **Classic pipeline**.

### Input Options
- Type an invention description
- Upload a `.pdf`, `.docx`, `.txt`, `.md` or `.json` document
- Paste a Google Patent ID (e.g. `US10123456B2`)

### Output
- Live agent trace, claim chart per candidate (title, SDTD, shared CPC/IPC, claim similarity), novelty assessment and recommendations
- Top-N prior-art patents ranked by SDTD
- Downloadable `prior_art_report.md`

### API
| Endpoint | |
|---|---|
| `POST /api/agent/search` | LangGraph agents, streamed NDJSON events (`step`, `review`, `result`, `error`) |
| `POST /api/agent/resume` | `{thread_id, features, cpc_codes}` continues a run paused for review |
| `POST /api/search` | classic pipeline, JSON |

---

## Patent Search Database

**118,840 patents** from the Google Patents Public Datasets on BigQuery (`pull_google_patents.py`): published
2010-2026, one publication per patent family, title + abstract + IPC/CPC codes.

| Office | Patents |
|---|---|
| China (CN) | 79,777 |
| United States (US) | 17,029 |
| WIPO (WO) | 5,109 |
| Japan (JP) | 4,810 |
| Europe (EP) | 3,835 |
| Korea (KR) | 3,128 |
| Other | 5,152 |

| File | Size | Description |
|---|---|---|
| `patents_faiss.index` | ~122 MB | 1024-dim Patent-BERT vectors, 8-bit scalar-quantized FAISS inner-product index |
| `patents_meta.parquet` | ~70 MB | ID, title, title + abstract, CPC/IPC codes, URL |
| `data/processed/patents_full.parquet` | ~109 MB | all fields incl. claims (17,029 US patents), citations, assignees |

The embedded text is 23.2M Patent-BERT tokens (~195 per patent; 59 patents exceed the 512-token limit).

### Rebuild the Index
1. `python pull_google_patents.py --project <gcp-project> --dry-run` (shows the BigQuery scan cost), then without `--dry-run`
2. Open `colab/patent_gpu_embedder.ipynb` in Colab with a **T4 GPU**, upload `data/processed/index_records.parquet`, run all
3. Put `patents_faiss.index` and `patents_meta.parquet` in the project root

Or build on CPU: `python build_google_patents_index.py` (slow).

---

## CLI Usage

```bash
# LangGraph agents
python run_agent_graph.py --query "Quantum key distribution transceiver with polarization drift compensation"
python run_agent_graph.py --file sample_invention.md --review     # edit features / codes before searching

# Classic pipeline
python run_multi_agent.py                                          # interactive menu
python run_multi_agent.py --query "US10123456B2" --top_k 5
python run_multi_agent.py --file "my_invention.pdf" --top_k 5
```

---

## Project Structure

```
.
+-- app.py                          Flask web application (agents + classic endpoints)
+-- run_agent_graph.py              CLI for the LangGraph agents
+-- run_multi_agent.py              CLI for the classic pipeline
+-- pull_google_patents.py          BigQuery pull of Google Patents
+-- build_google_patents_index.py   Builds the index locally (CPU)
+-- graph/                          LangGraph workflow
|   +-- state.py                    shared state + structured LLM outputs
|   +-- agents/                     one file per agent (harvester, intake, review, planner, retrieval,
|   |                               analyst, critic, claim_mapper, reporter) + common.py
|   +-- build.py                    graph wiring, checkpointer
|   +-- runner.py                   streaming events for UI / CLI
|   +-- llm.py                      Gemini on Vertex AI
|   +-- tools.py                    the original agents as tools
+-- agents/                         Harvester, Retrieval, Analyst, Orchestrator (classic), ColabWorker
+-- finetune/
|   +-- build_pairs.py              citation eval set + CPC training pairs
|   +-- evaluate.py                 Recall@k / MRR against citations
+-- train_models/sbert.py           Patent-BERT encoder
+-- colab/patent_gpu_embedder.ipynb GPU embedding notebook
+-- static/, templates/             web UI
```
