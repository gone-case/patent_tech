"""
OrchestratorAgent:
The Director of the Multi-Agent Patent Intelligence System.
Manages workflow routing, inter-agent messaging, error recovery, and end-to-end execution.
"""
import re
import os
import time
from typing import Dict, List, Optional
from .harvester_agent import HarvesterAgent
from .retrieval_agent import RetrievalAgent
from .analyst_agent import AnalystAgent
from .colab_worker import ColabWorkerAgent


class OrchestratorAgent:
    def __init__(
        self,
        index_path: str = "patents_faiss.index",
        meta_path: str = "patents_meta.parquet",
        device: Optional[str] = None,
    ):
        self.name = "OrchestratorAgent"
        self.role = "System Director & Workflow Supervisor"
        
        print("\n" + "=" * 65)
        print("[*] [OrchestratorAgent] Initializing Multi-Agent Patent System...")
        print("=" * 65)

        # Initialize subagents
        self.harvester = HarvesterAgent()
        self.colab_worker = ColabWorkerAgent()
        self.retrieval = RetrievalAgent(index_path=index_path, meta_path=meta_path, device=device)
        self.analyst = AnalystAgent()

        print(f"[{self.name}] All 4 specialist agents initialized and active.\n")

    def run_prior_art_search(
        self,
        query: str,
        custom_ipc_codes: Optional[List[str]] = None,
        top_k: int = 5,
        save_report_path: Optional[str] = "prior_art_report.md",
    ) -> Dict:
        """
        End-to-End Multi-Agent Workflow:
        1. HarvesterAgent: Ingests or scrapes target patent from Google Patents.
        2. RetrievalAgent: High-speed FAISS vector retrieval of top semantic matches (SD).
        3. AnalystAgent: Computes Technological Distance (TD) & Hybrid SDTD, generates intelligence report.
        """
        start_time = time.time()
        print("\n" + "-" * 60)
        print(f"[>] [Orchestrator] Starting Prior-Art & Similarity Pipeline for: '{query}'")
        print("-" * 60)

        # -------------------------------------------------------------
        # Step 1: HarvesterAgent
        # -------------------------------------------------------------
        clean_query = query.strip()
        is_file_path = os.path.exists(clean_query)
        is_patent_id = not is_file_path and bool(re.match(r"^[A-Z]{2}[0-9]{5,12}[A-Z0-9]*$", clean_query, re.IGNORECASE))
        
        if is_file_path:
            print(f"[{self.name}] Detected uploaded document path. Routing to HarvesterAgent (Document Parser)...")
            target_patent = self.harvester.parse_document_file(clean_query, custom_codes=custom_ipc_codes)
        elif is_patent_id:
            print(f"[{self.name}] Detected Patent ID format. Routing to HarvesterAgent (Google Patents)...")
            target_patent = self.harvester.fetch_google_patent(clean_query)
        else:
            print(f"[{self.name}] Detected invention description. Routing to HarvesterAgent for structuring...")
            target_patent = self.harvester.parse_patent_input(query, custom_codes=custom_ipc_codes)

        print(f"[{self.name}] Target Title: '{target_patent['title']}'")
        print(f"[{self.name}] Target IPC Codes: {target_patent['codes']}")

        # -------------------------------------------------------------
        # Step 2: RetrievalAgent
        # -------------------------------------------------------------
        print(f"\n[{self.name}] Dispatching RetrievalAgent for vector candidate search (Top {top_k})...")
        # Over-retrieve on abstract similarity, then let SDTD (abstract + IPC) pick the final top_k
        pool_size = max(top_k * 10, 50)
        candidates = self.retrieval.retrieve(target_patent["meta"], top_k=pool_size)
        print(f"[{self.name}] Retrieved {len(candidates)} candidate patents via FAISS vector search.")

        # -------------------------------------------------------------
        # Step 3: AnalystAgent
        # -------------------------------------------------------------
        print(f"\n[{self.name}] Dispatching AnalystAgent for Hybrid SDTD calculation and legal risk evaluation...")
        analyzed_candidates = self.analyst.evaluate_candidates(target_patent, candidates)[:top_k]
        report_markdown = self.analyst.generate_report(target_patent, analyzed_candidates)

        elapsed = time.time() - start_time
        print(f"[{self.name}] Pipeline completed successfully in {elapsed:.2f} seconds.")

        # Save report if requested
        if save_report_path:
            with open(save_report_path, "w", encoding="utf-8") as f:
                f.write(report_markdown)
            print(f"[{self.name}] Full executive report saved to: `{save_report_path}`")

        return {
            "target": target_patent,
            "candidates": analyzed_candidates,
            "report_markdown": report_markdown,
            "elapsed_seconds": round(elapsed, 2),
        }

    def prepare_colab_cloud_worker(self) -> str:
        """Invokes ColabWorkerAgent to provide Google Colab GPU notebook & instructions."""
        print(f"\n[{self.name}] Dispatching ColabWorkerAgent...")
        nb_path = self.colab_worker.generate_colab_notebook()
        instructions = self.colab_worker.get_instructions()
        print(instructions)
        return nb_path
