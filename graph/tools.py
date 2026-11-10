"""
The original specialist agents, shared by the graph nodes as tools:
HarvesterAgent (ingestion), RetrievalAgent (Patent-BERT + FAISS), AnalystAgent (SDTD),
plus PatentStore for the citation lists and claims that only patents_full.parquet has.
"""
import os
import threading
from functools import lru_cache
from typing import Dict, List

import pandas as pd

from agents.analyst_agent import AnalystAgent
from agents.harvester_agent import HarvesterAgent
from agents.retrieval_agent import RetrievalAgent

FULL_PATH = "data/processed/patents_full.parquet"


class PatentStore:
    def __init__(self, path: str = FULL_PATH):
        self.cites: Dict[str, List[str]] = {}
        self.claims: Dict[str, str] = {}
        if os.path.exists(path):
            df = pd.read_parquet(path, columns=["publication_number", "cites", "claims"])
            self.cites = {p: list(c) for p, c in zip(df.publication_number, df.cites) if len(c)}
            self.claims = {p: c for p, c in zip(df.publication_number, df.claims) if c}

    @property
    def available(self) -> bool:
        return bool(self.cites)


class Toolbox:
    def __init__(self):
        self.harvester = HarvesterAgent()
        self.retrieval = RetrievalAgent()
        self.analyst = AnalystAgent()
        self.store = PatentStore()
        # Patent-BERT runs on CPU; one forward pass at a time
        self.encode_lock = threading.Lock()


@lru_cache(maxsize=1)
def toolbox() -> Toolbox:
    return Toolbox()
