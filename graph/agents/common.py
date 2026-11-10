"""Settings and helpers shared by all agents."""
import re
import time
from typing import Dict, List


MAX_ROUNDS = 2          # extra search rounds the Critic may request
POOL_PER_QUERY = 100    # FAISS hits kept per query for SDTD re-ranking (best Recall@10 in finetune/evaluate.py: 100-300)
CITATION_SEEDS = 10     # top hits whose citations are followed
PATENT_ID_RE = re.compile(r"^[A-Z]{2}[0-9]{5,12}[A-Z0-9]*$", re.IGNORECASE)
STATUS_RANK = {"not_found": 0, "partial": 1, "disclosed": 2}


def _log(agent: str, message: str, t0: float = None, **data) -> List[Dict]:
    entry = {"agent": agent, "message": message}
    if t0 is not None:
        entry["seconds"] = round(time.time() - t0, 2)
    if data:
        entry["data"] = data
    return [entry]


def _norm_code(code: str) -> str:
    return re.sub(r"\s+", "", str(code).upper())


def _numbered(features: List[str]) -> str:
    return "\n".join(f"{i}. {f}" for i, f in enumerate(features, 1))
