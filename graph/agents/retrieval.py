"""
Retrieval agent (tool): Patent-BERT + FAISS search for every query, then follows the citations of the top hits.
"""
import time
from typing import Dict

from ..state import AgentState, merge_candidates
from ..tools import toolbox
from .common import CITATION_SEEDS, POOL_PER_QUERY, _log


def retrieve_node(state: AgentState) -> Dict:
    t0, tb = time.time(), toolbox()
    queries, rnd = state["queries"], state.get("iteration", 0)

    with tb.encode_lock:
        vecs = tb.retrieval.embed_queries([q["text"] for q in queries])
    results = tb.retrieval.search_vectors(vecs, POOL_PER_QUERY)

    found: Dict[str, Dict] = {}
    for qi, (q, cands) in enumerate(zip(queries, results)):
        for c in cands:
            hit = {"round": rnd, "query": qi, "feature": q.get("feature"), "sd": round(c["sd_score"], 4)}
            found = merge_candidates(found, {c["patent_id"]: {**c, "hits": [hit], "found_via": "vector search"}})
    new = {pid: c for pid, c in found.items() if pid not in state.get("candidates", {})}

    # Citation expansion: patents cited by the strongest hits are prior art by an examiner's judgement
    cited_msg = ""
    if rnd == 0 and tb.store.available:
        seeds = sorted(found.values(), key=lambda c: c["sd_score"], reverse=True)[:CITATION_SEEDS]
        cited_by = {}
        for s in seeds:
            for cid in tb.store.cites.get(s["patent_id"], []):
                if cid not in found:
                    cited_by.setdefault(cid, s["patent_id"])
        cited = tb.retrieval.similarity_to(vecs[0], list(cited_by))
        for c in cited:
            c.update(hits=[{"round": rnd, "query": 0, "feature": None, "sd": round(c["sd_score"], 4)}],
                     found_via=f"cited by {cited_by[c['patent_id']]}")
            found[c["patent_id"]] = c
        cited_msg = f", plus {len(cited)} patents cited by the top {len(seeds)} hits"
    elif rnd == 0:
        cited_msg = " (citation data not available, so citations were not followed)"

    msg = (f"Embedded {len(queries)} queries with Patent-BERT and searched {tb.retrieval.index.ntotal:,} patents: "
           f"{len(new)} new candidates{cited_msg}")
    return {"candidates": found, "log": _log("Retrieval", msg, t0)}
