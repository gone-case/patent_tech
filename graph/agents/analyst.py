"""
SDTD Analyst agent (tool): re-ranks candidates with (TD + 1) * SD / 2 and measures per-feature coverage.
"""
import time
from typing import Dict

from ..state import AgentState
from ..tools import toolbox
from .common import _log


def score_node(state: AgentState) -> Dict:
    t0, tb = time.time(), toolbox()
    inv = state["invention"]
    ranked = tb.analyst.evaluate_candidates({"codes": inv["cpc_codes"]}, list(state["candidates"].values()))

    coverage = []
    for i, feature in enumerate(inv["features"], 1):
        best = []
        for c in ranked:
            sds = [h["sd"] for h in c.get("hits", []) if h.get("feature") == i]
            if sds:
                best.append((max(sds), c))
        best.sort(key=lambda x: x[0], reverse=True)
        coverage.append({"feature": i, "text": feature,
                         "top": [{"patent_id": c["patent_id"], "title": c.get("title", ""), "sd": round(sd, 4)}
                                 for sd, c in best[:3]]})

    top = ranked[0] if ranked else None
    msg = f"Scored {len(ranked)} candidates with SDTD = (TD + 1) x SD / 2"
    if top:
        msg += f"; best {top['patent_id']} (SD {top['sd_score']:.3f}, TD {top['td_score']:.3f}, SDTD {top['sdtd_score']:.3f})"
    return {"ranked": ranked[:50], "coverage": coverage, "log": _log("SDTD Analyst", msg, t0)}
