"""
Human Review agent (LangGraph interrupt): pauses the run so the user can edit the features and CPC codes.
"""
import time
from typing import Dict

from langgraph.types import interrupt

from ..state import AgentState
from .common import _log, _norm_code


def review_node(state: AgentState) -> Dict:
    t0 = time.time()
    edited = interrupt({"invention": state["invention"], "codes_source": state["codes_source"]})

    invention = dict(state["invention"])
    edited = edited or {}
    features = [f.strip() for f in edited.get("features", invention["features"]) if f and f.strip()]
    codes = [_norm_code(c) for c in edited.get("cpc_codes", invention["cpc_codes"]) if c and c.strip()]
    changed = features != invention["features"] or codes != invention["cpc_codes"]
    invention.update(features=features or invention["features"], cpc_codes=codes or invention["cpc_codes"])

    return {"invention": invention,
            "codes_source": "edited by the user" if changed else state["codes_source"],
            "log": _log("Human Review", "User edited the features / codes" if changed else "User approved the analysis", t0)}
