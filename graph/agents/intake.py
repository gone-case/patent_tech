"""
Intake agent (Gemini): extracts the problem, solution, technical features and CPC codes of the invention.
"""
import time
from typing import Dict

from ..llm import structured
from ..state import AgentState, Invention
from .common import _log, _norm_code


INTAKE_PROMPT = """You are a senior patent analyst preparing a prior-art search.
Read the invention disclosure below and describe its technical substance.

- features: 3 to 6 distinct technical features that together define the invention. Each must be a
  self-contained sentence of at most 20 words, specific enough that an examiner could check whether a prior
  patent discloses it.
  Skip boilerplate and business benefits.
- cpc_codes: the most specific CPC codes (main group or subgroup, e.g. H04L9/0852, H05K7/20936) it
  would be classified in.

Disclosure:
\"\"\"{text}\"\"\""""


def intake_node(state: AgentState) -> Dict:
    t0 = time.time()
    target = state["target"]
    inv: Invention = structured(Invention).invoke(INTAKE_PROMPT.format(text=target["meta"][:12000]))

    invention = inv.model_dump()
    invention["features"] = [f.strip() for f in inv.features if f.strip()][:6]
    invention["predicted_codes"] = list(dict.fromkeys(_norm_code(c) for c in inv.cpc_codes))

    # Real classifications (user-given, scraped from Google Patents, or written in the document) win over predictions
    harvested = [c for c in target["codes"] if c]
    has_real = bool(state.get("custom_codes")) or (target["source"] not in ("User Submission", "Local Fallback")
                                                   and harvested != ["G06F"])
    if has_real and any("/" in c for c in harvested):
        invention["cpc_codes"], codes_source = harvested, "from the input"
    elif has_real:
        # Subclass-only codes (e.g. H04L) leave TD little to compare; add the predicted subgroups
        invention["cpc_codes"] = list(dict.fromkeys(harvested + invention["predicted_codes"]))
        codes_source = "from the input, refined by the Intake agent"
    else:
        invention["cpc_codes"], codes_source = invention["predicted_codes"], "predicted by the Intake agent"

    msg = (f"Extracted {len(invention['features'])} technical features; CPC codes "
           f"{', '.join(invention['cpc_codes'])} ({codes_source})")
    return {"invention": invention, "codes_source": codes_source,
            "log": _log("Intake", msg, t0, features=invention["features"], cpc_codes=invention["cpc_codes"])}


def route_after_intake(state: AgentState) -> str:
    return "review" if state.get("review") else "planner"
