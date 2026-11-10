"""
Harvester agent (tool): routes the input - file, patent ID or free text - to the HarvesterAgent.
"""
import os
import time
from typing import Dict

from ..state import AgentState
from ..tools import toolbox
from .common import PATENT_ID_RE, _log


def harvester_node(state: AgentState) -> Dict:
    t0, tb = time.time(), toolbox()
    query, custom = state["raw_input"].strip(), state.get("custom_codes")

    if os.path.exists(query):
        target, route = tb.harvester.parse_document_file(query, custom_codes=custom), "the document parser"
    elif PATENT_ID_RE.match(query):
        target, route = tb.harvester.fetch_google_patent(query), "the Google Patents scraper"
    else:
        target, route = tb.harvester.parse_patent_input(query, custom_codes=custom), "text structuring"

    msg = f"Routed input to {route}: \"{target['title'][:90]}\" ({target['source']})"
    if target["source"] == "Local Fallback":
        msg += ". Warning: Google Patents could not be reached, so only a placeholder record is available"
    return {"target": target, "iteration": 0, "log": _log("Harvester", msg, t0, codes=target["codes"])}
