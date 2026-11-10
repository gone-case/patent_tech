"""
Runs the graph and turns its progress into events for the web UI and the CLI:

  {"type": "step",   "node": ..., "agent": ..., "message": ..., "seconds": ..., "data": {...}}
  {"type": "review", "thread_id": ..., "invention": {...}, "codes_source": ...}   run paused for human review
  {"type": "result", "thread_id": ..., ...}                                     final output
  {"type": "error",  "message": ...}
"""
import time
import traceback
import uuid
from functools import lru_cache
from typing import Dict, Iterator, List, Optional

from langgraph.types import Command

from .build import build_graph


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def start_search(query: str, custom_codes: Optional[List[str]] = None, top_k: int = 5,
                 review: bool = False) -> Iterator[Dict]:
    inputs = {"raw_input": query, "custom_codes": custom_codes or None, "top_k": top_k, "review": review}
    yield from _run(inputs, str(uuid.uuid4()))


def resume_search(thread_id: str, edits: Optional[Dict] = None) -> Iterator[Dict]:
    yield from _run(Command(resume=edits or {}), thread_id)


def _run(payload, thread_id: str) -> Iterator[Dict]:
    graph, start = get_graph(), time.time()
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    try:
        for chunk in graph.stream(payload, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node == "__interrupt__":
                    yield {"type": "review", "thread_id": thread_id, **update[0].value}
                    return
                for entry in (update or {}).get("log", []):
                    yield {"type": "step", "node": node, **entry}
        yield {"type": "result", "thread_id": thread_id,
               **final_result(graph.get_state(config).values, time.time() - start)}
    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}


def final_result(state: Dict, elapsed: float) -> Dict:
    top_k = state.get("top_k", 5)
    return {
        "target": state["target"],
        "invention": state["invention"],
        "codes_source": state.get("codes_source"),
        "candidates": [{k: v for k, v in c.items() if k != "hits"} for c in state["ranked"][:top_k]],
        "charts": sorted(state.get("charts", []), key=lambda c: c["rank"]),
        "novelty": state.get("novelty", []),
        "critique": state.get("critique"),
        "rounds": state.get("iteration", 0) + 1,
        "narrative": state.get("narrative", ""),
        "report_markdown": state.get("report", ""),
        "elapsed_seconds": round(elapsed, 2),
    }
