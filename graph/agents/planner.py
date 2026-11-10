"""
Planner agent (Gemini): writes one patent-style search query per feature, or new queries for weak features.
"""
import time
from typing import Dict

from ..llm import structured
from ..state import AgentState, SearchPlan
from .common import _log, _numbered


PLANNER_PROMPT = """You are a patent search strategist. The search engine embeds each query with Patent-BERT
and compares it with the title + abstract of indexed patents, so write every query as 2-3 sentences that read
like the title and abstract of a prior patent disclosing that feature. Use the vocabulary patent drafters use
and generic terms for the technology rather than the inventor's brand names.

Invention: {title}
Problem: {problem}
Solution: {solution}
Features:
{features}
{retry}
Write {what}."""

RETRY_BLOCK = """
A previous search round was judged insufficient.
Critic assessment: {assessment}
Weak features: {weak}
Critic advice: {advice}
Queries already tried:
{tried}
"""


def planner_node(state: AgentState) -> Dict:
    t0 = time.time()
    inv, critique = state["invention"], state.get("critique")
    retry = ""
    if critique:
        retry = RETRY_BLOCK.format(assessment=critique["assessment"], weak=critique["weak_features"],
                                   advice=critique["advice"],
                                   tried="\n".join(f"- {q['text']}" for q in state.get("queries", [])))
        what = "one or two new queries for each weak feature, approaching it from a different angle"
    else:
        what = "one query per feature, plus one query describing the whole invention"

    plan: SearchPlan = structured(SearchPlan).invoke(PLANNER_PROMPT.format(
        title=inv["title"], problem=inv["problem"], solution=inv["solution"],
        features=_numbered(inv["features"]), retry=retry, what=what))
    queries = [q.model_dump() for q in plan.queries][:8]
    if not critique:
        # Always search the original text too, so the agents never find less than the classic pipeline
        queries.insert(0, {"feature": None, "text": state["target"]["meta"][:4000], "original": True})

    rnd = state.get("iteration", 0)
    return {"queries": queries,
            "log": _log("Planner", f"Round {rnd + 1}: {plan.reasoning}", t0,
                        queries=[{"feature": q["feature"], "text": q["text"][:300]} for q in queries])}
