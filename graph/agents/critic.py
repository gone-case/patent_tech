"""
Critic agent (Gemini): judges coverage and either loops weak features back to the Planner or fans out to the Claim Mappers.
"""
import time
from typing import Dict

from langgraph.types import Send

from ..llm import structured
from ..state import AgentState, Critique
from .common import MAX_ROUNDS, _log


CRITIC_PROMPT = """You review the results of a patent prior-art search before they go to the claim analysts.
For each feature you see the closest patents found by queries targeting it (title and cosine similarity).
Judge relevance from the titles and the technology, not only the numbers: a high similarity with an
off-topic title is a miss. Mark a feature weak when none of its patents address that technical feature.
Ask for another round only when better-targeted queries are likely to find closer prior art.

Invention: {title}
{coverage}

Overall top candidates by SDTD:
{top}

This was search round {round} of at most {max_rounds}."""


def critic_node(state: AgentState) -> Dict:
    t0 = time.time()
    rnd = state.get("iteration", 0)
    coverage = "\n".join(
        f"Feature {c['feature']}: {c['text']}\n" +
        ("\n".join(f"   - {t['title']} (similarity {t['sd']:.3f})" for t in c["top"]) or "   - nothing retrieved")
        for c in state["coverage"])
    top = "\n".join(f"- {c.get('title', c['patent_id'])} (SDTD {c['sdtd_score']:.3f})" for c in state["ranked"][:5])

    crit: Critique = structured(Critique).invoke(CRITIC_PROMPT.format(
        title=state["invention"]["title"], coverage=coverage, top=top, round=rnd + 1, max_rounds=MAX_ROUNDS + 1))
    retry = bool(crit.retry and crit.weak_features and rnd < MAX_ROUNDS)

    decision = (f"sending features {crit.weak_features} back to the Planner" if retry
                else "coverage accepted, handing the top candidates to the Claim Mappers")
    return {"critique": {**crit.model_dump(), "retry": retry},
            "iteration": rnd + 1 if retry else rnd,
            "log": _log("Critic", f"{crit.assessment} Decision: {decision}.", t0, weak_features=crit.weak_features)}


def route_after_critic(state: AgentState):
    if state["critique"]["retry"]:
        return "planner"
    top = state["ranked"][: state.get("top_k", 5)]
    if not top:
        return "reporter"
    # Map step: one Claim Mapper per candidate, run in parallel
    return [Send("claim_mapper", {"invention": state["invention"], "candidate": c, "rank": i})
            for i, c in enumerate(top, 1)]
