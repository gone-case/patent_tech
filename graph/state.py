"""
Shared state of the LangGraph prior-art workflow, plus the structured outputs the LLM agents return.
"""
import operator
from typing import Annotated, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- LLM output schemas

class Invention(BaseModel):
    title: str = Field(description="Short descriptive title of the invention")
    problem: str = Field(description="Technical problem the invention solves, 1-2 sentences")
    solution: str = Field(description="How the invention solves it, 1-3 sentences")
    features: List[str] = Field(
        description="3-6 distinct technical features that together define the invention, "
                    "each one self-contained sentence of at most 20 words that a patent examiner could check against prior art"
    )
    cpc_codes: List[str] = Field(
        description="2-6 most specific CPC codes (main group or subgroup level, e.g. H04L9/0852) "
                    "this invention would be classified in"
    )


class SearchQuery(BaseModel):
    feature: Optional[int] = Field(
        description="1-based number of the feature this query targets, or null for a query about the whole invention"
    )
    text: str = Field(
        description="2-3 sentences written like a patent title and abstract that would disclose this feature"
    )


class SearchPlan(BaseModel):
    reasoning: str = Field(description="One or two sentences on the search strategy")
    queries: List[SearchQuery]


class Critique(BaseModel):
    assessment: str = Field(description="1-3 sentences on how well the retrieved patents cover the invention")
    weak_features: List[int] = Field(
        description="1-based numbers of features whose retrieved patents are off-topic or only loosely related"
    )
    retry: bool = Field(description="true if another search round for the weak features is likely to find better prior art")
    advice: str = Field(description="How the next queries should differ (terminology, angle, scope); empty if retry is false")


class ChartRow(BaseModel):
    feature: int = Field(description="1-based feature number")
    status: Literal["disclosed", "partial", "not_found"]
    evidence: str = Field(description="Shortest verbatim quote (max 25 words) from the candidate text proving the status; "
                                      "empty if not_found")
    explanation: str = Field(description="One sentence, max 20 words")


class ClaimChart(BaseModel):
    rows: List[ChartRow]
    summary: str = Field(description="One sentence, max 30 words, on how close this patent is to the invention")


# ---------------------------------------------------------------- graph state

def merge_candidates(old: Optional[Dict[str, Dict]], new: Optional[Dict[str, Dict]]) -> Dict[str, Dict]:
    """Candidates found across queries and search rounds: keep the best SD and every query that hit."""
    merged = dict(old or {})
    for pid, cand in (new or {}).items():
        if pid not in merged:
            merged[pid] = cand
            continue
        prev = merged[pid]
        best = cand if cand["sd_score"] > prev["sd_score"] else prev
        merged[pid] = {**best, "hits": prev.get("hits", []) + cand.get("hits", []),
                       "found_via": prev.get("found_via", best.get("found_via"))}
    return merged


class AgentState(TypedDict, total=False):
    # input
    raw_input: str
    custom_codes: Optional[List[str]]
    top_k: int
    review: bool
    # Harvester / Intake
    target: Dict
    invention: Dict
    codes_source: str
    # Planner / Retrieval / Critic loop
    queries: List[Dict]
    candidates: Annotated[Dict[str, Dict], merge_candidates]
    ranked: List[Dict]
    coverage: List[Dict]
    critique: Dict
    iteration: int
    # Claim mapping / Report
    charts: Annotated[List[Dict], operator.add]
    novelty: List[Dict]
    narrative: str
    report: str
    # visible trace of what every agent did
    log: Annotated[List[Dict], operator.add]
