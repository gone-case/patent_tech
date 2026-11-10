"""
The agents of the LangGraph prior-art workflow, one module each. Every agent returns a "log" entry so its
work is visible in the UI / CLI trace.

  harvester.py     Harvester     tool    routes the input (file / patent ID / text) to HarvesterAgent
  intake.py        Intake        Gemini  extracts problem, solution, technical features, CPC codes
  review.py        Human Review  pause   the user can edit features / codes before searching (optional)
  planner.py       Planner       Gemini  one search query per feature (+ the original text)
  retrieval.py     Retrieval     tool    Patent-BERT + FAISS per query, then follows citations of the top hits
  analyst.py       SDTD Analyst  tool    (TD + 1) * SD / 2 re-ranking, per-feature coverage
  critic.py        Critic        Gemini  sends weak features back to the Planner (max MAX_ROUNDS)
  claim_mapper.py  Claim Mapper  Gemini  claim chart per top candidate, run in parallel
  reporter.py      Reporter      Gemini  novelty verdict, recommendations, markdown report
"""
from .analyst import score_node
from .claim_mapper import claim_mapper_node
from .critic import critic_node, route_after_critic
from .harvester import harvester_node
from .intake import intake_node, route_after_intake
from .planner import planner_node
from .reporter import reporter_node
from .retrieval import retrieve_node
from .review import review_node

__all__ = [
    "harvester_node", "intake_node", "route_after_intake", "review_node", "planner_node", "retrieve_node",
    "score_node", "critic_node", "route_after_critic", "claim_mapper_node", "reporter_node",
]
