"""
Claim Mapper agent (Gemini, one instance per top candidate, run in parallel): builds a feature-by-feature claim chart
and checks that every evidence quote really occurs in the candidate text.
"""
import re
import time
from typing import Dict

from ..llm import structured
from ..state import ClaimChart
from ..tools import toolbox
from .common import STATUS_RANK, _log, _numbered


MAPPER_PROMPT = """You are a patent examiner building a claim chart. Compare each feature of the invention
with the candidate prior-art patent below.

status: "disclosed" if the candidate clearly teaches the feature, "partial" if it teaches a similar or more
general version, "not_found" if it does not.
evidence: the shortest phrase (max 25 words) copied verbatim from the candidate text that proves the status
(no paraphrasing); empty if not_found.
explanation: one sentence, max 20 words. summary: one sentence, max 30 words.

Invention features:
{features}

Candidate {pid}:
\"\"\"{text}\"\"\""""


def _normalize(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def _shorten(text: str, max_words: int) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + " ..."


def claim_mapper_node(payload: Dict) -> Dict:
    t0, tb = time.time(), toolbox()
    inv, cand, rank = payload["invention"], payload["candidate"], payload["rank"]
    pid = cand["patent_id"]
    text = cand.get("meta", "")
    claims = tb.store.claims.get(pid)
    if claims:
        text += "\n\nClaims (excerpt): " + claims[:3000]

    chart: ClaimChart = structured(ClaimChart, "smart").invoke(
        MAPPER_PROMPT.format(features=_numbered(inv["features"]), pid=pid, text=text))

    # Each feature exactly once; evidence must really occur in the candidate text
    by_feature = {r.feature: r for r in chart.rows}
    haystack = _normalize(text)
    rows = []
    for i, feature in enumerate(inv["features"], 1):
        r = by_feature.get(i)
        status, evidence, why = (r.status, r.evidence.strip(), r.explanation) if r else ("not_found", "", "Not assessed")
        verified = bool(evidence) and _normalize(evidence) in haystack
        evidence = _shorten(evidence, 30)
        if status != "not_found" and not verified:
            status = "partial" if status == "disclosed" else status
        rows.append({"feature": i, "text": feature, "status": status, "evidence": evidence,
                     "verified": verified, "explanation": why})

    counts = {s: sum(r["status"] == s for r in rows) for s in STATUS_RANK}
    # Claim similarity: share of the invention's features this patent discloses (partial counts half)
    coverage = (counts["disclosed"] + 0.5 * counts["partial"]) / max(len(rows), 1)
    result = {"rank": rank, "patent_id": pid, "title": cand.get("title", ""), "url": cand.get("url", ""),
              "sd_score": cand["sd_score"], "td_score": cand["td_score"], "sdtd_score": cand["sdtd_score"],
              "shared_ipc": cand.get("shared_ipc", []), "codes": cand.get("codes", []),
              "claims_compared": bool(claims), "coverage": round(coverage, 3),
              "rows": rows, "summary": chart.summary}
    msg = (f"{pid} \"{cand.get('title', '')[:70]}\" (SDTD {cand['sdtd_score']:.3f}): claim similarity "
           f"{coverage:.0%}, {counts['disclosed']} disclosed, {counts['partial']} partial, "
           f"{counts['not_found']} not found. {chart.summary}")
    return {"charts": [result], "log": _log(f"Claim Mapper #{rank}", msg, t0)}
