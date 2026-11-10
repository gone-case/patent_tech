"""
Reporter agent (Gemini): novelty verdict and recommendations, assembled with the claim charts into the markdown report.
"""
import time
from typing import Dict, List

from ..llm import get_llm
from ..state import AgentState
from .common import STATUS_RANK, _log, _numbered


REPORT_PROMPT = """You are writing the conclusions of a prior-art search report for an inventor.
Use only the claim charts below. Write markdown with exactly these sections, concise and specific:

## Novelty assessment
One short paragraph, then one bullet per feature: likely novel, partially anticipated, or anticipated, naming
the patent(s) responsible.

## Closest prior art
The two or three most relevant patents and what they teach.

## Recommendations
3-5 bullets on how to focus or word the claims around what appears novel.

Invention: {title}
Features:
{features}

Claim charts:
{charts}"""


def reporter_node(state: AgentState) -> Dict:
    t0 = time.time()
    inv, charts = state["invention"], sorted(state.get("charts", []), key=lambda c: c["rank"])

    novelty = []
    for i, feature in enumerate(inv["features"], 1):
        rows = [(c, r) for c in charts for r in c["rows"] if r["feature"] == i]
        best = max((STATUS_RANK[r["status"]] for _, r in rows), default=0)
        status = {v: k for k, v in STATUS_RANK.items()}[best]
        novelty.append({"feature": i, "text": feature, "status": status,
                        "patents": [c["patent_id"] for c, r in rows if STATUS_RANK[r["status"]] == best and best > 0]})

    chart_text = "\n\n".join(
        f"#{c['rank']} {c['patent_id']} - {c['title']} (SDTD {c['sdtd_score']:.3f})\n" +
        "\n".join(f"  feature {r['feature']}: {r['status']}. {r['explanation']}" for r in c["rows"])
        for c in charts) or "No candidates were retrieved."
    narrative = structured_text(REPORT_PROMPT.format(title=inv["title"], features=_numbered(inv["features"]),
                                                     charts=chart_text))

    report = _assemble_report(state, charts, narrative)
    with open("prior_art_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    novel = sum(n["status"] == "not_found" for n in novelty)
    msg = f"{novel} of {len(novelty)} features look novel against the top {len(charts)} candidates. Report written."
    return {"novelty": novelty, "narrative": narrative, "report": report, "log": _log("Reporter", msg, t0)}


def structured_text(prompt: str) -> str:
    content = get_llm("smart").invoke(prompt).content
    if isinstance(content, list):  # Gemini can return content blocks
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return content.strip()



def _assemble_report(state: AgentState, charts: List[Dict], narrative: str) -> str:
    inv, target = state["invention"], state["target"]
    lines = [
        "# Prior-Art Search Report (LangGraph agents)",
        f"**Invention**: {inv['title']}",
        f"**Source**: {target.get('source', '')} | **Input ID**: `{target.get('patent_id', '')}`",
        f"**CPC codes** ({state.get('codes_source', '')}): `{', '.join(inv['cpc_codes'])}`",
        f"**Search rounds**: {state.get('iteration', 0) + 1}",
        "",
        "## Technical features",
        *[f"{i}. {f}" for i, f in enumerate(inv["features"], 1)],
        "",
        narrative,
        "",
    ]

    lines += ["", "## Claim charts"]
    for c in charts:
        n = len(c["rows"])
        d = sum(r["status"] == "disclosed" for r in c["rows"])
        p = sum(r["status"] == "partial" for r in c["rows"])
        lines += [
            f"### #{c['rank']} `{c['patent_id']}`: {c['title']}",
            f"- **Scores**: SDTD **{c['sdtd_score']:.4f}** | SD (semantic) {c['sd_score']:.4f} | TD (CPC/IPC) {c['td_score']:.4f}",
            f"- **Shared CPC/IPC**: `{', '.join(c['shared_ipc']) or 'None'}`",
            f"- **Candidate CPC/IPC**: `{', '.join(c['codes'][:12])}`",
            f"- **Claim similarity**: {c['coverage']:.0%} of features ({d} disclosed, {p} partial of {n}); compared against "
            + ("abstract + claims" if c["claims_compared"] else "abstract only (no English claims in the dataset)"),
            f"- **Link**: {c['url']}",
            "", c["summary"], ""]
        for r in c["rows"]:
            ev = f" Evidence: \"{r['evidence']}\"" + ("" if r["verified"] else " (quote not found in text)") if r["evidence"] else ""
            lines.append(f"- **Feature {r['feature']}: {r['status']}.** {r['explanation']}{ev}")
        lines.append("")

    lines += ["## SDTD ranking", "| Rank | Patent | Title | SD | TD | SDTD | Shared CPC/IPC | Found via |",
              "| :---: | :--- | :--- | :---: | :---: | :---: | :--- | :--- |"]
    for i, c in enumerate(state["ranked"][: state.get("top_k", 5)], 1):
        lines.append(f"| {i} | `{c['patent_id']}` | {c.get('title', '')[:70]} | {c['sd_score']:.4f} | "
                     f"{c['td_score']:.4f} | **{c['sdtd_score']:.4f}** | {', '.join(c.get('shared_ipc', [])) or '-'} | "
                     f"{c.get('found_via', '')} |")
    lines += ["", "---", "*Generated by the SDTD LangGraph agents (Gemini on Vertex AI + Patent-BERT/FAISS).*"]
    return "\n".join(lines)
