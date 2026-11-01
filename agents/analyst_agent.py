"""
AnalystAgent:
Specialized in applying the novel SDTD methodology (Semantic + Technological Distance),
re-ranking candidate patents, assessing infringement / prior-art risk, and generating
executive patent intelligence reports.
"""
import re
from typing import List, Dict, Any, Optional

# Similarity awarded for the deepest shared IPC level:
# section (H), class (H05), subclass (H05K), main group (H05K7), subgroup (H05K7/2029)
IPC_LEVEL_WEIGHTS = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 0.9, 5: 1.0}
_IPC_RE = re.compile(r"^([A-HY])(\d{2})?([A-Z])?(\d{1,4})?(?:/(\d{1,6}))?")


def to_code_list(codes) -> List[str]:
    if codes is None:
        return []
    if isinstance(codes, str):
        return [codes]
    try:
        return [str(c) for c in codes]
    except Exception:
        return [str(codes)]


def ipc_levels(code: str) -> Optional[List[str]]:
    """'H05K 7/2029' -> ['H', 'H05', 'H05K', 'H05K7', 'H05K7/2029']"""
    code = re.sub(r"\s+", "", str(code).upper())
    m = _IPC_RE.match(code)
    if not m:
        return None
    levels, prefix = [], ""
    for part, sep in zip(m.groups(), ("", "", "", "", "/")):
        if part is None:
            break
        prefix += sep + part
        levels.append(prefix)
    return levels


def common_depth(a: List[str], b: List[str]) -> int:
    depth = 0
    for x, y in zip(a, b):
        if x != y:
            break
        depth += 1
    return depth


def ipc_code_similarity(a: List[str], b: List[str]) -> float:
    if a == b:
        return 1.0
    return IPC_LEVEL_WEIGHTS[common_depth(a, b)]


class AnalystAgent:
    def __init__(self):
        self.name = "AnalystAgent"
        self.role = "Hybrid SDTD Reranker & Patent Intelligence Analyst"

    def calculate_technological_distance(self, query_codes, candidate_codes) -> float:
        """
        Hierarchical IPC/CPC similarity (soft Jaccard).
        Each query code is matched to its closest candidate code (and vice versa) by the
        deepest shared level of the IPC tree (section > class > subclass > group > subgroup),
        so e.g. H05K7/2029 vs H05K7/20 scores 0.9 instead of 0.
        """
        q_levels = [lv for lv in (ipc_levels(c) for c in to_code_list(query_codes)) if lv]
        c_levels = [lv for lv in (ipc_levels(c) for c in to_code_list(candidate_codes)) if lv]
        if not q_levels or not c_levels:
            return 0.0

        q_to_c = sum(max(ipc_code_similarity(q, c) for c in c_levels) for q in q_levels) / len(q_levels)
        c_to_q = sum(max(ipc_code_similarity(c, q) for q in q_levels) for c in c_levels) / len(c_levels)
        return float((q_to_c + c_to_q) / 2.0)

    def shared_ipc_prefixes(self, query_codes, candidate_codes) -> List[str]:
        """Deepest common IPC prefix for each query code with its closest candidate code."""
        c_levels = [lv for lv in (ipc_levels(c) for c in candidate_codes) if lv]
        shared = []
        for code in query_codes:
            q = ipc_levels(code)
            if not q or not c_levels:
                continue
            best = max(c_levels, key=lambda c: common_depth(q, c))
            depth = common_depth(q, best)
            if depth:
                shared.append(q[depth - 1])
        return list(dict.fromkeys(shared))

    def evaluate_candidates(self, query_patent: Dict, candidates: List[Dict]) -> List[Dict]:
        """
        Re-ranks candidates by computing:
        1. SD = Semantic Distance / Cosine Similarity (from RetrievalAgent)
        2. TD = Technological Distance / hierarchical IPC overlap
        3. SDTD = ((TD + 1) * SD) / 2
        """
        q_codes = to_code_list(query_patent.get("codes", []))
        analyzed = []

        for cand in candidates:
            c_codes = to_code_list(cand.get("codes", []))
            
            # Semantic Similarity (SD)
            sd = float(cand.get("sd_score", 0.0))
            
            # Technological Similarity (TD)
            td = self.calculate_technological_distance(q_codes, c_codes)
            
            # Hybrid SDTD
            sdtd = ((td + 1.0) * sd) / 2.0
            
            # Categorize overlap risk
            if sdtd >= 0.65:
                risk_level = "HIGH OVERLAP (Potential Direct Prior Art / Conflict)"
            elif sdtd >= 0.40:
                risk_level = "MODERATE OVERLAP (Related Technical Art)"
            else:
                risk_level = "LOW OVERLAP (Distinct Invention Area)"

            overlap_codes = self.shared_ipc_prefixes(q_codes, c_codes)
            distinct_query_codes = list(set(q_codes) - set(c_codes))
            distinct_cand_codes = list(set(c_codes) - set(q_codes))

            item = dict(cand)
            item["codes"] = c_codes  # Ensure python list for JSON serialization
            item.update({
                "sd_score": round(sd, 4),
                "td_score": round(td, 4),
                "sdtd_score": round(sdtd, 4),
                "risk_level": risk_level,
                "shared_ipc": overlap_codes,
                "distinct_query_ipc": distinct_query_codes,
                "distinct_cand_ipc": distinct_cand_codes,
            })
            analyzed.append(item)

        # Sort descending by hybrid SDTD score
        analyzed.sort(key=lambda x: x["sdtd_score"], reverse=True)
        return analyzed

    def generate_report(self, query_patent: Dict, analyzed_candidates: List[Dict]) -> str:
        """
        Generates an executive markdown report comparing the query to top prior art.
        """
        lines = []
        lines.append("# Patent Intelligence & Prior-Art Analysis Report")
        lines.append(f"**Target Invention / Query**: {query_patent.get('title', 'N/A')}")
        lines.append(f"**Document ID**: `{query_patent.get('patent_id', 'N/A')}`")
        lines.append(f"**Classifications (IPC)**: `{', '.join(query_patent.get('codes', [])) or 'None'}`")
        lines.append(f"**Source**: {query_patent.get('source', 'Unknown')}\n")
        lines.append("## Executive Summary")
        lines.append(
            f"Evaluated top {len(analyzed_candidates)} candidate patents using the "
            "**Hybrid SDTD (Semantic Distance + Technological Distance)** methodology."
        )
        lines.append("")
        lines.append("| Rank | Candidate Patent | SD (Semantic) | TD (IPC Overlap) | SDTD (Hybrid) | Risk Level |")
        lines.append("| :---: | :--- | :---: | :---: | :---: | :--- |")

        for i, c in enumerate(analyzed_candidates, 1):
            pid = c.get("patent_id", f"Candidate {i}")
            sd = c.get("sd_score", 0.0)
            td = c.get("td_score", 0.0)
            sdtd = c.get("sdtd_score", 0.0)
            risk = c.get("risk_level", "Unknown")
            lines.append(f"| {i} | `{pid}` | {sd:.4f} | {td:.4f} | **{sdtd:.4f}** | {risk} |")

        lines.append("\n## Detailed Candidate Breakdown\n")

        for i, c in enumerate(analyzed_candidates, 1):
            lines.append(f"### {i}. Patent `{c.get('patent_id')}` (SDTD: {c.get('sdtd_score'):.4f})")
            lines.append(f"- **Shared IPC Classes**: `{', '.join(c.get('shared_ipc', [])) or 'None'}`")
            lines.append(f"- **Candidate IPC Classes**: `{', '.join(c.get('codes', []))}`")
            meta_snippet = c.get("meta", "")
            if len(meta_snippet) > 280:
                meta_snippet = meta_snippet[:280] + "..."
            lines.append(f"- **Summary Text**: *\"{meta_snippet}\"*")
            lines.append(f"- **Analyst Note**: {c.get('risk_level')}. "
                         f"Semantic cosine similarity is {c.get('sd_score'):.4f} with technological overlap of {c.get('td_score'):.4f}.")
            lines.append("")

        lines.append("---")
        lines.append("*Generated by SDTD Multi-Agent Patent Intelligence Engine.*")
        return "\n".join(lines)
