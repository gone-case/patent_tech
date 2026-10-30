"""
AnalystAgent:
Specialized in applying the novel SDTD methodology (Semantic + Technological Distance),
re-ranking candidate patents, assessing infringement / prior-art risk, and generating
executive patent intelligence reports.
"""
from typing import List, Dict, Any


class AnalystAgent:
    def __init__(self):
        self.name = "AnalystAgent"
        self.role = "Hybrid SDTD Reranker & Patent Intelligence Analyst"

    def calculate_technological_distance(self, query_codes, candidate_codes) -> float:
        """
        Calculates Jaccard overlap between IPC/CPC classifications.
        TD = |Intersection| / |Union|
        """
        def to_set(codes):
            if codes is None:
                return set()
            if isinstance(codes, (list, tuple, set)):
                return set(codes)
            try:
                return set(list(codes))
            except Exception:
                return {str(codes)}

        set_q = to_set(query_codes)
        set_c = to_set(candidate_codes)
        
        union = set_q | set_c
        if not union:
            return 0.0
        
        intersection = set_q & set_c
        return float(len(intersection) / len(union))

    def evaluate_candidates(self, query_patent: Dict, candidates: List[Dict]) -> List[Dict]:
        """
        Re-ranks candidates by computing:
        1. SD = Semantic Distance / Cosine Similarity (from RetrievalAgent)
        2. TD = Technological Distance / IPC Jaccard Overlap
        3. SDTD = ((TD + 1) * SD) / 2
        """
        def to_list(codes):
            if codes is None:
                return []
            if isinstance(codes, list):
                return codes
            try:
                return list(codes)
            except Exception:
                return [str(codes)]

        q_codes = to_list(query_patent.get("codes", []))
        analyzed = []

        for cand in candidates:
            c_codes = to_list(cand.get("codes", []))
            
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

            overlap_codes = list(set(q_codes) & set(c_codes))
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
