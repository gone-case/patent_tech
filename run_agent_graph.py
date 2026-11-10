"""
CLI for the LangGraph multi-agent prior-art search.

  python run_agent_graph.py --query "Quantum key distribution transceiver with polarization drift compensation"
  python run_agent_graph.py --query US10123456B2 --top_k 5
  python run_agent_graph.py --file sample_invention.md --review     # edit features / codes before searching
"""
import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from graph import resume_search, start_search


def print_events(events):
    for ev in events:
        if ev["type"] == "step":
            secs = f" ({ev['seconds']:.1f}s)" if "seconds" in ev else ""
            print(f"[{ev['agent']}]{secs} {ev['message']}")
            for q in ev.get("data", {}).get("queries", []):
                tag = f"feature {q['feature']}" if q["feature"] else "whole invention"
                print(f"    - ({tag}) {q['text'][:140]}")
        elif ev["type"] in ("review", "result", "error"):
            return ev
    return None


def review_prompt(ev):
    inv = ev["invention"]
    print("\n--- Human review ---")
    for i, f in enumerate(inv["features"], 1):
        print(f"  {i}. {f}")
    print(f"  CPC ({ev['codes_source']}): {', '.join(inv['cpc_codes'])}")
    edits = {}
    drop = input("Feature numbers to drop (comma separated, Enter to keep all): ").strip()
    if drop:
        gone = {int(x) for x in drop.split(",") if x.strip().isdigit()}
        edits["features"] = [f for i, f in enumerate(inv["features"], 1) if i not in gone]
    extra = input("Add a feature (Enter to skip): ").strip()
    if extra:
        edits["features"] = edits.get("features", inv["features"]) + [extra]
    codes = input("Replace CPC codes (comma separated, Enter to keep): ").strip()
    if codes:
        edits["cpc_codes"] = [c.strip() for c in codes.split(",") if c.strip()]
    return edits


def main():
    parser = argparse.ArgumentParser(description="SDTD LangGraph multi-agent prior-art search")
    parser.add_argument("--query", type=str, help="Patent ID or invention description")
    parser.add_argument("--file", type=str, help="Invention document (.pdf, .docx, .txt, .md, .json)")
    parser.add_argument("--ipc", type=str, default=None, help="Comma-separated IPC/CPC codes (otherwise predicted)")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--review", action="store_true", help="Pause after intake to edit features / codes")
    args = parser.parse_args()

    query = args.file or args.query
    if not query:
        parser.error("give --query or --file")
    codes = [c.strip().upper() for c in args.ipc.split(",") if c.strip()] if args.ipc else None

    ev = print_events(start_search(query, codes, args.top_k, args.review))
    while ev and ev["type"] == "review":
        ev = print_events(resume_search(ev["thread_id"], review_prompt(ev)))

    if ev and ev["type"] == "result":
        print("\n" + ev["report_markdown"])
        print(f"\nFinished in {ev['elapsed_seconds']}s over {ev['rounds']} search round(s). Saved to prior_art_report.md")
    elif ev and ev["type"] == "error":
        print(f"\n[Error] {ev['message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
