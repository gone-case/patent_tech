import sys
import argparse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agents.orchestrator import OrchestratorAgent


def interactive_menu(orchestrator: OrchestratorAgent):
    while True:
        print("\n" + "=" * 60)
        print(">> SDTD MULTI-AGENT PATENT INTELLIGENCE ENGINE <<")
        print("=" * 60)
        print("1. Search Google Patent ID (e.g., US10123456B2, US9876543B1)")
        print("2. Search Custom Invention Idea / Research Abstract")
        print("3. Upload Document File (.pdf, .docx, .txt, .md, .json)")
        print("4. Export Google Colab GPU Worker (for embedding 1,000,000+ patents)")
        print("5. Quick Benchmark Test (Check existing index & sample search)")
        print("6. Exit")
        print("-" * 60)

        choice = input("Select an option (1-6): ").strip()

        if choice == "1":
            patent_id = input("\nEnter Google Patent Publication Number (e.g., US10123456B2): ").strip()
            if patent_id:
                res = orchestrator.run_prior_art_search(query=patent_id, top_k=5)
                print("\n" + res["report_markdown"])
        elif choice == "2":
            text = input("\nEnter invention title & abstract description:\n").strip()
            codes = input("Enter expected IPC classifications (e.g., G06F, H04L or press Enter for G06F): ").strip()
            ipc_list = [c.strip().upper() for c in codes.split(",") if c.strip()] or ["G06F"]
            if text:
                res = orchestrator.run_prior_art_search(query=text, custom_ipc_codes=ipc_list, top_k=5)
                print("\n" + res["report_markdown"])
        elif choice == "3":
            file_path = input("\nEnter absolute or relative document path (.pdf, .docx, .txt, .md, .json):\n").strip().strip('"').strip("'")
            if file_path:
                codes = input("Enter optional IPC classifications (press Enter to auto-detect from text): ").strip()
                ipc_list = [c.strip().upper() for c in codes.split(",") if c.strip()] if codes else None
                try:
                    res = orchestrator.run_prior_art_search(query=file_path, custom_ipc_codes=ipc_list, top_k=5)
                    print("\n" + res["report_markdown"])
                except Exception as e:
                    print(f"\n[Error reading document]: {e}")
        elif choice == "4":
            orchestrator.prepare_colab_cloud_worker()
        elif choice == "5":
            sample_query = "A machine learning system for patent document similarity using semantic embeddings and technological distance."
            res = orchestrator.run_prior_art_search(query=sample_query, custom_ipc_codes=["G06F", "G06N"], top_k=5)
            print("\n" + res["report_markdown"])
        elif choice == "6":
            print("\nExiting Multi-Agent Patent System. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid option. Please enter 1, 2, 3, 4, 5, or 6.")


def main():
    parser = argparse.ArgumentParser(description="SDTD Multi-Agent Patent Intelligence CLI")
    parser.add_argument("--query", type=str, help="Patent ID (e.g., US10123456B2) or natural language description")
    parser.add_argument("--file", type=str, help="Path to uploaded document file (.pdf, .docx, .txt, .md, .json)")
    parser.add_argument("--ipc", type=str, default="G06F", help="Comma-separated IPC codes (e.g., 'G06F,H04L')")
    parser.add_argument("--top_k", type=int, default=5, help="Number of prior art candidates to retrieve")
    parser.add_argument("--colab", action="store_true", help="Generate and display Google Colab cloud worker instructions")

    args = parser.parse_args()

    orchestrator = OrchestratorAgent()

    if args.colab:
        orchestrator.prepare_colab_cloud_worker()
        return

    ipc_list = [c.strip().upper() for c in args.ipc.split(",") if c.strip()] if args.ipc else None

    if args.file:
        res = orchestrator.run_prior_art_search(query=args.file, custom_ipc_codes=ipc_list, top_k=args.top_k)
        print("\n" + res["report_markdown"])
    elif args.query:
        res = orchestrator.run_prior_art_search(query=args.query, custom_ipc_codes=ipc_list, top_k=args.top_k)
        print("\n" + res["report_markdown"])
    else:
        # Launch interactive menu
        interactive_menu(orchestrator)


if __name__ == "__main__":
    main()
