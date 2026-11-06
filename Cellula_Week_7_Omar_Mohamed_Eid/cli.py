"""Command-line interface.

  python cli.py ingest sample_docs/*.txt
  python cli.py ask "What is the average CNN accuracy across the three papers?"
  python cli.py chat
  python cli.py list
  python cli.py clear
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chat with your documents using Retriever, Analyst and Answer agents.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ingest = sub.add_parser("ingest", help="Add PDF / DOCX / TXT files to the knowledge base")
    p_ingest.add_argument("files", nargs="+")
    p_ask = sub.add_parser("ask", help="Ask one question")
    p_ask.add_argument("question")
    p_ask.add_argument("--trace", action="store_true", help="Show what each agent did")
    sub.add_parser("chat", help="Interactive chat (keeps conversation history)")
    sub.add_parser("list", help="List documents in the knowledge base")
    sub.add_parser("clear", help="Delete everything from the knowledge base")
    args = parser.parse_args(argv)

    from rag_agents.system import RagSystem

    rag = RagSystem()

    if args.cmd == "ingest":
        failed = False
        for r in rag.ingest(args.files):
            if r.ok:
                print(f"OK    {r.source}: {r.text_chunks} text + {r.table_chunks} table chunks")
            else:
                failed = True
                print(f"FAIL  {r.source}: {r.error}")
            for w in r.warnings:
                print(f"WARN  {r.source}: {w}")
        return 1 if failed else 0

    if args.cmd == "list":
        docs = rag.documents()
        if not docs:
            print("(empty)")
        for name, info in docs.items():
            print(f"{name}: {info['chunks']} chunks ({info['tables']} tables)")
        return 0

    if args.cmd == "clear":
        rag.clear()
        print("Knowledge base cleared.")
        return 0

    if args.cmd == "ask":
        result = rag.ask(args.question)
        print(result.text)
        if args.trace:
            _print_trace(result)
        return 0

    if args.cmd == "chat":
        history: list[dict] = []
        print("Ask about your documents. Type 'exit' to quit.")
        while True:
            try:
                q = input("\nyou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if q.lower() in {"exit", "quit"}:
                return 0
            if not q:
                continue
            result = rag.ask(q, history)
            print(f"\n{result.text}")
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": result.text}]
    return 0


def _print_trace(result) -> None:
    print("\n--- trace ---")
    for n, r in enumerate(result.retrievals, start=1):
        print(f"Retriever pass {n}:")
        for line in r.trace:
            print(f"  {line}")
    for n, trace in enumerate(result.analyst_trace, start=1):
        print(f"Analyst pass {n}:")
        for line in trace or ["(no tools used)"]:
            print(f"  {line}")
    if result.analysis:
        print(f"Sufficient evidence: {result.analysis.sufficient} (feedback loops: {result.feedback_loops})")


if __name__ == "__main__":
    sys.exit(main())
