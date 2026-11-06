"""Analyst Agent: reasons over the evidence with tools and decides whether it is sufficient."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from ..config import Settings
from ..formatting import format_evidence_block, parse_evidence_id
from ..llm import structured
from ..schemas import AnalystReport, Evidence
from ..tools.analysis import AnalystToolkit

_SYSTEM = """You are the Analyst Agent in a document question-answering team.
The Retriever Agent has gathered numbered evidence (E1, E2, ...) from the user's documents.
Your job: understand, combine, compare, calculate and reason over THAT evidence only.

Rules:
- Use ONLY the evidence. Never use outside knowledge and never guess missing numbers.
- Use the calculator for ANY arithmetic. Use extract_tables / analyze_data for tabular numbers and
  compare_documents when several documents are involved.
- If something needed to answer is missing from the evidence, call request_more_evidence with a specific
  query (you may call it several times), then finish.
- Refer to evidence by id, e.g. [E2], inside your reasoning.
When you have finished using tools, reply with a short plain-text note (the structured report is requested next)."""

_FINAL = """Now write the final structured report.
- sufficient: true ONLY if the evidence fully supports an answer to every part of the question.
- analysis: the reasoning and findings with exact figures (cite evidence ids like E2).
- calculations: each calculation done with its result (taken from the calculator, not from memory).
- comparisons: key similarities / differences across documents, if relevant.
- used_evidence: ids of the evidence your analysis relies on (e.g. ["E1", "E3"]).
- If not sufficient: missing_information (what is missing) and 1-3 precise follow_up_queries for the Retriever."""


class AnalystAgent:
    def __init__(self, llm, settings: Settings):
        self.llm = llm
        self.settings = settings

    def analyze(self, question: str, evidence: list[Evidence]) -> tuple[AnalystReport, list[str]]:
        """Returns (report, tool_trace)."""
        toolkit = AnalystToolkit(evidence, self.llm)
        tools = toolkit.tools()
        by_name = {t.name: t for t in tools}
        trace: list[str] = []

        messages = [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"Question: {question}\n\nEvidence:\n{format_evidence_block(evidence)}"),
        ]
        tool_llm = self.llm.bind_tools(tools)
        for _ in range(self.settings.max_analyst_tool_steps):
            ai = tool_llm.invoke(messages)
            messages.append(ai)
            calls = getattr(ai, "tool_calls", None) or []
            if not calls:
                break
            for call in calls:
                name, args = call["name"], call.get("args", {}) or {}
                tool = by_name.get(name)
                try:
                    output = tool.invoke(args) if tool else f"Unknown tool '{name}'."
                except Exception as exc:
                    output = f"Tool error: {exc}"
                output = output if isinstance(output, str) else json.dumps(output, default=str)
                trace.append(f"{name}({json.dumps(args, ensure_ascii=False, default=str)[:160]}) -> {output[:160].replace(chr(10), ' ')}")
                messages.append(ToolMessage(content=output[:6000], tool_call_id=call.get("id", name), name=name))

        try:
            messages.append(HumanMessage(content=_FINAL))
            report = structured(self.llm, AnalystReport).invoke(messages)
            if isinstance(report, dict):
                report = AnalystReport(**report)
        except Exception as exc:
            trace.append(f"Structured report failed ({exc}); evidence marked insufficient")
            report = AnalystReport(
                sufficient=False,
                analysis="The analysis could not be completed.",
                missing_information=["analysis failed"],
                follow_up_queries=[question],
            )

        # keep only valid evidence ids
        valid = []
        for ref in report.used_evidence:
            idx = parse_evidence_id(ref, len(evidence))
            if idx is not None and f"E{idx + 1}" not in valid:
                valid.append(f"E{idx + 1}")
        report.used_evidence = valid

        # queries requested through the request_more_evidence tool count as a signal of insufficiency
        if toolkit.more_evidence_requests:
            report.sufficient = False
            for q in toolkit.more_evidence_requests:
                if q and q not in report.follow_up_queries:
                    report.follow_up_queries.append(q)
        if not report.sufficient:
            report.follow_up_queries = [q for q in report.follow_up_queries if q.strip()][:3]
        else:
            report.follow_up_queries = []
        return report, trace
