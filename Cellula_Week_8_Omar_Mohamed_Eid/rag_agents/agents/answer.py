"""Answer Agent: turns evidence + analysis into the final cited response."""
from __future__ import annotations

import re

from langchain_core.messages import HumanMessage, SystemMessage

from ..formatting import format_history, location_label, parse_evidence_id
from ..schemas import AnalystReport, Evidence, FinalAnswer
from ..tools.answer_tools import citation_formatter, response_formatter, source_formatter

_SYSTEM = """You are the Answer Agent. You write the final reply to the user from the analysis and evidence you are given.

Rules:
- Base the answer ONLY on the analysis and the numbered sources below. Do not add outside knowledge.
- Cite with the source numbers in square brackets, like [1] or [1][2], right after the claim they support.
  Only use numbers that appear in the source list. Do not write a sources list yourself; it is appended automatically.
- Be clear and well organised. Use a markdown table when comparing several items; use short lists for enumerations.
- Report numbers exactly as given in the analysis/calculations.
- If the analysis says the evidence is incomplete, answer what the documents do support and state plainly what is missing.
- Answer in the language the user wrote the question in."""


class AnswerAgent:
    def __init__(self, llm):
        self.llm = llm

    def answer(
        self,
        question: str,
        evidence: list[Evidence],
        report: AnalystReport | None,
        history: list[dict] | None = None,
    ) -> FinalAnswer:
        if not evidence or report is None:
            return FinalAnswer(
                text="I couldn't find anything in the uploaded documents that answers this question.", citations=[]
            )

        used = [i for i in (parse_evidence_id(r, len(evidence)) for r in report.used_evidence) if i is not None]
        if not used:
            used = list(range(len(evidence)))
        citations, idx_to_number = citation_formatter(evidence, used)

        sources_block = "\n\n".join(
            f"[{idx_to_number[i]}] {evidence[i].source}, {location_label(evidence[i])}\n{evidence[i].content[:2000]}"
            for i in used
        )
        analysis_block = report.analysis
        if report.calculations:
            analysis_block += "\n\nCalculations:\n" + "\n".join(f"- {c}" for c in report.calculations)
        if report.comparisons:
            analysis_block += "\n\nComparisons:\n" + "\n".join(f"- {c}" for c in report.comparisons)
        # the analysis cites evidence ids (E3); the answer must cite source numbers ([1]) -> translate for the model
        for i in sorted(idx_to_number, reverse=True):
            analysis_block = analysis_block.replace(f"[E{i + 1}]", f"[{idx_to_number[i]}]").replace(f"E{i + 1}", f"[{idx_to_number[i]}]")
        if not report.sufficient and report.missing_information:
            analysis_block += "\n\nNOT FOUND in the documents: " + "; ".join(report.missing_information)

        reply = self.llm.invoke(
            [
                SystemMessage(content=_SYSTEM),
                HumanMessage(
                    content=(
                        f"Conversation so far:\n{format_history(history, 4)}\n\n"
                        f"Question: {question}\n\nAnalysis:\n{analysis_block}\n\nNumbered sources:\n{sources_block}"
                    )
                ),
            ]
        )
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        text = response_formatter(text, {c.number for c in citations})
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        final_citations = [c for c in citations if c.number in cited] or citations
        sources = source_formatter(final_citations)
        return FinalAnswer(text=f"{text}\n\n{sources}" if sources else text, citations=final_citations)
