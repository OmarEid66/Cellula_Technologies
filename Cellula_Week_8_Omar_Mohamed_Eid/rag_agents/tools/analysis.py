"""Analyst Agent tools: calculator, table extractor, document comparison, data analysis, request more evidence."""
from __future__ import annotations

import ast
import json
import math
import operator
import re
import statistics
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..formatting import evidence_id, format_evidence_block, location_label, parse_evidence_id
from ..schemas import Evidence

# ---------------------------------------------------------------- 1. Calculator (safe, no eval)


def _flat(args: tuple) -> list[float]:
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return list(args[0])
    return list(args)


_FUNCS = {
    "abs": abs,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": pow,
    "min": lambda *a: min(_flat(a)),
    "max": lambda *a: max(_flat(a)),
    "sum": lambda *a: sum(_flat(a)),
    "mean": lambda *a: statistics.fmean(_flat(a)),
    "median": lambda *a: statistics.median(_flat(a)),
    "stdev": lambda *a: statistics.stdev(_flat(a)),
    "pstdev": lambda *a: statistics.pstdev(_flat(a)),
    "pct_change": lambda old, new: (new - old) / old * 100,  # percentage change from old to new
    "pct_of": lambda part, whole: part / whole * 100,  # part as a percentage of whole
}
_CONSTS = {"pi": math.pi, "e": math.e}
_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 1000:
                raise ValueError("exponent too large")
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](ev(node.operand))
        if isinstance(node, (ast.List, ast.Tuple)):
            return [ev(x) for x in node.elts]
        if isinstance(node, ast.Name) and node.id in _CONSTS:
            return _CONSTS[node.id]
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")

    cleaned = expression.strip().replace("^", "**")
    return ev(ast.parse(cleaned, mode="eval"))


def run_calculator(expression: str) -> str:
    try:
        value = safe_eval(expression)
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:
        return (
            f"Error: {exc}. Use only numbers, + - * / ** %, parentheses and the functions "
            f"{sorted(_FUNCS)} (write percentages as x/100)."
        )
    if isinstance(value, float):
        value = round(value, 10)
    return f"{expression} = {value}"


# ---------------------------------------------------------------- 2. Table extractor


def parse_markdown_tables(text: str) -> list[dict[str, Any]]:
    """Find markdown tables in text and return {columns, rows[list[dict]]} structures."""
    tables = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1].strip().lstrip("|")):
            header = _split_row(lines[i])
            j = i + 2
            rows = []
            while j < len(lines) and lines[j].strip().startswith("|"):
                cells = _split_row(lines[j])
                cells += [""] * (len(header) - len(cells))
                rows.append({h: c for h, c in zip(header, cells)})
                j += 1
            if rows:
                tables.append({"columns": header, "rows": rows})
            i = j
        else:
            i += 1
    return tables


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    return [c.replace("\\|", "|").strip() for c in re.split(r"(?<!\\)\|", line)]


_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")


def to_number(cell: Any) -> Optional[float]:
    if isinstance(cell, (int, float)):
        return float(cell)
    m = _NUMBER.search(str(cell).replace("−", "-"))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


# ---------------------------------------------------------------- tool argument schemas


class _CalcArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. 'mean(92, 95, 89)' or '(95-92)/92*100'.")


class _TablesArgs(BaseModel):
    source: Optional[str] = Field(default=None, description="Only tables from this document (optional).")
    evidence_ids: Optional[list[str]] = Field(default=None, description="Only tables in these evidence ids (optional).")


class _CompareArgs(BaseModel):
    aspects: list[str] = Field(description="What to compare, e.g. ['accuracy', 'methodology', 'dataset'].")


class _AnalyzeArgs(BaseModel):
    values: Optional[dict[str, float]] = Field(default=None, description="label -> number, e.g. {'Paper A': 92}.")
    table_id: Optional[str] = Field(default=None, description="Table id returned by extract_tables (e.g. 'T1').")
    label_column: Optional[str] = Field(default=None, description="Column holding the row labels (for table_id).")
    value_column: Optional[str] = Field(default=None, description="Numeric column to analyse (for table_id).")
    higher_is_better: bool = Field(default=True, description="Whether larger values are better when ranking.")


class _MoreEvidenceArgs(BaseModel):
    query: str = Field(description="A specific search query for the Retriever Agent.")
    reason: str = Field(description="What information is missing and why it is needed.")


_COMPARE_SYSTEM = (
    "You compare documents using ONLY the evidence provided. For each requested aspect, list what each document "
    "says (cite evidence ids like [E2]), then note similarities and differences. If a document gives no "
    "information on an aspect, write 'not stated'. Do not add outside knowledge."
)


class AnalystToolkit:
    """Builds the Analyst's tools around the current evidence. One instance per analysis run."""

    def __init__(self, evidence: list[Evidence], llm):
        self.evidence = evidence
        self.llm = llm
        self.tables: dict[str, dict[str, Any]] = {}
        self.more_evidence_requests: list[str] = []

    # --- tool bodies ---
    def extract_tables(self, source: str | None = None, evidence_ids: list[str] | None = None) -> str:
        wanted = None
        if evidence_ids:
            wanted = {parse_evidence_id(x, len(self.evidence)) for x in evidence_ids} - {None}
        found = []
        for idx, ev in enumerate(self.evidence):
            if wanted is not None and idx not in wanted:
                continue
            if source and ev.source != source:
                continue
            for table in parse_markdown_tables(ev.content):
                tid = f"T{len(self.tables) + 1}"
                record = {"id": tid, "evidence_id": evidence_id(idx), "source": ev.source, "location": location_label(ev), **table}
                self.tables[tid] = record
                found.append(record)
        if not found:
            return "No tables found in the evidence. Numbers may be in plain text; read the evidence directly."
        preview = [{**t, "rows": t["rows"][:40]} for t in found]
        return json.dumps(preview, ensure_ascii=False, indent=1)

    def compare_documents(self, aspects: list[str]) -> str:
        by_doc: dict[str, list[str]] = {}
        for idx, ev in enumerate(self.evidence):
            by_doc.setdefault(ev.source, []).append(
                f"[{evidence_id(idx)}] ({location_label(ev)}) {ev.content[:1500]}"
            )
        if len(by_doc) < 2:
            return "Only one document is present in the evidence; nothing to compare across documents."
        grouped = "\n\n".join(f"### {doc}\n" + "\n\n".join(parts) for doc, parts in by_doc.items())
        reply = self.llm.invoke(
            [
                SystemMessage(content=_COMPARE_SYSTEM),
                HumanMessage(content=f"Aspects to compare: {aspects}\n\nEvidence grouped by document:\n{grouped}"),
            ]
        )
        return reply.content if isinstance(reply.content, str) else str(reply.content)

    def analyze_data(
        self,
        values: dict[str, float] | None = None,
        table_id: str | None = None,
        label_column: str | None = None,
        value_column: str | None = None,
        higher_is_better: bool = True,
    ) -> str:
        data: dict[str, float] = {}
        if values:
            data = {str(k): float(v) for k, v in values.items()}
        elif table_id:
            table = self.tables.get(table_id)
            if not table:
                return f"Unknown table id '{table_id}'. Call extract_tables first."
            if not value_column or value_column not in table["columns"]:
                return f"value_column must be one of {table['columns']}"
            label_col = label_column if label_column in table["columns"] else table["columns"][0]
            for n, row in enumerate(table["rows"]):
                num = to_number(row.get(value_column))
                if num is not None:
                    label = row.get(label_col) or f"row {n + 1}"
                    data[label] = num
        if not data:
            return "No numeric data to analyse. Provide `values` or a `table_id` with `value_column`."
        nums = list(data.values())
        ranking = sorted(data.items(), key=lambda kv: kv[1], reverse=higher_is_better)
        result: dict[str, Any] = {
            "n": len(nums),
            "mean": round(statistics.fmean(nums), 6),
            "median": round(statistics.median(nums), 6),
            "min": min(nums),
            "max": max(nums),
            "range": round(max(nums) - min(nums), 6),
            "stdev": round(statistics.stdev(nums), 6) if len(nums) > 1 else 0.0,
            "ranking_best_to_worst": [{"label": k, "value": v} for k, v in ranking],
            "best": {"label": ranking[0][0], "value": ranking[0][1]},
            "worst": {"label": ranking[-1][0], "value": ranking[-1][1]},
        }
        if len(ranking) > 1 and ranking[-1][1] != 0:
            result["best_vs_worst_pct_difference"] = round((ranking[0][1] - ranking[-1][1]) / abs(ranking[-1][1]) * 100, 4)
        return json.dumps(result, ensure_ascii=False, indent=1)

    def request_more_evidence(self, query: str, reason: str) -> str:
        self.more_evidence_requests.append(query.strip())
        return (
            "Request recorded: the Retriever Agent will search for this after your report. "
            "Finish your analysis with what you have now and set sufficient=false."
        )

    # --- LangChain tool objects ---
    def tools(self) -> list[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=run_calculator,
                name="calculator",
                description="Exact arithmetic (averages, percentages, differences, ratios, statistics). Always use it instead of calculating in your head.",
                args_schema=_CalcArgs,
            ),
            StructuredTool.from_function(
                func=self.extract_tables,
                name="extract_tables",
                description="Convert tables found in the evidence into structured rows/columns (returns table ids like T1).",
                args_schema=_TablesArgs,
            ),
            StructuredTool.from_function(
                func=self.compare_documents,
                name="compare_documents",
                description="Compare what different documents say about given aspects (similarities, differences, results, methods).",
                args_schema=_CompareArgs,
            ),
            StructuredTool.from_function(
                func=self.analyze_data,
                name="analyze_data",
                description="Statistics, ranking and best/worst for numeric values given directly or taken from an extracted table column.",
                args_schema=_AnalyzeArgs,
            ),
            StructuredTool.from_function(
                func=self.request_more_evidence,
                name="request_more_evidence",
                description="Ask the Retriever Agent for additional evidence when what you have is insufficient. Be specific.",
                args_schema=_MoreEvidenceArgs,
            ),
        ]
