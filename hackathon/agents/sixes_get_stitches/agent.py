# This file is part of Prometheus
# https://inl.gov/super-intelligence/prometheus/
#
# Copyright 2026 Battelle Energy Alliance, LLC
# All rights reserved, see NOTICE.txt for full restrictions
# https://github.com/idaholab/prometheus_core/blob/main/NOTICE.txt
#
# Licensed under the MIT License, please see LICENSE for details
# https://github.com/idaholab/prometheus_core/blob/main/LICENSE
#
# SPDX-License-Identifier: MIT

"""
Read Extract Plot Agent.

Two skills:

* ``extract_table`` reads an uploaded markdown document, finds the table that
  matches a description, and publishes it as a JSON artifact.
* ``plot_table`` takes that table artifact and publishes a PNG plot of it.

This is a Prometheus specialist agent.  Subclass PrometheusAgent and implement
`_execute` — the base class handles the HTTP server, authentication, agent-card
serving, and registry registration for you.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from prometheus_api.agent import EventQueue, PrometheusAgent, RequestContext
from prometheus_api.llm_client import LLMConfig
from prometheus_api.storage_client import ConnectionConfig

# Read secrets and configuration from environment variables.
# Never hard-code API keys or URLs here.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CHART_TYPES = ("line", "bar", "scatter")

# Words that say "a table" but not *which* table; ignored when matching by keyword.
_STOPWORDS = {"the", "table", "with", "data", "that", "this", "about", "from", "and", "for"}
_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)*\|?$")


# ----------------------------------------------------------------------
# Markdown table parsing (pure functions, easy to unit-test)
# ----------------------------------------------------------------------


def _split_row(line: str) -> list[str]:
    """Split one markdown table line into stripped cell strings."""
    line = line.strip().removeprefix("|").removesuffix("|")
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line)]


def _convert_cell(cell: str) -> Any:
    """Turn a numeric-looking cell into an int or float; keep anything else a string."""
    text = cell.replace(",", "") if re.fullmatch(r"[-+]?[\d,]+(\.\d+)?", cell) else cell
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return cell


def parse_markdown_tables(markdown: str) -> list[dict[str, Any]]:
    """Find every pipe table in ``markdown``.

    Returns a list of ``{"caption", "columns", "rows"}`` dicts in document order.
    The caption is the nearest non-blank line above the table (bold markers
    removed), which in the almanac is the "**Table N. ...**" line.
    """
    lines = markdown.splitlines()
    tables: list[dict[str, Any]] = []
    i = 0
    while i < len(lines) - 1:
        is_header = lines[i].lstrip().startswith("|")
        if is_header and _SEPARATOR_RE.match(lines[i + 1].strip()):
            columns = _split_row(lines[i])
            rows: list[list[Any]] = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                cells = _split_row(lines[j])
                cells = (cells + [""] * len(columns))[: len(columns)]  # pad or trim
                rows.append([_convert_cell(c) for c in cells])
                j += 1

            k = i - 1
            while k >= 0 and not lines[k].strip():
                k -= 1
            caption = lines[k].replace("**", "").lstrip("# ").strip() if k >= 0 else ""
            tables.append({"caption": caption, "columns": columns, "rows": rows})
            i = j
        else:
            i += 1
    return tables


def _pick_by_number(tables: list[dict[str, Any]], description: str) -> dict[str, Any] | None:
    """Match 'Table 2' in the description to a caption that starts 'Table 2'."""
    wanted = re.search(r"\btable\s*(\d+)\b", description, re.IGNORECASE)
    if not wanted:
        return None
    for table in tables:
        have = re.match(r"table\s*(\d+)\b", table["caption"], re.IGNORECASE)
        if have and have.group(1) == wanted.group(1):
            return table
    return None


def _pick_by_keywords(tables: list[dict[str, Any]], description: str) -> dict[str, Any] | None:
    """Pick the table whose caption and headers share the most words with the description."""
    words = {w for w in re.findall(r"[a-z]{3,}", description.lower()) if w not in _STOPWORDS}
    best, best_score = None, 0
    for table in tables:
        text = f"{table['caption']} {' '.join(table['columns'])}".lower()
        score = len(words & set(re.findall(r"[a-z]{3,}", text)))
        if score > best_score:
            best, best_score = table, score
    return best


class ReadExtractPlotAgent(PrometheusAgent):
    """Read Extract Plot Agent."""

    async def _execute(
        self,
        params: dict[str, Any],
        context: RequestContext,
        event_queue: EventQueue,
    ) -> dict[str, Any]:
        """Route one request to the handler for the skill the orchestrator chose.

        Args:
            params: Parameters from the caller's JSON payload, plus ``skill_id``
                from the message metadata (the skill the orchestrator selected).
            context: The A2A request context.  Use ``context.context_id`` as a
                stable key for per-conversation state.
            event_queue: Pass to the ``send_*`` helpers to emit progress updates
                and artifacts.

        Returns:
            A plain ``dict``.  The base class wraps it as the final ``completed``
            task status automatically — do not enqueue that event yourself.
        """
        # The skill ids here must match the "id" fields in agent_card.json.
        handler = {
            "extract_table": self._handle_extract_table,
            "plot_table": self._handle_plot_table,
        }.get(params.get("skill_id"))
        if handler is None:
            return {"status": "error", "error": f"Unknown skill: {params.get('skill_id')}"}
        return await handler(params, context, event_queue)

    # ----------------------------------------------------------------------
    # Skill 1: read the document and extract a table
    # ----------------------------------------------------------------------

    async def _handle_extract_table(
        self,
        params: dict[str, Any],
        context: RequestContext,
        event_queue: EventQueue,
    ) -> dict[str, Any]:
        # Input names must match the properties in the card's inputSchema.
        document_artifact_id = params.get("document_artifact_id")
        table_description = params.get("table_description")
        if not document_artifact_id or not table_description:
            return {
                "status": "error",
                "error": "document_artifact_id and table_description are required",
            }

        await self.send_status_update(
            context=context,
            event_queue=event_queue,
            message="Reading the document…",
        )

        # The skill receives an artifact id, not the file. Resolve it to a local path.
        try:
            path = await self.get_artifact_file_path(
                document_artifact_id, context=context, artifacts=params.get("artifacts")
            )
        except FileNotFoundError:
            return {"status": "error", "error": "That artifact is not in this conversation."}
        markdown = path.read_text(encoding="utf-8")

        await self.send_status_update(
            context=context,
            event_queue=event_queue,
            message="Looking for the table…",
        )

        try:
            caption, columns, rows = await self._find_table(markdown, table_description, context)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}

        # catalog_artifact=True registers the table in the conversation catalog,
        # which is how plot_table finds it later by id.
        table_artifact_id = await self.send_json_artifact(
            context,
            event_queue,
            {"caption": caption, "columns": columns, "rows": rows},
            name="extracted_table",
            description=caption or table_description,
            catalog_artifact=True,
        )
        return {
            "status": "success",
            "table_artifact_id": table_artifact_id,
            "caption": caption,
            "columns": columns,
            "row_count": len(rows),
        }

    async def _find_table(
        self, markdown: str, table_description: str, context: RequestContext
    ) -> tuple[str, list[str], list[list[Any]]]:
        """Find the table in ``markdown`` that matches ``table_description``.

        Returns ``(caption, columns, rows)``. Raises ``ValueError`` with a
        user-readable message when no table matches.

        Order of attempts: an explicit "Table N" in the description, the only
        table in the document, the LLM's pick, then keyword overlap.
        """
        tables = parse_markdown_tables(markdown)
        if not tables:
            raise ValueError("I found no markdown tables in that document.")

        table = _pick_by_number(tables, table_description)
        if table is None and len(tables) == 1:
            table = tables[0]
        if table is None:
            table = await self._pick_with_llm(tables, table_description, context)
        if table is None:
            table = _pick_by_keywords(tables, table_description)
        if table is None:
            captions = "; ".join(t["caption"] or "(untitled)" for t in tables)
            raise ValueError(
                f"None of the {len(tables)} tables matched '{table_description}'. "
                f"Tables found: {captions}"
            )
        return table["caption"], table["columns"], table["rows"]

    async def _pick_with_llm(
        self, tables: list[dict[str, Any]], description: str, context: RequestContext
    ) -> dict[str, Any] | None:
        """Ask the LLM which table matches. Returns None if it fails or answers badly."""
        listing = "\n".join(
            f"{i}: {t['caption']} (columns: {', '.join(t['columns'])})"
            for i, t in enumerate(tables)
        )
        prompt = (
            "A user wants one table from a document. Reply with ONLY the number of the "
            "table that best matches their description.\n\n"
            f"Description: {description}\n\nTables:\n{listing}"
        )
        try:
            cfg = LLMConfig.from_env()
            if cfg.proxy_url:  # only the platform's LiteLLM proxy needs the master key
                cfg = cfg.with_api_key(os.getenv("LITELLM_MASTER_KEY"))
            async with self.llm_client(context, config=cfg) as llm:
                response = await llm.medium(prompt, max_tokens=20)
        except Exception:
            logger.exception("LLM table pick failed; falling back to keyword match")
            return None

        match = re.search(r"\d+", response.text)
        if match and int(match.group()) < len(tables):
            return tables[int(match.group())]
        return None

    # ----------------------------------------------------------------------
    # Skill 2: plot a table that extract_table produced
    # ----------------------------------------------------------------------

    async def _handle_plot_table(
        self,
        params: dict[str, Any],
        context: RequestContext,
        event_queue: EventQueue,
    ) -> dict[str, Any]:
        table_artifact_id = params.get("table_artifact_id")
        x_column = params.get("x_column")
        y_columns = params.get("y_columns")
        chart_type = params.get("chart_type") or "line"

        if isinstance(y_columns, str):  # tolerate a single column name
            y_columns = [y_columns]
        if not table_artifact_id or not x_column or not y_columns:
            return {
                "status": "error",
                "error": "table_artifact_id, x_column and y_columns are required",
            }
        if chart_type not in CHART_TYPES:
            return {
                "status": "error",
                "error": f"chart_type must be one of {', '.join(CHART_TYPES)}",
            }

        # Load the table that extract_table published.
        try:
            path = await self.get_artifact_file_path(
                table_artifact_id, context=context, artifacts=params.get("artifacts")
            )
        except FileNotFoundError:
            return {"status": "error", "error": "That artifact is not in this conversation."}
        table = json.loads(path.read_text(encoding="utf-8"))
        columns: list[str] = table["columns"]
        rows: list[list[Any]] = table["rows"]

        missing = [c for c in [x_column, *y_columns] if c not in columns]
        if missing:
            return {
                "status": "error",
                "error": f"Unknown column(s): {', '.join(missing)}. Available: {', '.join(columns)}",
            }

        await self.send_status_update(
            context=context,
            event_queue=event_queue,
            message="Drawing the plot…",
        )

        title = table.get("caption") or "Table plot"
        with tempfile.TemporaryDirectory() as tmp:
            png = Path(tmp) / "plot.png"
            try:
                # Plotting is CPU work; keep it off the event loop.
                await asyncio.to_thread(
                    self._render_plot, png, columns, rows, x_column, y_columns, chart_type, title
                )
            except ValueError as exc:
                return {"status": "error", "error": str(exc)}
            # Upload inside the `with` block: the temp file is deleted when it exits.
            ref = await self.send_deeplynx_artifact(
                context=context,
                event_queue=event_queue,
                file_path=png,
                name="plot.png",
                description=f"{chart_type} plot of {', '.join(y_columns)} vs {x_column}",
                storage_config=ConnectionConfig.from_env(),
                mime_type="image/png",
            )
        return {
            "status": "success",
            "artifact_id": ref.artifact_id,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_columns": y_columns,
        }

    def _render_plot(
        self,
        output_path: Path,
        columns: list[str],
        rows: list[list[Any]],
        x_column: str,
        y_columns: list[str],
        chart_type: str,
        title: str,
    ) -> None:
        """Draw the chart and save it as a PNG at ``output_path``.

        Raises ``ValueError`` if a y column is not numeric.
        """
        x_index = columns.index(x_column)
        x_values = [row[x_index] for row in rows]
        x_is_numeric = all(isinstance(v, (int, float)) for v in x_values)

        # Bars (and text x values) sit at 0..n-1 with the x values as tick labels.
        categorical = chart_type == "bar" or not x_is_numeric
        positions = list(range(len(x_values))) if categorical else x_values

        # Figure() instead of pyplot: no global state, no display needed.
        fig = Figure(figsize=(9, 5.5))
        ax = fig.subplots()
        width = 0.8 / len(y_columns)

        for k, name in enumerate(y_columns):
            y_values = [row[columns.index(name)] for row in rows]
            if not all(isinstance(v, (int, float)) for v in y_values):
                raise ValueError(f"Column '{name}' is not numeric, so it can't be plotted.")
            if chart_type == "line":
                ax.plot(positions, y_values, marker="o", label=name)
            elif chart_type == "scatter":
                ax.scatter(positions, y_values, label=name)
            else:
                offset = (k - (len(y_columns) - 1) / 2) * width
                ax.bar([p + offset for p in positions], y_values, width=width, label=name)

        if categorical:
            ax.set_xticks(positions)
            ax.set_xticklabels([str(v) for v in x_values], rotation=45, ha="right")
        ax.set_xlabel(x_column)
        ax.set_ylabel(", ".join(y_columns))
        ax.set_title(title, fontsize=10, wrap=True)
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    agent = ReadExtractPlotAgent(agent_name="Read Extract Plot Agent")
    agent.run()
