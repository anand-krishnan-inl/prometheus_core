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
* ``plot_table`` takes that table artifact (or an uploaded CSV file) and
  publishes a PNG plot of it.  When the columns are missing or don't match, it
  asks the user with a dropdown of the real column names.

This is a Prometheus specialist agent.  Subclass PrometheusAgent and implement
`_execute` — the base class handles the HTTP server, authentication, agent-card
serving, and registry registration for you.
"""

import asyncio
import csv
import io
import json
import logging
import os
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from prometheus_api.agent import EventQueue, MissingInput, PrometheusAgent, RequestContext
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

# Cells that mean "no value" (the almanac uses an em dash).
_MISSING = {"", "—", "–", "-", "n/a", "na", "nan", "none", "null", "?", "…", "..."}
# A number with optional thousands separators (comma, space, thin space), decimals,
# exponent, and percent sign: "1,020", "2 610", "-3.5", "1.2e3", "45%".
_NUMBER_RE = re.compile(
    r"[-+]?(?:\d{1,3}(?:[,    ]\d{3})+|\d+|(?=\.\d))(?:\.\d+)?(?:[eE][-+]?\d+)?%?"
)
# Units in a header: "Core temp. [°C]" or "Brie Basin (residents)".
_UNITS_RE = re.compile(r"\s*[\[(][^\])]*[\])]")


# ----------------------------------------------------------------------
# Markdown and CSV table parsing (pure functions, easy to unit-test)
# ----------------------------------------------------------------------


def _split_row(line: str) -> list[str]:
    """Split one markdown table line into stripped cell strings."""
    line = line.strip().removeprefix("|").removesuffix("|")
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", line)]


def _convert_cell(cell: Any) -> Any:
    """Turn a numeric-looking cell into an int or float, a missing marker into None,
    and keep anything else a string."""
    if not isinstance(cell, str):
        return cell
    text = cell.strip().strip("*`").strip()
    if text.lower() in _MISSING:
        return None
    candidate = text.replace("−", "-")  # unicode minus sign
    if not _NUMBER_RE.fullmatch(candidate):
        return text
    candidate = re.sub(r"[,    %]", "", candidate)
    try:
        return int(candidate)
    except ValueError:
        return float(candidate)


def _clean_columns(columns: list[str]) -> list[str]:
    """Give empty headers a name and make duplicate headers unique."""
    cleaned: list[str] = []
    for i, name in enumerate(columns):
        name = name.strip().strip("*`").strip() or f"Column {i + 1}"
        base, n = name, 2
        while name in cleaned:
            name, n = f"{base} ({n})", n + 1
        cleaned.append(name)
    return cleaned


def _caption_above(lines: list[str], header_index: int) -> str:
    """Return the paragraph directly above a table, joined into one line.

    Captions in the almanac wrap over several lines, so walk up to the blank
    line that starts the paragraph. Bold markers and heading hashes are removed.
    """
    k = header_index - 1
    while k >= 0 and not lines[k].strip():
        k -= 1
    paragraph: list[str] = []
    while k >= 0 and lines[k].strip() and not lines[k].lstrip().startswith("|"):
        paragraph.insert(0, lines[k].strip())
        k -= 1
    caption = " ".join(paragraph).replace("**", "").replace("__", "")
    return caption.lstrip("# ").strip()


def parse_markdown_tables(markdown: str) -> list[dict[str, Any]]:
    """Find every pipe table in ``markdown``.

    Returns a list of ``{"caption", "columns", "rows"}`` dicts in document order.
    The caption is the paragraph directly above the table (bold markers
    removed), which in the almanac is the "**Table N. ...**" paragraph.
    Numeric cells become numbers and missing-value markers become ``None``.
    """
    lines = markdown.splitlines()
    tables: list[dict[str, Any]] = []
    i = 0
    while i < len(lines) - 1:
        is_header = lines[i].lstrip().startswith("|")
        if is_header and _SEPARATOR_RE.match(lines[i + 1].strip()):
            columns = _clean_columns(_split_row(lines[i]))
            rows: list[list[Any]] = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                cells = _split_row(lines[j])
                cells = (cells + [""] * len(columns))[: len(columns)]  # pad or trim
                rows.append([_convert_cell(c) for c in cells])
                j += 1
            tables.append({"caption": _caption_above(lines, i), "columns": columns, "rows": rows})
            i = j
        else:
            i += 1
    return tables


def parse_csv_table(text: str) -> dict[str, Any]:
    """Parse CSV (or TSV / semicolon-separated) text into ``{"caption", "columns", "rows"}``.

    Raises ``ValueError`` when the text has no header row or only one column.
    """
    try:
        dialect: Any = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    records = [r for r in csv.reader(io.StringIO(text), dialect) if any(c.strip() for c in r)]
    if not records or len(records[0]) < 2:
        raise ValueError("That file doesn't look like a table: I need a header row and 2+ columns.")
    columns = _clean_columns(records[0])
    rows = [
        [_convert_cell(c) for c in (r + [""] * len(columns))[: len(columns)]] for r in records[1:]
    ]
    return {"caption": "", "columns": columns, "rows": rows}


def load_table_file(path: Path) -> dict[str, Any]:
    """Load a table from a JSON table artifact, a CSV file, or a one-table markdown file.

    Raises ``ValueError`` with a user-readable message for anything else.
    """
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "That file is not a text table. I can plot a table from extract_table or a CSV file."
        ) from exc
    if not text.strip():
        raise ValueError("That file is empty.")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and isinstance(data.get("columns"), list):
        return {
            "caption": str(data.get("caption") or ""),
            "columns": [str(c) for c in data["columns"]],
            "rows": [list(r) for r in data.get("rows") or []],
        }
    if isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
        columns = list(data[0].keys())  # a list of records, e.g. [{"x": 1, "y": 2}, ...]
        return {
            "caption": "",
            "columns": columns,
            "rows": [[_convert_cell(r.get(c)) for c in columns] for r in data],
        }
    if data is not None:
        raise ValueError("That JSON file is not a table (I expected 'columns' and 'rows').")

    tables = parse_markdown_tables(text)
    if len(tables) == 1:
        return tables[0]
    if len(tables) > 1:
        captions = "; ".join(t["caption"] or "(untitled)" for t in tables)
        raise ValueError(
            f"That document has {len(tables)} tables. Extract one with extract_table first. "
            f"Tables found: {captions}"
        )
    return parse_csv_table(text)


def _pick_by_number(tables: list[dict[str, Any]], description: str) -> dict[str, Any] | None:
    """Match 'Table 2' in the description to a caption that starts 'Table 2'.

    Raises ``ValueError`` if the description names a table number that doesn't exist.
    """
    wanted = re.search(r"\btable\s*(\d+)\b", description, re.IGNORECASE)
    if not wanted:
        return None
    numbered = False
    for table in tables:
        have = re.match(r"table\s*(\d+)\b", table["caption"], re.IGNORECASE)
        if have and have.group(1) == wanted.group(1):
            return table
        numbered = numbered or bool(have)
    if not numbered:  # the captions carry no numbers; let the other matchers try
        return None
    captions = "; ".join(t["caption"] or "(untitled)" for t in tables)
    raise ValueError(f"There is no Table {wanted.group(1)} in that document. Tables found: {captions}")


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


# ----------------------------------------------------------------------
# Column matching (pure functions; the LLM handles what these can't)
# ----------------------------------------------------------------------


def _strip_units(name: str) -> str:
    """'Core temp. [°C]' -> 'Core temp.'"""
    return _UNITS_RE.sub("", name).strip()


def _tokens(name: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", name.lower()))


def _unique(candidates: list[str]) -> str | None:
    return candidates[0] if len(candidates) == 1 else None


def match_column(requested: str, columns: list[str]) -> str | None:
    """Map a user's column name to a real column, or None if it's not clear.

    Tries, in order: the exact name, the name without units (case-sensitive,
    so 't' is not 'T'), the same ignoring case, then a unique column that
    contains all of the requested words ("porcini" -> "Porcini yield (kg)").
    """
    requested = requested.strip()
    if not requested:
        return None
    if requested in columns:
        return requested
    bare = _strip_units(requested)
    found = _unique([c for c in columns if _strip_units(c) == bare]) or _unique(
        [c for c in columns if _strip_units(c).lower() == bare.lower()]
    )
    if found:
        return found
    words = _tokens(bare)
    if words:
        return _unique([c for c in columns if words <= _tokens(c)])
    return None


def numeric_columns(columns: list[str], rows: list[list[Any]]) -> list[str]:
    """Columns whose non-missing values are all numbers (and that have at least one)."""
    result = []
    for i, name in enumerate(columns):
        values = [r[i] for r in rows if i < len(r) and r[i] is not None]
        if values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            result.append(name)
    return result


def _as_list(value: Any) -> list[str]:
    """Accept a list of names, one name, or a JSON-encoded list of names."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                return [text]
        else:
            return [text] if text else []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


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
        try:
            return await handler(params, context, event_queue)
        except Exception as exc:
            # Last line of defence: the user gets a sentence, the log gets the traceback.
            logger.exception("Skill %s failed", params.get("skill_id"))
            return {"status": "error", "error": f"Something went wrong: {exc}"}

    async def _load_text_artifact(
        self, artifact_id: str, context: RequestContext, params: dict[str, Any]
    ) -> Path:
        """Resolve an artifact id to a local path, or raise ``ValueError`` with a clear message."""
        try:
            return await self.get_artifact_file_path(
                artifact_id, context=context, artifacts=params.get("artifacts")
            )
        except FileNotFoundError as exc:
            raise ValueError("That artifact is not in this conversation.") from exc

    async def _llm_text(self, prompt: str, context: RequestContext, max_tokens: int) -> str:
        """Ask the LLM a short question. Raises on any LLM failure; callers fall back."""
        cfg = LLMConfig.from_env()
        if cfg.proxy_url:  # only the platform's LiteLLM proxy needs the master key
            cfg = cfg.with_api_key(os.getenv("LITELLM_MASTER_KEY"))
        async with self.llm_client(context, config=cfg) as llm:
            response = await llm.medium(prompt, max_tokens=max_tokens)
        return response.text or ""

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
        table_description = str(params.get("table_description") or "").strip()
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
            path = await self._load_text_artifact(document_artifact_id, context, params)
            markdown = path.read_text(encoding="utf-8-sig")
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        except UnicodeDecodeError:
            return {
                "status": "error",
                "error": "That file is not a text document. Upload a markdown (.md) file.",
            }

        try:
            caption, columns, rows = await self._find_table(
                markdown, table_description, context, event_queue
            )
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
            "numeric_columns": numeric_columns(columns, rows),
            "row_count": len(rows),
        }

    async def _find_table(
        self,
        markdown: str,
        table_description: str,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> tuple[str, list[str], list[list[Any]]]:
        """Find the table in ``markdown`` that matches ``table_description``.

        Returns ``(caption, columns, rows)``. Raises ``ValueError`` with a
        user-readable message when no table matches.

        Order of attempts: an explicit "Table N" in the description, the only
        table in the document, the LLM's pick, then keyword overlap (only if
        the LLM could not be reached; an LLM "none" is final).
        """
        tables = parse_markdown_tables(markdown)
        if not tables:
            hint = " If it is a CSV file, plot it directly." if "," in markdown else ""
            raise ValueError(f"I found no markdown tables in that document.{hint}")

        await self.send_status_update(
            context=context,
            event_queue=event_queue,
            message=f"Found {len(tables)} table(s); picking the one you described…",
        )
        captions = "; ".join(t["caption"] or "(untitled)" for t in tables)

        table = _pick_by_number(tables, table_description)
        if table is None and len(tables) == 1:
            table = tables[0]
        if table is None:
            table, llm_answered = await self._pick_with_llm(tables, table_description, context)
            if table is None and llm_answered:
                raise ValueError(
                    f"None of the {len(tables)} tables matches '{table_description}'. "
                    f"Tables found: {captions}"
                )
        if table is None:
            table = _pick_by_keywords(tables, table_description)
        if table is None:
            raise ValueError(
                f"None of the {len(tables)} tables matched '{table_description}'. "
                f"Tables found: {captions}"
            )
        return table["caption"], table["columns"], table["rows"]

    async def _pick_with_llm(
        self, tables: list[dict[str, Any]], description: str, context: RequestContext
    ) -> tuple[dict[str, Any] | None, bool]:
        """Ask the LLM which table matches.

        Returns ``(table, answered)``: ``(None, True)`` means the LLM said no table
        matches; ``(None, False)`` means the LLM failed or answered badly.
        """
        listing = "\n".join(
            f"{i}: {t['caption'] or '(untitled)'} (columns: {', '.join(t['columns'])})"
            for i, t in enumerate(tables)
        )
        prompt = (
            "A user wants one table from a document. Users describe tables by what the "
            "caption says, often in other words. Reply with ONLY the number of the table "
            "that best matches their description, or NONE if no table is about that topic.\n\n"
            f"Description: {description}\n\nTables:\n{listing}"
        )
        try:
            text = await self._llm_text(prompt, context, max_tokens=20)
        except Exception:
            logger.exception("LLM table pick failed; falling back to keyword match")
            return None, False

        match = re.search(r"\d+", text)
        if match and int(match.group()) < len(tables):
            return tables[int(match.group())], True
        if "none" in text.lower():
            return None, True
        return None, False

    # ----------------------------------------------------------------------
    # Skill 2: plot a table that extract_table produced, or an uploaded CSV
    # ----------------------------------------------------------------------

    async def _handle_plot_table(
        self,
        params: dict[str, Any],
        context: RequestContext,
        event_queue: EventQueue,
    ) -> dict[str, Any]:
        # On a resume after a follow-up question, the answers arrive separately
        # and the Python state is gone: load what was saved before asking.
        ctx = str(context.context_id)
        provided = self.get_provided_inputs(context)
        state = self.load_context_state(ctx) if provided else {}

        table_artifact_id = params.get("table_artifact_id") or state.get("table_artifact_id")
        chart_type = str(params.get("chart_type") or state.get("chart_type") or "line")
        chart_type = chart_type.strip().lower().removesuffix(" chart").removesuffix(" plot")
        chart_type = {"lines": "line", "bars": "bar", "column": "bar", "points": "scatter"}.get(
            chart_type, chart_type
        )
        x_requested = provided.get("x_column") or params.get("x_column")
        y_requested = _as_list(provided.get("y_columns") or params.get("y_columns"))

        if not table_artifact_id:
            return {
                "status": "error",
                "error": "table_artifact_id is required: a table from extract_table or a CSV file",
            }
        if chart_type not in CHART_TYPES:
            return {
                "status": "error",
                "error": f"chart_type must be one of {', '.join(CHART_TYPES)}",
            }

        # Load the table: a JSON table artifact, a CSV upload, or a one-table markdown file.
        try:
            path = await self._load_text_artifact(table_artifact_id, context, params)
            table = await asyncio.to_thread(load_table_file, path)
        except ValueError as exc:
            return {"status": "error", "error": str(exc)}
        columns: list[str] = table["columns"]
        rows: list[list[Any]] = table["rows"]
        if not rows:
            return {"status": "error", "error": "That table has no rows to plot."}
        numeric = numeric_columns(columns, rows)
        if not numeric:
            return {
                "status": "error",
                "error": f"No column in that table is numeric, so there is nothing to plot. "
                f"Columns: {', '.join(columns)}",
            }

        x_column, y_columns, unmatched = await self._resolve_columns(
            x_requested, y_requested, columns, numeric, table.get("caption", ""), context
        )
        non_numeric_y = [c for c in y_columns if c not in numeric]
        y_columns = [c for c in y_columns if c in numeric]

        # Ask for whatever is still missing, offering the real column names.
        missing: list[MissingInput] = []
        if not x_column:
            missing.append(
                MissingInput(
                    name="x_column",
                    description="Column for the x axis",
                    user_prompt="Which column goes on the x axis?",
                    type="enum",
                    options=columns,
                )
            )
        if not y_columns:
            missing.append(
                MissingInput(
                    name="y_columns",
                    description="Numeric column for the y axis",
                    user_prompt="Which column do you want to plot on the y axis?",
                    type="enum",
                    options=[c for c in numeric if c != x_column] or numeric,
                )
            )
        if missing:
            self.save_context_state(
                ctx, {"table_artifact_id": table_artifact_id, "chart_type": chart_type}
            )
            hint = "Choose the columns to plot."
            if non_numeric_y and not y_columns:
                hint = f"{', '.join(repr(c) for c in non_numeric_y)} is not numeric. {hint}"
            if unmatched:
                hint = f"I couldn't match {', '.join(repr(u) for u in unmatched)}. {hint}"
            await self.send_input_required(
                context, event_queue, missing_inputs=missing, hint=hint
            )
            return {}

        await self.send_status_update(
            context=context,
            event_queue=event_queue,
            message=f"Plotting {', '.join(y_columns)} vs {x_column}…",
        )

        title = table.get("caption") or f"{', '.join(y_columns)} vs {x_column}"
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
            try:
                ref = await self.send_deeplynx_artifact(
                    context=context,
                    event_queue=event_queue,
                    file_path=png,
                    name="plot.png",
                    description=f"{chart_type} plot of {', '.join(y_columns)} vs {x_column}",
                    storage_config=ConnectionConfig.from_env(),
                    mime_type="image/png",
                )
            except Exception:
                logger.exception("Uploading the plot to DeepLynx failed")
                return {
                    "status": "error",
                    "error": "I drew the plot but could not store it. Check the DeepLynx "
                    "settings in the agent's .env (/diagnose can help).",
                }
        result = {
            "status": "success",
            "artifact_id": ref.artifact_id,
            "chart_type": chart_type,
            "x_column": x_column,
            "y_columns": y_columns,
        }
        if non_numeric_y:
            result["skipped_non_numeric_columns"] = non_numeric_y
        return result

    async def _resolve_columns(
        self,
        x_requested: Any,
        y_requested: list[str],
        columns: list[str],
        numeric: list[str],
        caption: str,
        context: RequestContext,
    ) -> tuple[str | None, list[str], list[str]]:
        """Map the requested column names to real ones.

        Returns ``(x_column or None, matched y columns, names that could not be matched)``.
        Rule-based matching first; the LLM handles words like "temperature" for
        "Core temp. [°C]". A name the LLM can't map is reported, never guessed.
        """
        requested = ([str(x_requested)] if x_requested else []) + y_requested
        mapping = {name: match_column(name, columns) for name in requested}
        unresolved = [name for name, col in mapping.items() if col is None]
        if unresolved:
            mapping.update(await self._match_columns_with_llm(unresolved, columns, caption, context))

        x_column = mapping.get(str(x_requested)) if x_requested else None
        unmatched = [str(x_requested)] if x_requested and not x_column else []
        y_columns: list[str] = []
        for name in y_requested:
            found = [mapping[name]] if mapping.get(name) else []
            if not found and "," in name:  # "a, b" sent as one string
                parts = [match_column(p, columns) for p in name.split(",")]
                found = [p for p in parts if p] if all(parts) else []
            if not found:
                unmatched.append(name)
            y_columns.extend(c for c in found if c not in y_columns)
        return x_column, y_columns, unmatched

    async def _match_columns_with_llm(
        self, requested: list[str], columns: list[str], caption: str, context: RequestContext
    ) -> dict[str, str]:
        """Ask the LLM to map user words to column names. Returns only valid mappings."""
        prompt = (
            "Map each name a user wrote to the one table column it means. Columns often use "
            "symbols, abbreviations and units (e.g. 'temperature' -> 'T [K]', 'time' -> 't [min]'; "
            "note 't' and 'T' differ). Use null when no column clearly fits.\n"
            'Reply with ONLY a JSON object like {"user name": "exact column name or null"}.\n\n'
            f"Table caption: {caption or '(none)'}\n"
            f"Columns: {json.dumps(columns, ensure_ascii=False)}\n"
            f"User names: {json.dumps(requested, ensure_ascii=False)}"
        )
        try:
            text = await self._llm_text(prompt, context, max_tokens=300)
            found = re.search(r"\{.*\}", text, re.DOTALL)
            answer = json.loads(found.group()) if found else {}
        except Exception:
            logger.exception("LLM column match failed; asking the user instead")
            return {}
        if not isinstance(answer, dict):
            return {}
        return {
            name: answer[name]
            for name in requested
            if isinstance(answer.get(name), str) and answer[name] in columns
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

        Missing values (``None``) are skipped point by point. Raises
        ``ValueError`` if a y column is not numeric or has nothing to plot.
        """
        x_index = columns.index(x_column)
        rows = [r for r in rows if x_index < len(r) and r[x_index] is not None]
        if not rows:
            raise ValueError(f"Column '{x_column}' has no values to plot against.")
        x_values = [row[x_index] for row in rows]
        x_is_numeric = all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in x_values
        )

        # Bars (and text x values) sit at 0..n-1 with the x values as tick labels.
        categorical = chart_type == "bar" or not x_is_numeric
        positions = list(range(len(x_values))) if categorical else x_values

        # Figure() instead of pyplot: no global state, no display needed (Agg backend).
        fig = Figure(figsize=(9, 5.5))
        ax = fig.subplots()
        width = 0.8 / len(y_columns)

        plotted = 0
        for k, name in enumerate(y_columns):
            y_index = columns.index(name)
            points = [
                (p, row[y_index])
                for p, row in zip(positions, rows)
                if y_index < len(row) and row[y_index] is not None
            ]
            if any(not isinstance(y, (int, float)) or isinstance(y, bool) for _, y in points):
                raise ValueError(f"Column '{name}' is not numeric, so it can't be plotted.")
            if not points:
                continue
            if chart_type == "line" and not categorical:
                points.sort(key=lambda pt: pt[0])
            xs, ys = zip(*points)
            if chart_type == "line":
                ax.plot(xs, ys, marker="o", label=name)
            elif chart_type == "scatter":
                ax.scatter(xs, ys, label=name)
            else:
                offset = (k - (len(y_columns) - 1) / 2) * width
                ax.bar([p + offset for p in xs], ys, width=width, label=name)
            plotted += 1
        if not plotted:
            raise ValueError(f"{', '.join(y_columns)} has no values to plot.")

        if categorical:
            labels = [str(v) for v in x_values]
            ax.set_xticks(positions)
            crowded = len(labels) > 6 or max(len(s) for s in labels) > 6
            ax.set_xticklabels(
                labels, rotation=45 if crowded else 0, ha="right" if crowded else "center"
            )
        ax.set_xlabel(x_column)
        # One series: its name. Several sharing a unit ("(residents)"): just the unit.
        units = {u.group() if (u := _UNITS_RE.search(c)) else "" for c in y_columns}
        shared_unit = units.pop().strip() if len(units) == 1 else ""
        ylabel = y_columns[0] if len(y_columns) == 1 else shared_unit or ", ".join(y_columns)
        ax.set_ylabel(textwrap.fill(ylabel, 50))
        ax.set_title(textwrap.fill(title, 90), fontsize=10)
        if len(y_columns) > 1:
            ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    agent = ReadExtractPlotAgent(agent_name="Read Extract Plot Agent")
    agent.run()
