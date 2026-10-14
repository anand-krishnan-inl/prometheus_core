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

"""Tests for the Read Extract Plot agent.

Parsing, matching and plotting are tested as pure functions; the skills are
tested through `_execute` with the artifact store, the LLM, DeepLynx, and the
context state mocked. Run with `uv run pytest`.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import agent as agent_module
from agent import (
    ReadExtractPlotAgent,
    load_table_file,
    match_column,
    numeric_columns,
    parse_csv_table,
    parse_markdown_tables,
    table_to_markdown,
)

ALMANAC = """\
# Report

**Table 1. Registry of heavy mining equipment in service at the end of FY2051,
listing each machine's call sign and condition.**

| Call sign | Condition |
|:----------|:----------|
| WHEY-01   | Good      |

Some prose in between.

**Table 2. Census of the resident population of the three lunar mining
settlements, 2042–2051.**

| Year | Brie Basin (residents) | Camembert Ridge (residents) | Feral cheese mice (est.) |
|-----:|-----------------------:|----------------------------:|-------------------------:|
| 2042 |                    188 |                           — |                      350 |
| 2043 |                    203 |                           — |                      420 |
| 2047 |                    341 |                         198 |                    1,020 |

**Table 3. Daily thermal log of the Great Wheel of Emmental.**

| Day | Core temp. [°C] | Rind temp. [°C] | t [min] | T [K] |
|----:|----------------:|----------------:|--------:|------:|
|   0 |            12.0 |            11.5 |       0 | 285.1 |
|   1 |            14.8 |            13.1 |      60 | 288.0 |
"""

CSV = "night,mice_sighted,cheese_lost_kg,tunnel_temperature_C\n1,14,0.8,4.1\n2,19,1.1,4.3\n3,,1.6,4.0\n"


def _make_agent() -> ReadExtractPlotAgent:
    # agent_card.json is loaded lazily by run()/build_app(), not by the
    # constructor, so unit tests can instantiate the agent directly.
    agent = ReadExtractPlotAgent(agent_name="Sixes Get Stitches Table Plot Agent")
    agent.send_status_update = AsyncMock()
    agent.send_json_artifact = AsyncMock(return_value="table-123")
    agent.send_deeplynx_artifact = AsyncMock(return_value=SimpleNamespace(artifact_id="png-456"))
    agent.send_input_required = AsyncMock()
    agent.save_context_state = MagicMock()
    agent.load_context_state = MagicMock(return_value={})
    agent.get_provided_inputs = MagicMock(return_value={})
    agent._llm_text = AsyncMock(side_effect=RuntimeError("no LLM in tests"))
    return agent


def _serve(agent: ReadExtractPlotAgent, tmp_path: Path, name: str, content: str | bytes) -> None:
    """Make get_artifact_file_path return a file with ``content``."""
    path = tmp_path / name
    path.write_bytes(content if isinstance(content, bytes) else content.encode())
    agent.get_artifact_file_path = AsyncMock(return_value=path)


@pytest.fixture(autouse=True)
def _no_deeplynx_env(monkeypatch):
    monkeypatch.setattr(agent_module.ConnectionConfig, "from_env", MagicMock())


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------


def test_parse_joins_multiline_captions():
    tables = parse_markdown_tables(ALMANAC)

    assert len(tables) == 3
    assert tables[1]["caption"].startswith("Table 2. Census of the resident population")
    assert tables[1]["caption"].endswith("2042–2051.")


def test_parse_converts_thousands_separators_and_dashes():
    rows = parse_markdown_tables(ALMANAC)[1]["rows"]

    assert rows[0] == [2042, 188, None, 350]
    assert rows[2][3] == 1020


def test_parse_ignores_pipe_lines_without_separator():
    assert parse_markdown_tables("| not | a table |\nprose\n") == []


def test_parse_csv_keeps_missing_values_as_none():
    table = parse_csv_table(CSV)

    assert table["columns"][0] == "night"
    assert table["rows"][2] == [3, None, 1.6, 4.0]


def test_parse_csv_rejects_one_column_text():
    with pytest.raises(ValueError):
        parse_csv_table("just some prose\nmore prose\n")


def test_load_table_file_rejects_binary(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe\x00")

    with pytest.raises(ValueError, match="not a text table"):
        load_table_file(path)


def test_numeric_columns_skips_text_and_allows_missing():
    table = parse_markdown_tables(ALMANAC)[1]

    assert numeric_columns(table["columns"], table["rows"]) == table["columns"]
    assert numeric_columns(["a", "b"], [["x", 1], ["y", None]]) == ["b"]


# ----------------------------------------------------------------------
# Column matching
# ----------------------------------------------------------------------


COLUMNS = ["Day", "Core temp. [°C]", "Rind temp. [°C]", "t [min]", "T [K]"]


@pytest.mark.parametrize(
    "requested, expected",
    [
        ("Day", "Day"),
        ("day", "Day"),
        ("t", "t [min]"),
        ("T", "T [K]"),
        ("core temp.", "Core temp. [°C]"),
        ("core", "Core temp. [°C]"),
        ("temp", None),  # two columns contain 'temp': ambiguous, don't guess
        ("pressure", None),
    ],
)
def test_match_column(requested, expected):
    assert match_column(requested, COLUMNS) == expected


# ----------------------------------------------------------------------
# Skill dispatch
# ----------------------------------------------------------------------


async def test_unknown_skill(context, event_queue):
    result = await _make_agent()._execute({"skill_id": "nope"}, context, event_queue)

    assert result["status"] == "error"


async def test_unexpected_exception_becomes_error_message(context, event_queue):
    agent = _make_agent()
    agent.get_artifact_file_path = AsyncMock(side_effect=RuntimeError("boom"))

    result = await agent._execute(
        {"skill_id": "sixes_extract_table", "document_artifact_id": "d", "table_description": "x"},
        context,
        event_queue,
    )

    assert result["status"] == "error"
    assert "boom" in result["error"]


# ----------------------------------------------------------------------
# sixes_extract_table
# ----------------------------------------------------------------------


async def test_extract_requires_inputs(context, event_queue):
    result = await _make_agent()._execute({"skill_id": "sixes_extract_table"}, context, event_queue)

    assert result["status"] == "error"


async def test_extract_by_table_number(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "doc.md", ALMANAC)

    result = await agent._execute(
        {"skill_id": "sixes_extract_table", "document_artifact_id": "d", "table_description": "Table 3"},
        context,
        event_queue,
    )

    assert result["status"] == "success"
    assert result["table_artifact_id"] == "table-123"
    assert result["columns"] == COLUMNS


async def test_extract_returns_table_data(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "doc.md", ALMANAC)

    result = await agent._execute(
        {"skill_id": "sixes_extract_table", "document_artifact_id": "d", "table_description": "Table 3"},
        context,
        event_queue,
    )

    assert result["table_markdown"].splitlines()[0].startswith("| " + COLUMNS[0])
    assert len(result["table_markdown"].splitlines()) == result["row_count"] + 2
    assert "answer" not in result


async def test_extract_answers_question(context, event_queue, tmp_path):
    agent = _make_agent()
    agent._llm_text = AsyncMock(return_value="42")
    _serve(agent, tmp_path, "doc.md", ALMANAC)

    result = await agent._execute(
        {
            "skill_id": "sixes_extract_table",
            "document_artifact_id": "d",
            "table_description": "Table 3",
            "question": "What is the largest value?",
        },
        context,
        event_queue,
    )

    assert result["answer"] == "42"
    assert "What is the largest value?" in agent._llm_text.await_args.args[0]


def test_table_to_markdown_truncates():
    text = table_to_markdown(["a", "b"], [[i, None] for i in range(100)], max_chars=60)

    assert len(text) < 100
    assert "more rows not shown" in text


async def test_extract_unknown_table_number(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "doc.md", ALMANAC)

    result = await agent._execute(
        {"skill_id": "sixes_extract_table", "document_artifact_id": "d", "table_description": "Table 9"},
        context,
        event_queue,
    )

    assert result["status"] == "error"
    assert "no Table 9" in result["error"]


async def test_extract_uses_llm_pick(context, event_queue, tmp_path):
    agent = _make_agent()
    agent._llm_text = AsyncMock(return_value="1")
    _serve(agent, tmp_path, "doc.md", ALMANAC)

    result = await agent._execute(
        {
            "skill_id": "sixes_extract_table",
            "document_artifact_id": "d",
            "table_description": "how many people live in each settlement",
        },
        context,
        event_queue,
    )

    assert result["caption"].startswith("Table 2.")


async def test_extract_llm_none_is_a_clear_error(context, event_queue, tmp_path):
    agent = _make_agent()
    agent._llm_text = AsyncMock(return_value="NONE")
    _serve(agent, tmp_path, "doc.md", ALMANAC)

    result = await agent._execute(
        {
            "skill_id": "sixes_extract_table",
            "document_artifact_id": "d",
            "table_description": "the dragon register",
        },
        context,
        event_queue,
    )

    assert result["status"] == "error"
    assert "Tables found" in result["error"]


async def test_extract_falls_back_to_keywords_without_llm(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "doc.md", ALMANAC)

    result = await agent._execute(
        {
            "skill_id": "sixes_extract_table",
            "document_artifact_id": "d",
            "table_description": "the census population table",
        },
        context,
        event_queue,
    )

    assert result["caption"].startswith("Table 2.")


async def test_extract_document_without_tables(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "doc.md", "just prose\n")

    result = await agent._execute(
        {"skill_id": "sixes_extract_table", "document_artifact_id": "d", "table_description": "x"},
        context,
        event_queue,
    )

    assert result["status"] == "error"
    assert "no markdown tables" in result["error"]


async def test_extract_missing_artifact(context, event_queue):
    agent = _make_agent()
    agent.get_artifact_file_path = AsyncMock(side_effect=FileNotFoundError)

    result = await agent._execute(
        {"skill_id": "sixes_extract_table", "document_artifact_id": "d", "table_description": "x"},
        context,
        event_queue,
    )

    assert result == {"status": "error", "error": "That artifact is not in this conversation."}


# ----------------------------------------------------------------------
# sixes_plot_table
# ----------------------------------------------------------------------


def _table_json() -> str:
    table = parse_markdown_tables(ALMANAC)[1]
    return json.dumps(table)


async def test_plot_table_artifact(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "table.json", _table_json())

    result = await agent._execute(
        {
            "skill_id": "sixes_plot_table",
            "table_artifact_id": "t",
            "x_column": "year",
            "y_columns": ["Brie Basin", "Camembert Ridge"],
        },
        context,
        event_queue,
    )

    assert result["status"] == "success", result
    assert result["x_column"] == "Year"
    assert result["y_columns"] == ["Brie Basin (residents)", "Camembert Ridge (residents)"]
    png = agent.send_deeplynx_artifact.await_args.kwargs["file_path"]
    assert agent.send_deeplynx_artifact.await_args.kwargs["mime_type"] == "image/png"
    assert png.name == "plot1.png"
    agent.save_context_state.assert_called_with("test-conversation", {"plot_count": 1})


async def test_plot_numbers_each_plot_in_a_conversation(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "table.json", _table_json())
    agent.load_context_state = MagicMock(return_value={"plot_count": 2})

    result = await agent._execute(
        {"skill_id": "sixes_plot_table", "table_artifact_id": "t"}, context, event_queue
    )

    assert result["plot_name"] == "plot3.png", result
    assert agent.send_deeplynx_artifact.await_args.kwargs["name"] == "plot3.png"
    agent.save_context_state.assert_called_with("test-conversation", {"plot_count": 3})


async def test_plot_csv_directly(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "sightings.csv", CSV)

    result = await agent._execute(
        {
            "skill_id": "sixes_plot_table",
            "table_artifact_id": "c",
            "x_column": "night",
            "y_columns": "mice_sighted",
            "chart_type": "Bar",
        },
        context,
        event_queue,
    )

    assert result["status"] == "success", result
    assert result["chart_type"] == "bar"


async def test_plot_without_columns_uses_defaults(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "sightings.csv", CSV)

    result = await agent._execute(
        {"skill_id": "sixes_plot_table", "table_artifact_id": "c"}, context, event_queue
    )

    assert result["status"] == "success", result
    assert result["x_column"] == "night"
    assert result["y_columns"] == ["mice_sighted", "cheese_lost_kg", "tunnel_temperature_C"]
    agent.send_input_required.assert_not_awaited()


async def test_plot_resumes_with_answers(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "sightings.csv", CSV)
    agent.get_provided_inputs = MagicMock(
        return_value={"x_column": "night", "y_columns": "cheese_lost_kg"}
    )
    agent.load_context_state = MagicMock(
        return_value={"table_artifact_id": "c", "chart_type": "scatter"}
    )

    result = await agent._execute({"skill_id": "sixes_plot_table"}, context, event_queue)

    assert result["status"] == "success", result
    assert result["chart_type"] == "scatter"
    assert result["y_columns"] == ["cheese_lost_kg"]


async def test_plot_unmatched_column_asks_instead_of_guessing(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "sightings.csv", CSV)

    result = await agent._execute(
        {
            "skill_id": "sixes_plot_table",
            "table_artifact_id": "c",
            "x_column": "night",
            "y_columns": ["pressure"],
        },
        context,
        event_queue,
    )

    assert result == {}
    assert "pressure" in agent.send_input_required.await_args.kwargs["hint"]


async def test_plot_llm_maps_user_words(context, event_queue, tmp_path):
    agent = _make_agent()
    agent._llm_text = AsyncMock(return_value='{"temperature": "tunnel_temperature_C"}')
    _serve(agent, tmp_path, "sightings.csv", CSV)

    result = await agent._execute(
        {
            "skill_id": "sixes_plot_table",
            "table_artifact_id": "c",
            "x_column": "night",
            "y_columns": ["temperature"],
        },
        context,
        event_queue,
    )

    assert result["y_columns"] == ["tunnel_temperature_C"]


async def test_plot_ignores_llm_invented_columns(context, event_queue, tmp_path):
    agent = _make_agent()
    agent._llm_text = AsyncMock(return_value='{"heat": "Heat (C)"}')
    _serve(agent, tmp_path, "sightings.csv", CSV)

    result = await agent._execute(
        {
            "skill_id": "sixes_plot_table",
            "table_artifact_id": "c",
            "x_column": "night",
            "y_columns": ["heat"],
        },
        context,
        event_queue,
    )

    assert result == {}  # asked the user instead


async def test_plot_text_only_column_asks_for_numeric(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "doc.md", ALMANAC.split("Some prose")[0])  # Table 1 only

    result = await agent._execute(
        {
            "skill_id": "sixes_plot_table",
            "table_artifact_id": "t",
            "x_column": "Call sign",
            "y_columns": ["Condition"],
        },
        context,
        event_queue,
    )

    assert result["status"] == "error"
    assert "numeric" in result["error"]


async def test_plot_rejects_bad_chart_type(context, event_queue):
    result = await _make_agent()._execute(
        {"skill_id": "sixes_plot_table", "table_artifact_id": "t", "chart_type": "pie"},
        context,
        event_queue,
    )

    assert result["status"] == "error"
    assert "chart_type" in result["error"]


async def test_plot_wrong_file_type(context, event_queue, tmp_path):
    agent = _make_agent()
    _serve(agent, tmp_path, "photo.png", b"\x89PNG\r\n\x1a\n\xff\xfe\x00")

    result = await agent._execute(
        {"skill_id": "sixes_plot_table", "table_artifact_id": "p", "x_column": "a", "y_columns": "b"},
        context,
        event_queue,
    )

    assert result["status"] == "error"
    assert "not a text table" in result["error"]


async def test_plot_upload_failure_is_reported(context, event_queue, tmp_path):
    agent = _make_agent()
    agent.send_deeplynx_artifact = AsyncMock(side_effect=RuntimeError("401"))
    _serve(agent, tmp_path, "sightings.csv", CSV)

    result = await agent._execute(
        {
            "skill_id": "sixes_plot_table",
            "table_artifact_id": "c",
            "x_column": "night",
            "y_columns": "mice_sighted",
        },
        context,
        event_queue,
    )

    assert result["status"] == "error"
    assert "DeepLynx" in result["error"]


def test_render_plot_skips_missing_values(tmp_path):
    table = parse_markdown_tables(ALMANAC)[1]
    png = tmp_path / "p.png"

    _make_agent()._render_plot(
        png,
        table["columns"],
        table["rows"],
        "Year",
        ["Camembert Ridge (residents)"],
        "line",
        table["caption"],
    )

    assert png.read_bytes().startswith(b"\x89PNG")
