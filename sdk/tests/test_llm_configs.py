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

"""Tests for installed LLM configuration profiles."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prometheus_api.llm_configs import LLM_CONFIG_PROFILES, get_llm_config_path


EXPECTED_FLAVORS = {"default", "small", "medium", "large", "vision", "embedding"}


def test_packaged_llm_config_paths_are_readable():
    """REQ-005-13, REQ-013-15: every installed profile resolves to readable YAML."""
    assert LLM_CONFIG_PROFILES == (
        "registry",
        "inl_hpc",
        "inl_litellm",
        "amsc",
        "anthropic",
        "openai",
        "anl_asksage",
        "azure",
        "gemma",
    )

    for profile in LLM_CONFIG_PROFILES:
        config_path = get_llm_config_path(profile)
        assert config_path == Path(config_path).resolve()
        assert config_path.is_file()
        assert yaml.safe_load(config_path.read_text(encoding="utf-8")) is not None


@pytest.mark.parametrize(
    ("profile", "api_key_references"),
    [
        ("inl_hpc", {"os.environ/INL_HPC_API_KEY"}),
        ("inl_litellm", {"os.environ/INL_LITELLM_API_KEY"}),
        ("amsc", {"os.environ/AMSC_I2_API_KEY"}),
        (
            "anthropic",
            {"os.environ/ANTHROPIC_API_KEY", "os.environ/EMBEDDING_API_KEY"},
        ),
        ("openai", {"os.environ/OPENAI_API_KEY"}),
        (
            "anl_asksage",
            {"os.environ/ASKSAGE_API_KEY", "os.environ/EMBEDDING_API_KEY"},
        ),
    ],
)
def test_provider_profiles_scaffold_every_flavor(profile, api_key_references):
    """REQ-005-16: provider profiles contain the complete semantic flavor contract."""
    payload = yaml.safe_load(get_llm_config_path(profile).read_text(encoding="utf-8"))
    models = payload["model_list"]

    assert {entry["model_name"] for entry in models} == EXPECTED_FLAVORS
    assert {entry["litellm_params"]["api_key"] for entry in models} == api_key_references


def test_unknown_packaged_llm_config_is_rejected():
    """REQ-005-13: arbitrary paths cannot escape the installed profile directory."""
    with pytest.raises(ValueError, match="Unknown LLM config profile"):
        get_llm_config_path("../private")
