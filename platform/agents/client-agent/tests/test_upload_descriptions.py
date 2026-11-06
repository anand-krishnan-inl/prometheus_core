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


"""Upload descriptions stay under the DeepLynx record limit (250 characters).

deeplynx-server v0.8.0 rejects a longer record description with HTTP 400, which fails the record update after
the file upload and drops the user's file. The LLM is asked for a description of each upload and can overshoot.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from file_upload_handler import (
    MAX_DESCRIPTION_CHARS,
    UploadedFile,
    _fallback_description,
    _limit_description,
    infer_file_descriptions,
)

# A real description the platform LLM wrote for this file (303 characters); its upload failed.
LONG = (
    "NumPy compressed archive that likely holds a sampled flow simulation case (the 'atr' name suggests a "
    "different geometry or dataset, with a60 possibly a parameter), probably with pressure and velocity field "
    "arrays; it is not a valve case, so it was likely uploaded as part of the same dataset collection."
)


class FakeLLM:
    def __init__(self, text: str):
        self.text, self.prompts = text, []

    async def medium(self, prompt, max_tokens=None):
        self.prompts.append(prompt)
        return SimpleNamespace(text=self.text)


def upload(name="atr__sample_00542_a60.npz"):
    return UploadedFile(name=name, mime_type="application/octet-stream", size_bytes=2_075_863, upload_id="u-1")


def test_limit_description():
    assert _limit_description("short") == "short"
    assert _limit_description("x" * MAX_DESCRIPTION_CHARS) == "x" * MAX_DESCRIPTION_CHARS
    cut = _limit_description(LONG)
    assert len(LONG) > MAX_DESCRIPTION_CHARS and len(cut) == MAX_DESCRIPTION_CHARS and cut.endswith("…")
    assert _limit_description("two\n  lines") == "two lines"
    assert len(_limit_description("é" * 400)) == MAX_DESCRIPTION_CHARS  # characters, not bytes


@pytest.mark.asyncio
async def test_llm_description_is_limited_and_prompt_states_the_limit():
    f = upload()
    llm = FakeLLM('{"%s": "%s"}' % (f.name, LONG))
    out = await infer_file_descriptions("visualize this valve case", [f], llm)
    assert len(out[f.name]) <= MAX_DESCRIPTION_CHARS and out[f.name].startswith("NumPy compressed archive")
    assert f"at most {MAX_DESCRIPTION_CHARS} characters" in llm.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["not json", '{"other.npz": "x"}', '{"%s": ""}'])
async def test_fallback_description_is_limited(reply):
    f = upload("v" * 400 + ".npz")  # a long filename makes the fallback itself too long
    out = await infer_file_descriptions("q", [f], FakeLLM(reply.replace("%s", f.name)))
    assert len(out[f.name]) <= MAX_DESCRIPTION_CHARS and out[f.name].startswith("User-uploaded")
    assert len(_fallback_description(f)) <= MAX_DESCRIPTION_CHARS
