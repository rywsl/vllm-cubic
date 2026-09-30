# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Temporarily execute verifier tests whose source has a skip marker."""

from __future__ import annotations

import os


def pytest_collection_modifyitems(config, items):
    del config
    if os.environ.get("KVV_UNSKIP") != "1":
        return
    for item in items:
        item.own_markers[:] = [mark for mark in item.own_markers if mark.name != "skip"]
