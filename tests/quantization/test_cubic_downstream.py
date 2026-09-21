# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tools.check_cubic_downstream import check_contracts


def test_downstream_contract_survives_upstream_sync() -> None:
    assert check_contracts() == []
