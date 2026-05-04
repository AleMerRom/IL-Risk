from decimal import Decimal

import pytest

from il_risk.schemas import _coerce_rows, mint_burn_events_schema


def _mint_row(**overrides):
    row = {
        "block_number": 1,
        "block_timestamp": None,
        "tx_hash": "0x00",
        "log_index": 0,
        "event_type": "mint",
        "owner": "0x00",
        "tick_lower": 0,
        "tick_upper": 10,
        "liquidity_delta": Decimal(1),
        "liquidity_amount_raw": Decimal(1),
        "amount0_raw": Decimal(0),
        "amount1_raw": Decimal(0),
        "amount0_usdc": 0.0,
        "amount1_weth": 0.0,
        "date": "1970-01-01",
    }
    row.update(overrides)
    return row


def test_coerce_rows_rejects_negative_log_index():
    with pytest.raises(ValueError, match="negative log_index"):
        _coerce_rows([_mint_row(log_index=-1)], mint_burn_events_schema())


def test_coerce_rows_rejects_oversized_log_index():
    with pytest.raises(ValueError, match="oversized log_index"):
        _coerce_rows([_mint_row(log_index=2**32 - 1)], mint_burn_events_schema())


def test_coerce_rows_still_normalizes_tick_fields():
    row = _mint_row(tick_lower=2**32 - 10, tick_upper=2**32)

    fixed = _coerce_rows([row], mint_burn_events_schema())[0]

    assert fixed["tick_lower"] == -10
    assert fixed["tick_upper"] == 0
