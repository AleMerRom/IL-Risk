from __future__ import annotations

import pandas as pd

from module4.lp_analytics import (
    build_representative_positions,
    compute_fee_income_timeseries,
    compute_lp_principal_timeseries,
)


def test_build_representative_positions_sizes_entry_value() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0)

    assert list(positions["position_id"]) == ["P1", "P2", "P3", "P4", "P5"]
    assert positions["liquidity_raw"].gt(0).all()
    assert positions["initial_weth"].ge(0).all()
    assert positions["initial_usdc"].ge(0).all()
    assert (positions["tick_lower"] < 193_300).all()
    assert (positions["tick_upper"] > 193_300).all()
    assert positions["initial_value_usd"].between(99_999.99, 100_000.01).all()


def test_lp_principal_has_zero_il_at_entry() -> None:
    slot0 = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "snapshot_block": 100,
                "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "price_usdc_per_weth": 4_000.0,
                "current_tick": 193_300,
            },
            {
                "date": "2026-01-02",
                "snapshot_block": 200,
                "snapshot_timestamp": pd.Timestamp("2026-01-02", tz="UTC"),
                "price_usdc_per_weth": 4_200.0,
                "current_tick": 192_810,
            },
        ]
    )
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0)

    result = compute_lp_principal_timeseries(slot0, positions)
    entry = result[result["snapshot_block"] == 100]

    assert len(result) == 10
    assert entry["lp_value_usd"].between(99_999.99, 100_000.01).all()
    assert entry["impermanent_loss_usd"].abs().le(0.01).all()


def test_fee_income_uses_only_in_range_swaps() -> None:
    slot0 = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "snapshot_block": 100,
                "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "price_usdc_per_weth": 4_000.0,
                "current_tick": 193_300,
            },
            {
                "date": "2026-01-02",
                "snapshot_block": 200,
                "snapshot_timestamp": pd.Timestamp("2026-01-02", tz="UTC"),
                "price_usdc_per_weth": 4_000.0,
                "current_tick": 193_300,
            },
        ]
    )
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0)
    p1 = positions.loc[positions["position_id"] == "P1"].iloc[0]
    swaps = pd.DataFrame(
        [
            {
                "block_number": 150,
                "tick": int(p1["tick_lower"]),
                "amount0_usdc": 10_000.0,
                "amount1_weth": -2.5,
                "price_usdc_per_weth": 4_000.0,
                "active_liquidity": float(p1["liquidity_raw"]) * 2,
            },
            {
                "block_number": 160,
                "tick": int(p1["tick_upper"]) + 100,
                "amount0_usdc": 10_000.0,
                "amount1_weth": -2.5,
                "price_usdc_per_weth": 4_000.0,
                "active_liquidity": float(p1["liquidity_raw"]) * 2,
            },
        ]
    )

    result = compute_fee_income_timeseries(swaps, slot0, positions)
    p1_end = result[(result["position_id"] == "P1") & (result["snapshot_block"] == 200)].iloc[0]

    assert p1_end["cumulative_fee_usd"] == 2.5
