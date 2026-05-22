from __future__ import annotations

from decimal import Decimal

import pandas as pd

from module4.lp_analytics import (
    build_representative_positions,
    compute_fee_income_timeseries,
    compute_lp_principal_timeseries,
)
from shared.uniswap_math import get_sqrt_ratio_at_tick


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


def test_fee_income_adds_synthetic_liquidity_to_denominator() -> None:
    slot0 = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "snapshot_block": 100,
                "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(0)),
                "price_usdc_per_weth": 1_000_000_000_000.0,
                "current_tick": 0,
            },
            {
                "date": "2026-01-02",
                "snapshot_block": 200,
                "snapshot_timestamp": pd.Timestamp("2026-01-02", tz="UTC"),
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(0)),
                "price_usdc_per_weth": 1_000_000_000_000.0,
                "current_tick": 0,
            },
        ]
    )
    position_liquidity = Decimal("1e30")
    positions = pd.DataFrame(
        [
            {
                "position_id": "P1",
                "tick_lower": -10,
                "tick_upper": 10,
                "liquidity_raw": position_liquidity,
            }
        ]
    )
    liquidity_snapshots = pd.DataFrame(
        [
            {
                "snapshot_block": 100,
                "tick": -100,
                "liquidityNet": position_liquidity,
                "active_liquidity": position_liquidity,
            },
            {
                "snapshot_block": 100,
                "tick": 100,
                "liquidityNet": -position_liquidity,
                "active_liquidity": Decimal(0),
            },
        ]
    )
    swaps = pd.DataFrame(
        [
            {
                "block_number": 150,
                "log_index": 1,
                "amount0_raw": Decimal(10_000 * 10**6),
                "amount1_raw": Decimal("-1"),
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(0)),
                "tick": 0,
                "amount0_usdc": 10_000.0,
                "amount1_weth": -0.000001,
                "price_usdc_per_weth": 1_000_000_000_000.0,
                "active_liquidity": position_liquidity,
            },
        ]
    )
    mint_burns = pd.DataFrame(columns=["block_number", "log_index", "tick_lower", "tick_upper", "liquidity_delta"])

    result = compute_fee_income_timeseries(swaps, slot0, positions, liquidity_snapshots, mint_burns)
    p1_end = result[(result["position_id"] == "P1") & (result["snapshot_block"] == 200)].iloc[0]

    assert p1_end["cumulative_fee_usd"] == 2.5
    assert p1_end["cumulative_fee0_usdc"] == 2.5
    assert p1_end["cumulative_fee1_weth"] == 0.0
    assert p1_end["cumulative_fee_usd_brief"] == 5.0
    assert p1_end["cumulative_fee0_usdc_brief"] == 5.0
    assert p1_end["cumulative_fee1_weth_brief"] == 0.0


def test_fee_income_splits_swap_at_synthetic_range_boundary() -> None:
    slot0 = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "snapshot_block": 100,
                "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(5)),
                "price_usdc_per_weth": 1_000_000_000_000 / (1.0001**5),
                "current_tick": 5,
            },
            {
                "date": "2026-01-02",
                "snapshot_block": 200,
                "snapshot_timestamp": pd.Timestamp("2026-01-02", tz="UTC"),
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(20)),
                "price_usdc_per_weth": 1_000_000_000_000 / (1.0001**20),
                "current_tick": 20,
            },
        ]
    )
    pool_liquidity = Decimal("1e24")
    positions = pd.DataFrame(
        [
            {
                "position_id": "P1",
                "tick_lower": 0,
                "tick_upper": 10,
                "liquidity_raw": pool_liquidity,
            }
        ]
    )
    liquidity_snapshots = pd.DataFrame(
        [
            {
                "snapshot_block": 100,
                "tick": -100,
                "liquidityNet": pool_liquidity,
                "active_liquidity": pool_liquidity,
            },
            {
                "snapshot_block": 100,
                "tick": 100,
                "liquidityNet": -pool_liquidity,
                "active_liquidity": Decimal(0),
            },
        ]
    )
    amount1_raw = Decimal("1e21")
    swaps = pd.DataFrame(
        [
            {
                "block_number": 150,
                "log_index": 1,
                "amount0_raw": Decimal("-1"),
                "amount1_raw": amount1_raw,
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(20)),
                "tick": 20,
                "amount0_usdc": -1.0,
                "amount1_weth": float(amount1_raw / Decimal(10**18)),
                "price_usdc_per_weth": 1_000_000_000_000 / (1.0001**20),
                "active_liquidity": pool_liquidity,
            },
        ]
    )
    mint_burns = pd.DataFrame(columns=["block_number", "log_index", "tick_lower", "tick_upper", "liquidity_delta"])

    result = compute_fee_income_timeseries(swaps, slot0, positions, liquidity_snapshots, mint_burns)
    p1_end = result[(result["position_id"] == "P1") & (result["snapshot_block"] == 200)].iloc[0]
    whole_swap_half_share_fee = float((amount1_raw * Decimal("0.0005") / Decimal(10**18)) * Decimal(str(slot0.iloc[0]["price_usdc_per_weth"])) / 2)

    assert 0 < p1_end["cumulative_fee_usd"] < whole_swap_half_share_fee
    assert p1_end["cumulative_fee0_usdc"] == 0.0
    assert p1_end["cumulative_fee1_weth"] > 0
    assert p1_end["cumulative_fee_usd_brief"] > p1_end["cumulative_fee_usd"]


def test_fee_income_crosses_exact_upper_boundary_before_buy_interval() -> None:
    slot0 = pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "snapshot_block": 100,
                "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(10)),
                "price_usdc_per_weth": 1_000_000_000_000 / (1.0001**10),
                "current_tick": 10,
            },
            {
                "date": "2026-01-02",
                "snapshot_block": 200,
                "snapshot_timestamp": pd.Timestamp("2026-01-02", tz="UTC"),
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(-10)),
                "price_usdc_per_weth": 1_000_000_000_000 / (1.0001**-10),
                "current_tick": -10,
            },
        ]
    )
    pool_liquidity = Decimal("1e24")
    positions = pd.DataFrame(
        [
            {
                "position_id": "P1",
                "tick_lower": 0,
                "tick_upper": 10,
                "liquidity_raw": pool_liquidity,
            }
        ]
    )
    liquidity_snapshots = pd.DataFrame(
        [
            {
                "snapshot_block": 100,
                "tick": -100,
                "liquidityNet": pool_liquidity,
                "active_liquidity": pool_liquidity,
            },
            {
                "snapshot_block": 100,
                "tick": 100,
                "liquidityNet": -pool_liquidity,
                "active_liquidity": Decimal(0),
            },
        ]
    )
    amount0_raw = Decimal("1e21")
    swaps = pd.DataFrame(
        [
            {
                "block_number": 150,
                "log_index": 1,
                "amount0_raw": amount0_raw,
                "amount1_raw": Decimal("-1"),
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(-10)),
                "tick": -10,
                "amount0_usdc": float(amount0_raw / Decimal(10**6)),
                "amount1_weth": -1.0,
                "price_usdc_per_weth": 1_000_000_000_000 / (1.0001**-10),
                "active_liquidity": pool_liquidity,
            },
        ]
    )
    mint_burns = pd.DataFrame(columns=["block_number", "log_index", "tick_lower", "tick_upper", "liquidity_delta"])

    result = compute_fee_income_timeseries(swaps, slot0, positions, liquidity_snapshots, mint_burns)
    p1_end = result[(result["position_id"] == "P1") & (result["snapshot_block"] == 200)].iloc[0]

    assert p1_end["cumulative_fee_usd"] > 0
    assert p1_end["cumulative_fee0_usdc"] > 0
    assert p1_end["cumulative_fee1_weth"] == 0.0
    assert p1_end["cumulative_fee_usd_brief"] > p1_end["cumulative_fee_usd"]
