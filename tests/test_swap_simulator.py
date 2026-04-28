from __future__ import annotations

from decimal import Decimal

import pandas as pd

from il_risk.swap_simulator import (
    LiquidityTick,
    Slot0Snapshot,
    simulate_swap,
    simulate_swap_from_parquet,
)
from il_risk.tickmath import get_sqrt_ratio_at_tick


CURRENT_TICK = 200_005
CURRENT_SQRT_PRICE_X96 = get_sqrt_ratio_at_tick(CURRENT_TICK)
MID_PRICE = 1_000_000_000_000 / (1.0001**CURRENT_TICK)


def _snapshot(liquidity: int = 10**24) -> tuple[list[LiquidityTick], Slot0Snapshot]:
    ticks = [
        LiquidityTick(tick=199_990, liquidity_net=liquidity, active_liquidity=liquidity),
        LiquidityTick(tick=200_000, liquidity_net=liquidity, active_liquidity=2 * liquidity),
        LiquidityTick(tick=200_010, liquidity_net=-liquidity, active_liquidity=liquidity),
        LiquidityTick(tick=200_020, liquidity_net=-liquidity, active_liquidity=0),
    ]
    slot0 = Slot0Snapshot(
        sqrt_price_x96=CURRENT_SQRT_PRICE_X96,
        current_tick=CURRENT_TICK,
        price_usdc_per_weth=MID_PRICE,
        snapshot_block=123,
        date="2026-01-01",
    )
    return ticks, slot0


def test_buy_weth_single_interval_swap() -> None:
    ticks, slot0 = _snapshot()

    result = simulate_swap(ticks, slot0, "buy_weth", 1_000)

    assert result.direction == "buy_weth"
    assert result.input_amount_usdc == 1_000
    assert result.output_amount_weth > 0
    assert result.output_amount_usdc == 0
    assert result.average_execution_price > result.pool_mid_price
    assert result.price_impact_bps > 5
    assert abs(result.slippage_bps - (result.price_impact_bps - 5)) < 1e-9
    assert result.tick_boundaries_crossed == 0
    assert result.final_sqrt_price_x96 < slot0.sqrt_price_x96


def test_sell_weth_single_interval_swap() -> None:
    ticks, slot0 = _snapshot()

    result = simulate_swap(ticks, slot0, "sell_weth", 1_000)

    assert result.direction == "sell_weth"
    assert result.input_amount_weth > 0
    assert result.output_amount_usdc > 0
    assert result.input_amount_usdc == 0
    assert result.average_execution_price < result.pool_mid_price
    assert result.price_impact_bps > 5
    assert result.tick_boundaries_crossed == 0
    assert result.final_sqrt_price_x96 > slot0.sqrt_price_x96


def test_multi_interval_swap_crosses_tick_boundaries() -> None:
    liquidity = 10**18
    ticks = [
        LiquidityTick(tick=199_980, liquidity_net=2 * liquidity, active_liquidity=2 * liquidity),
        LiquidityTick(tick=199_990, liquidity_net=0, active_liquidity=2 * liquidity),
        LiquidityTick(tick=200_000, liquidity_net=0, active_liquidity=2 * liquidity),
        LiquidityTick(tick=200_010, liquidity_net=0, active_liquidity=2 * liquidity),
        LiquidityTick(tick=200_020, liquidity_net=-2 * liquidity, active_liquidity=0),
    ]
    _, slot0 = _snapshot(liquidity=liquidity)

    buy = simulate_swap(ticks, slot0, "buy_weth", 30_000)
    sell = simulate_swap(ticks, slot0, "sell_weth", 30_000)

    assert buy.tick_boundaries_crossed >= 1
    assert sell.tick_boundaries_crossed >= 1
    assert buy.final_sqrt_price_x96 < slot0.sqrt_price_x96
    assert sell.final_sqrt_price_x96 > slot0.sqrt_price_x96


def test_fee_is_deducted_from_input_step() -> None:
    ticks, slot0 = _snapshot()

    no_fee = simulate_swap(ticks, slot0, "buy_weth", 1_000, fee_bps=0)
    with_fee = simulate_swap(ticks, slot0, "buy_weth", 1_000, fee_bps=5)

    assert with_fee.output_amount_weth < no_fee.output_amount_weth
    assert with_fee.average_execution_price > no_fee.average_execution_price
    assert with_fee.price_impact_bps > no_fee.price_impact_bps


def test_simulate_swap_from_parquet_loads_requested_snapshot(tmp_path) -> None:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        return

    ticks, slot0 = _snapshot()
    processed = tmp_path / "processed"
    processed.mkdir()

    liquidity_df = pd.DataFrame(
        [
            {
                "date": slot0.date,
                "snapshot_block": slot0.snapshot_block,
                "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "tick": tick.tick,
                "liquidityNet": Decimal(tick.liquidity_net),
                "liquidityGross": Decimal(abs(tick.liquidity_net)),
                "active_liquidity": Decimal(tick.active_liquidity),
                "price_lower": 0.0,
                "price_upper": 0.0,
            }
            for tick in ticks
        ]
    )
    slot0_df = pd.DataFrame(
        [
            {
                "date": slot0.date,
                "snapshot_block": slot0.snapshot_block,
                "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "sqrt_price_x96": Decimal(slot0.sqrt_price_x96),
                "price_usdc_per_weth": slot0.price_usdc_per_weth,
                "current_tick": slot0.current_tick,
            }
        ]
    )
    liquidity_df.to_parquet(processed / "liquidity_snapshots.parquet", index=False)
    slot0_df.to_parquet(processed / "slot0_snapshots.parquet", index=False)

    result = simulate_swap_from_parquet(
        direction="buy_weth",
        notional_usd=1_000,
        data_dir=tmp_path,
        snapshot_block=slot0.snapshot_block,
    )

    assert result.snapshot_block == slot0.snapshot_block
    assert result.date == slot0.date
    assert result.output_amount_weth > 0
