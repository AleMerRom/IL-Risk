from __future__ import annotations

from decimal import Decimal

import pandas as pd

from il_risk.pipelines.module3.slippage_analysis import (
    assign_trade_size_bucket,
    compute_effective_spreads,
    plot_price_impact_curves,
    plot_effective_spread_comparison,
    run_simulation_grid,
    summarize_effective_spreads,
    summarize_price_impact,
)
from il_risk.uniswap_v3.math import get_sqrt_ratio_at_tick


def test_run_simulation_grid_with_synthetic_snapshots(tmp_path) -> None:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        return

    processed = tmp_path / "processed"
    processed.mkdir()
    liquidity_path = processed / "liquidity_snapshots.parquet"
    slot0_path = processed / "slot0_snapshots.parquet"
    output_path = processed / "simulated_trades.parquet"

    liquidity = 10**24
    blocks = [(123, "2026-01-01", 200_005), (124, "2026-01-02", 200_015)]
    liquidity_rows = []
    slot0_rows = []
    for block, date, current_tick in blocks:
        for tick, liquidity_net, active_liquidity in [
            (199_990, liquidity, liquidity),
            (200_000, liquidity, 2 * liquidity),
            (200_010, 0, 2 * liquidity),
            (200_020, -liquidity, liquidity),
            (200_030, -liquidity, 0),
        ]:
            liquidity_rows.append(
                {
                    "date": date,
                    "snapshot_block": block,
                    "snapshot_timestamp": pd.Timestamp(date, tz="UTC"),
                    "tick": tick,
                    "liquidityNet": Decimal(liquidity_net),
                    "liquidityGross": Decimal(abs(liquidity_net)),
                    "active_liquidity": Decimal(active_liquidity),
                    "price_lower": 0.0,
                    "price_upper": 0.0,
                }
            )
        slot0_rows.append(
            {
                "date": date,
                "snapshot_block": block,
                "snapshot_timestamp": pd.Timestamp(date, tz="UTC"),
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(current_tick)),
                "price_usdc_per_weth": 1_000_000_000_000 / (1.0001**current_tick),
                "current_tick": current_tick,
            }
        )

    pd.DataFrame(liquidity_rows).to_parquet(liquidity_path, index=False)
    pd.DataFrame(slot0_rows).to_parquet(slot0_path, index=False)

    result = run_simulation_grid(
        data_dir=tmp_path,
        trade_sizes_usd=(1_000, 10_000),
        output_path=output_path,
    )

    assert len(result) == 2 * 2 * 2
    assert output_path.exists()
    assert set(result["direction"]) == {"buy_weth", "sell_weth"}
    assert set(result["notional_usd"]) == {1_000.0, 10_000.0}
    assert set(result["snapshot_block"]) == {123, 124}
    assert result["average_execution_price"].notna().all()
    assert result["pool_mid_price"].notna().all()
    assert result["price_impact_bps"].notna().all()
    assert result["slippage_bps"].notna().all()
    assert result["tick_boundaries_crossed"].ge(0).all()


def test_run_simulation_grid_can_skip_writing(tmp_path) -> None:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        return

    processed = tmp_path / "processed"
    processed.mkdir()
    liquidity_path = processed / "liquidity_snapshots.parquet"
    slot0_path = processed / "slot0_snapshots.parquet"

    liquidity = 10**24
    pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "snapshot_block": 123,
                "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "tick": 199_990,
                "liquidityNet": Decimal(liquidity),
                "liquidityGross": Decimal(liquidity),
                "active_liquidity": Decimal(liquidity),
                "price_lower": 0.0,
                "price_upper": 0.0,
            },
            {
                "date": "2026-01-01",
                "snapshot_block": 123,
                "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "tick": 200_010,
                "liquidityNet": Decimal(-liquidity),
                "liquidityGross": Decimal(liquidity),
                "active_liquidity": Decimal(0),
                "price_lower": 0.0,
                "price_upper": 0.0,
            },
        ]
    ).to_parquet(liquidity_path, index=False)
    pd.DataFrame(
        [
            {
                "date": "2026-01-01",
                "snapshot_block": 123,
                "snapshot_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "sqrt_price_x96": Decimal(get_sqrt_ratio_at_tick(200_000)),
                "price_usdc_per_weth": 1_000_000_000_000 / (1.0001**200_000),
                "current_tick": 200_000,
            }
        ]
    ).to_parquet(slot0_path, index=False)

    result = run_simulation_grid(
        data_dir=tmp_path,
        trade_sizes_usd=(1_000,),
        directions=("buy_weth",),
        write_output=False,
    )

    assert len(result) == 1
    assert not (processed / "simulated_trades.parquet").exists()


def test_summarize_price_impact_from_dataframe() -> None:
    trades = pd.DataFrame(
        [
            {"direction": "buy_weth", "notional_usd": 1_000, "price_impact_bps": 10.0},
            {"direction": "buy_weth", "notional_usd": 1_000, "price_impact_bps": 20.0},
            {"direction": "buy_weth", "notional_usd": 1_000, "price_impact_bps": 30.0},
            {"direction": "sell_weth", "notional_usd": 1_000, "price_impact_bps": 12.0},
            {"direction": "sell_weth", "notional_usd": 1_000, "price_impact_bps": 22.0},
            {"direction": "sell_weth", "notional_usd": 1_000, "price_impact_bps": 32.0},
            {"direction": "buy_weth", "notional_usd": 10_000, "price_impact_bps": 100.0},
            {"direction": "sell_weth", "notional_usd": 10_000, "price_impact_bps": 110.0},
        ]
    )

    summary = summarize_price_impact(trades, write_output=False)
    buy_1k = summary[
        (summary["direction"] == "buy_weth") & (summary["notional_usd"] == 1_000)
    ].iloc[0]

    assert len(summary) == 4
    assert buy_1k["median_price_impact_bps"] == 20.0
    assert buy_1k["p10_price_impact_bps"] == 12.0
    assert buy_1k["p90_price_impact_bps"] == 28.0
    assert buy_1k["observations"] == 3


def test_plot_price_impact_curves_from_summary(tmp_path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        return

    summary = pd.DataFrame(
        [
            {
                "direction": direction,
                "notional_usd": size,
                "median_price_impact_bps": impact,
                "p10_price_impact_bps": impact * 0.8,
                "p90_price_impact_bps": impact * 1.2,
                "observations": 2,
            }
            for direction, base in (("buy_weth", 10.0), ("sell_weth", 12.0))
            for size, impact in ((1_000, base), (10_000, base * 5), (100_000, base * 20))
        ]
    )
    output_path = tmp_path / "module3_price_impact_curves.png"

    path = plot_price_impact_curves(summary, output_path=output_path)

    assert path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_compute_effective_spreads_from_dataframes() -> None:
    swaps = pd.DataFrame(
        [
            {
                "block_number": 100,
                "block_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "tx_hash": "0xabc",
                "log_index": 1,
                "amount0_usdc": 3_000.0,
                "amount1_weth": -1.0,
                "trade_direction": "buy_weth",
                "usd_notional": 3_000.0,
                "date": "2026-01-01",
            },
            {
                "block_number": 101,
                "block_timestamp": pd.Timestamp("2026-01-01", tz="UTC"),
                "tx_hash": "0xdef",
                "log_index": 2,
                "amount0_usdc": -2_980.0,
                "amount1_weth": 1.0,
                "trade_direction": "sell_weth",
                "usd_notional": 2_980.0,
                "date": "2026-01-01",
            },
        ]
    )
    mid_prices = pd.DataFrame(
        [
            {"block_number": 100, "mid_price_usdc_per_weth": 2_990.0},
            {"block_number": 101, "mid_price_usdc_per_weth": 2_990.0},
        ]
    )

    spreads = compute_effective_spreads(
        swaps,
        mid_prices,
        trade_size_buckets_usd=(1_000, 10_000),
        write_output=False,
    )

    buy = spreads[spreads["trade_direction"] == "buy_weth"].iloc[0]
    sell = spreads[spreads["trade_direction"] == "sell_weth"].iloc[0]
    expected_bps = 2 * 10 / 2_990 * 10_000
    assert abs(buy["effective_spread_bps"] - expected_bps) < 1e-12
    assert abs(sell["effective_spread_bps"] - expected_bps) < 1e-12
    assert buy["execution_price_usdc_per_weth"] == 3_000.0
    assert sell["execution_price_usdc_per_weth"] == 2_980.0
    assert set(spreads["size_bucket_usd"]) == {1_000.0}


def test_summarize_effective_spreads_from_dataframe() -> None:
    spreads = pd.DataFrame(
        [
            {"trade_direction": "buy_weth", "size_bucket_usd": 1_000, "effective_spread_bps": 10.0},
            {"trade_direction": "buy_weth", "size_bucket_usd": 1_000, "effective_spread_bps": 20.0},
            {"trade_direction": "buy_weth", "size_bucket_usd": 1_000, "effective_spread_bps": 30.0},
            {"trade_direction": "sell_weth", "size_bucket_usd": 10_000, "effective_spread_bps": 50.0},
        ]
    )

    summary = summarize_effective_spreads(spreads, write_output=False)
    buy_1k = summary[
        (summary["direction"] == "buy_weth") & (summary["notional_usd"] == 1_000)
    ].iloc[0]

    assert len(summary) == 2
    assert buy_1k["median_effective_spread_bps"] == 20.0
    assert buy_1k["p10_effective_spread_bps"] == 12.0
    assert buy_1k["p90_effective_spread_bps"] == 28.0
    assert buy_1k["observations"] == 3


def test_assign_trade_size_bucket_uses_log_midpoints() -> None:
    notionals = pd.Series([999, 3_000, 9_999, 30_000, 800_000])

    buckets = assign_trade_size_bucket(notionals, (1_000, 10_000, 100_000, 1_000_000))

    assert buckets.tolist() == [1_000.0, 1_000.0, 10_000.0, 10_000.0, 1_000_000.0]


def test_plot_effective_spread_comparison_from_summaries(tmp_path) -> None:
    try:
        import matplotlib  # noqa: F401
    except ModuleNotFoundError:
        return

    simulated = pd.DataFrame(
        [
            {
                "direction": direction,
                "notional_usd": size,
                "median_price_impact_bps": impact,
                "p10_price_impact_bps": impact * 0.8,
                "p90_price_impact_bps": impact * 1.2,
            }
            for direction, base in (("buy_weth", 8.0), ("sell_weth", 9.0))
            for size, impact in ((1_000, base), (10_000, base * 4))
        ]
    )
    empirical = pd.DataFrame(
        [
            {
                "direction": direction,
                "notional_usd": size,
                "median_effective_spread_bps": impact,
                "p10_effective_spread_bps": impact * 0.7,
                "p90_effective_spread_bps": impact * 1.3,
            }
            for direction, base in (("buy_weth", 10.0), ("sell_weth", 11.0))
            for size, impact in ((1_000, base), (10_000, base * 5))
        ]
    )
    output_path = tmp_path / "module3_effective_spread_by_size.png"

    path = plot_effective_spread_comparison(simulated, empirical, output_path=output_path)

    assert path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
