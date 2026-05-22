from __future__ import annotations

import pandas as pd
import pytest

from module4.lp_analytics import build_representative_positions
from module5.hedge_backtest import run_active_recenter_backtest, run_hedge_backtest


def _market_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.date_range("2026-01-01", periods=4, freq="h", tz="UTC")
    prices = pd.DataFrame(
        {
            "open_time": ts,
            "close_price": [4_000.0, 4_100.0, 4_050.0, 4_200.0],
        }
    )
    funding = pd.DataFrame(
        {
            "funding_time": ts,
            "funding_rate": [0.0, 0.001, -0.0005, 0.0],
            "oracle_price": [4_000.0, 4_100.0, 4_050.0, 4_200.0],
        }
    )
    return prices, funding


def test_hedge_pnl_sign_convention_for_short_delta_hedge() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0).head(1)
    prices, funding = _market_frames()

    result = run_hedge_backtest(
        positions,
        prices,
        funding,
        rebalance_hours=(24,),
        trading_fee_rate=0.0,
    )
    first_short = float(result.iloc[0]["short_size_eth"])

    assert result.iloc[1]["hedge_pnl_usd"] == -first_short * 100.0


def test_fixed_rebalance_timing() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0).head(1)
    prices, funding = _market_frames()

    result = run_hedge_backtest(
        positions,
        prices,
        funding,
        rebalance_hours=(2,),
        trading_fee_rate=0.0,
    )

    assert result["rebalanced"].tolist() == [True, False, True, False]


def test_short_funding_pnl_uses_previous_short_size_and_oracle_price() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0).head(1)
    prices, funding = _market_frames()

    result = run_hedge_backtest(
        positions,
        prices,
        funding,
        rebalance_hours=(24,),
        trading_fee_rate=0.0,
    )
    first_short = float(result.iloc[0]["short_size_eth"])

    assert result.iloc[1]["funding_pnl_usd"] == first_short * 4_100.0 * 0.001


def test_active_recenter_triggers_when_price_exits_range() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0).head(1)
    prices, funding = _market_frames()
    prices["close_price"] = [4_000.0, 4_500.0, 4_500.0, 4_500.0]
    funding["oracle_price"] = prices["close_price"]

    result = run_active_recenter_backtest(
        positions,
        prices,
        funding,
        include_passive=False,
        hedge_modes=("none",),
        trading_fee_rate=0.0,
        lp_swap_fee_rate=0.0,
        gas_cost_usd=0.0,
    )

    assert result["recentered"].any()
    assert int(result.iloc[-1]["range_recenter_count"]) >= 1
    assert int(result[result["recentered"]].iloc[0]["tick_lower"]) != int(
        positions.iloc[0]["tick_lower"]
    )


def test_hodl_tracking_hedge_targets_initial_weth_minus_lp_delta() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0).head(1)
    prices, funding = _market_frames()

    result = run_active_recenter_backtest(
        positions,
        prices,
        funding,
        include_active=False,
        hedge_modes=("hodl_tracking",),
        trading_fee_rate=0.0,
    )
    first = result.iloc[0]
    expected_short = float(first["target_delta_eth"]) - float(positions.iloc[0]["initial_weth"])

    assert first["short_size_eth"] == pytest.approx(expected_short)
    assert first["hodl_tracking_long_size_eth"] == pytest.approx(-expected_short)


def test_active_recenter_costs_reduce_redeployed_lp_value() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0).head(1)
    prices, funding = _market_frames()
    prices["close_price"] = [4_000.0, 3_500.0, 3_500.0, 3_500.0]
    funding["oracle_price"] = prices["close_price"]

    result = run_active_recenter_backtest(
        positions,
        prices,
        funding,
        include_passive=False,
        hedge_modes=("none",),
        trading_fee_rate=0.0,
        lp_swap_fee_rate=0.001,
        gas_cost_usd=10.0,
    )
    recentered = result[result["recentered"]].iloc[0]

    assert recentered["lp_rebalance_swap_notional_usd"] > 0
    assert recentered["lp_rebalance_cost_usd"] > 10.0
    assert recentered["lp_value_usd"] == pytest.approx(
        recentered["pre_recenter_lp_value_usd"] - recentered["lp_rebalance_cost_usd"]
    )


def test_threshold_recenter_can_trigger_before_range_exit() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0)
    p4 = positions[positions["position_id"] == "P4"]
    prices, funding = _market_frames()
    prices["close_price"] = [4_000.0, 4_330.0, 4_330.0, 4_330.0]
    funding["oracle_price"] = prices["close_price"]

    result = run_active_recenter_backtest(
        p4,
        prices,
        funding,
        include_passive=False,
        include_active=False,
        include_threshold=True,
        hedge_modes=("none",),
        trading_fee_rate=0.0,
        lp_swap_fee_rate=0.0,
        gas_cost_usd=0.0,
        expected_fee_window_hours=0,
    )
    recentered = result[result["recentered"]].iloc[0]

    assert recentered["range_active_before_recenter"] == True
    assert recentered["threshold_recenter_signal"] == True


def test_threshold_recenter_respects_cooldown() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0)
    p4 = positions[positions["position_id"] == "P4"]
    prices, funding = _market_frames()
    prices["close_price"] = [4_000.0, 4_330.0, 4_700.0, 4_700.0]
    funding["oracle_price"] = prices["close_price"]

    result = run_active_recenter_backtest(
        p4,
        prices,
        funding,
        include_passive=False,
        include_active=False,
        include_threshold=True,
        hedge_modes=("none",),
        trading_fee_rate=0.0,
        lp_swap_fee_rate=0.0,
        gas_cost_usd=0.0,
        expected_fee_window_hours=0,
        cooldown_hours=24,
    )

    assert int(result.iloc[-1]["range_recenter_count"]) == 1


def test_threshold_fee_filter_blocks_unfunded_recenter() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0)
    p4 = positions[positions["position_id"] == "P4"]
    prices, funding = _market_frames()
    prices["close_price"] = [4_000.0, 4_330.0, 4_330.0, 4_330.0]
    funding["oracle_price"] = prices["close_price"]

    result = run_active_recenter_backtest(
        p4,
        prices,
        funding,
        include_passive=False,
        include_active=False,
        include_threshold=True,
        hedge_modes=("none",),
        trading_fee_rate=0.0,
        lp_swap_fee_rate=0.0,
        gas_cost_usd=10.0,
        expected_fee_window_hours=24,
    )

    assert not result["recentered"].any()
    assert result.iloc[1]["threshold_recenter_signal"] == True
    assert result.iloc[1]["expected_fee_filter_passed"] == False


def test_hodl_tracking_no_trade_band_skips_small_hedge_adjustment() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0)
    p4 = positions[positions["position_id"] == "P4"]
    prices, funding = _market_frames()
    prices["close_price"] = [4_000.0, 4_010.0, 4_010.0, 4_010.0]
    funding["oracle_price"] = prices["close_price"]

    result = run_active_recenter_backtest(
        p4,
        prices,
        funding,
        include_active=False,
        hedge_modes=("hodl_tracking",),
        trading_fee_rate=0.0,
        hedge_no_trade_band_eth=0.5,
    )

    assert result.iloc[0]["hedge_rebalanced"] == True
    assert result.iloc[1]["hedge_rebalanced"] == False
    assert result.iloc[1]["delta_change_eth"] == 0.0
