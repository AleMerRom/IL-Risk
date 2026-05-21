from __future__ import annotations

import pandas as pd
import pytest

from module4.lp_analytics import build_representative_positions
from module5.hedge_backtest import (
    MonteCarloHedgeConfig,
    run_monte_carlo_hedge_backtest,
)
from module5.lp_derivatives import lp_delta_from_ticks


def _market(prices: list[float], funding_rates: list[float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    times = pd.date_range("2026-01-01", periods=len(prices), freq="h", tz="UTC")
    if funding_rates is None:
        funding_rates = [0.0] * len(prices)
    return (
        pd.DataFrame({"open_time": times, "close": prices}),
        pd.DataFrame(
            {
                "funding_time": times,
                "funding_rate": funding_rates,
                "oracle_price": prices,
            }
        ),
    )


def _one_position() -> pd.DataFrame:
    return build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0).head(1)


def test_zero_vol_monte_carlo_target_matches_current_lp_delta() -> None:
    prices, funding = _market([4_000.0, 4_100.0])
    positions = _one_position()
    position = positions.iloc[0]
    expected_delta = lp_delta_from_ticks(
        4_000.0,
        float(position["liquidity_raw"]),
        int(position["tick_lower"]),
        int(position["tick_upper"]),
    )

    result = run_monte_carlo_hedge_backtest(
        prices,
        funding,
        positions,
        config=MonteCarloHedgeConfig(
            rebalance_hours=(1,),
            annual_vol=0.0,
            annual_drift=0.0,
            mc_paths=8,
            trading_fee_rate=0.0,
        ),
    )

    first = result.iloc[0]
    assert first["rebalanced"] == True
    assert first["short_size_eth"] == pytest.approx(expected_delta)


def test_short_hedge_pnl_sign_is_negative_when_eth_rises() -> None:
    prices, funding = _market([4_000.0, 4_100.0])
    positions = _one_position()

    result = run_monte_carlo_hedge_backtest(
        prices,
        funding,
        positions,
        config=MonteCarloHedgeConfig(
            rebalance_hours=(1,),
            annual_vol=0.0,
            annual_drift=0.0,
            mc_paths=8,
            trading_fee_rate=0.0,
        ),
    )

    second = result.iloc[1]
    assert second["hedge_pnl_usd"] < 0
    assert second["hedge_pnl_usd"] == pytest.approx(
        -result.iloc[0]["short_size_eth"] * 100.0
    )


def test_positive_funding_rate_pays_short_hedger() -> None:
    prices, funding = _market([4_000.0, 4_000.0], funding_rates=[0.0, 0.001])
    positions = _one_position()

    result = run_monte_carlo_hedge_backtest(
        prices,
        funding,
        positions,
        config=MonteCarloHedgeConfig(
            rebalance_hours=(1,),
            annual_vol=0.0,
            annual_drift=0.0,
            mc_paths=8,
            trading_fee_rate=0.0,
        ),
    )

    second = result.iloc[1]
    assert second["funding_pnl_usd"] > 0
    assert second["funding_pnl_usd"] == pytest.approx(
        result.iloc[0]["short_size_eth"] * 4_000.0 * 0.001
    )


def test_rebalance_timing_respects_frequency() -> None:
    prices, funding = _market([4_000.0, 4_010.0, 4_020.0, 4_030.0, 4_040.0])
    positions = _one_position()

    result = run_monte_carlo_hedge_backtest(
        prices,
        funding,
        positions,
        config=MonteCarloHedgeConfig(
            rebalance_hours=(4,),
            annual_vol=0.0,
            annual_drift=0.0,
            mc_paths=8,
            trading_fee_rate=0.0,
        ),
    )

    assert list(result["rebalanced"]) == [True, False, False, False, True]
