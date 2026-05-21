from __future__ import annotations

import pandas as pd

from module4.lp_analytics import build_representative_positions
from module5.hedge_backtest import run_hedge_backtest


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
