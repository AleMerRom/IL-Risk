from __future__ import annotations

import numpy as np

from module4.lp_analytics import build_representative_positions
from module5.lp_derivatives import (
    RAW_PRICE_SCALE,
    lp_delta_from_ticks,
    lp_gamma_from_ticks,
    sqrt_raw_price_at_tick,
)


def _human_price_at_tick(tick: int) -> float:
    sqrt_price = sqrt_raw_price_at_tick(tick)
    return RAW_PRICE_SCALE / sqrt_price**2


def test_lp_delta_is_continuous_at_tick_boundaries() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0)
    position = positions.loc[positions["position_id"] == "P3"].iloc[0]
    liquidity = float(position["liquidity_raw"])
    tick_lower = int(position["tick_lower"])
    tick_upper = int(position["tick_upper"])

    low_price_boundary = _human_price_at_tick(tick_upper)
    high_price_boundary = _human_price_at_tick(tick_lower)

    low_left = lp_delta_from_ticks(low_price_boundary * (1 - 1e-10), liquidity, tick_lower, tick_upper)
    low_exact = lp_delta_from_ticks(low_price_boundary, liquidity, tick_lower, tick_upper)
    low_right = lp_delta_from_ticks(low_price_boundary * (1 + 1e-10), liquidity, tick_lower, tick_upper)
    high_left = lp_delta_from_ticks(high_price_boundary * (1 - 1e-10), liquidity, tick_lower, tick_upper)
    high_exact = lp_delta_from_ticks(high_price_boundary, liquidity, tick_lower, tick_upper)
    high_right = lp_delta_from_ticks(high_price_boundary * (1 + 1e-10), liquidity, tick_lower, tick_upper)

    assert np.isclose(low_left, low_exact, rtol=0, atol=1e-5)
    assert np.isclose(low_right, low_exact, rtol=0, atol=1e-5)
    assert np.isclose(high_left, high_exact, rtol=0, atol=1e-5)
    assert np.isclose(high_right, high_exact, rtol=0, atol=1e-5)


def test_lp_gamma_is_zero_outside_active_range() -> None:
    positions = build_representative_positions(4_000.0, 193_300, notional_usd=100_000.0)
    position = positions.loc[positions["position_id"] == "P2"].iloc[0]
    liquidity = float(position["liquidity_raw"])
    tick_lower = int(position["tick_lower"])
    tick_upper = int(position["tick_upper"])

    low_price_boundary = _human_price_at_tick(tick_upper)
    high_price_boundary = _human_price_at_tick(tick_lower)
    outside_prices = np.array([low_price_boundary * 0.99, high_price_boundary * 1.01])

    gamma = lp_gamma_from_ticks(outside_prices, liquidity, tick_lower, tick_upper)

    assert np.array_equal(gamma, np.zeros_like(outside_prices))
