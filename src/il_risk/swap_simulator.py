"""Module 3.1 — Uniswap V3 exact-input swap simulator.

The core simulator is intentionally parquet-free: callers pass one reconstructed
tick-level liquidity snapshot plus the matching slot0 row.  Thin adapters at the
bottom of this module load Module 1 parquet outputs for convenience.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from il_risk.constants import FEE_TIER, Q96, TOKEN0_DECIMALS, TOKEN1_DECIMALS
from il_risk.tickmath import get_sqrt_ratio_at_tick, get_tick_at_sqrt_ratio

getcontext().prec = 80

Direction = Literal["buy_weth", "sell_weth"]

_USDC_SCALE = Decimal(10) ** TOKEN0_DECIMALS
_WETH_SCALE = Decimal(10) ** TOKEN1_DECIMALS
_Q96 = Decimal(Q96)


@dataclass(frozen=True)
class LiquidityTick:
    """Initialized tick state for one liquidity snapshot."""

    tick: int
    liquidity_net: int
    active_liquidity: int


@dataclass(frozen=True)
class Slot0Snapshot:
    """Pool state at a snapshot block."""

    sqrt_price_x96: int
    current_tick: int
    price_usdc_per_weth: float
    snapshot_block: int | None = None
    snapshot_timestamp: object | None = None
    date: str | None = None


@dataclass(frozen=True)
class SwapSimulationResult:
    """Human-readable result for one exact-input simulated trade."""

    direction: Direction
    notional_usd: float
    average_execution_price: float
    pool_mid_price: float
    price_impact_bps: float
    slippage_bps: float
    input_amount_usdc: float
    input_amount_weth: float
    output_amount_usdc: float
    output_amount_weth: float
    final_sqrt_price_x96: int
    final_tick: int
    tick_boundaries_crossed: int
    snapshot_block: int | None = None
    snapshot_timestamp: object | None = None
    date: str | None = None


def simulate_swap(
    liquidity_map: Iterable[LiquidityTick | Mapping[str, object]],
    slot0: Slot0Snapshot | Mapping[str, object],
    direction: Direction,
    notional_usd: float | Decimal,
    *,
    fee_bps: float | Decimal = Decimal(FEE_TIER) / Decimal(100),
) -> SwapSimulationResult:
    """Simulate an exact-input Uniswap V3 swap from one reconstructed snapshot.

    ``buy_weth`` means USDC in and WETH out. ``sell_weth`` means WETH in and
    USDC out, sized so the WETH input is worth ``notional_usd`` at the pool mid.
    The 0.05% pool fee is deducted from each input step before price movement.
    """

    if direction not in ("buy_weth", "sell_weth"):
        raise ValueError("direction must be 'buy_weth' or 'sell_weth'")
    if Decimal(str(notional_usd)) <= 0:
        raise ValueError("notional_usd must be positive")

    ticks = _normalize_liquidity_map(liquidity_map)
    if not ticks:
        raise ValueError("liquidity_map is empty")
    slot = _normalize_slot0(slot0)
    sqrt_price = Decimal(slot.sqrt_price_x96) / _Q96
    mid_price = Decimal(str(slot.price_usdc_per_weth))
    fee_fraction = Decimal(str(fee_bps)) / Decimal(10_000)
    if fee_fraction < 0 or fee_fraction >= 1:
        raise ValueError("fee_bps must be in [0, 10000)")

    liquidity = Decimal(_active_liquidity_at(ticks, slot.current_tick))
    if liquidity <= 0:
        raise ValueError("active liquidity at current tick must be positive")

    if direction == "buy_weth":
        remaining_input_raw = Decimal(str(notional_usd)) * _USDC_SCALE
    else:
        remaining_input_raw = (Decimal(str(notional_usd)) / mid_price) * _WETH_SCALE

    total_input_raw = remaining_input_raw
    total_output_raw = Decimal(0)
    crossed = 0
    tick_by_value = {t.tick: t for t in ticks}
    initialized_ticks = [t.tick for t in ticks]

    while remaining_input_raw > 0:
        boundary_tick = _next_initialized_tick(initialized_ticks, sqrt_price, direction)
        if boundary_tick is None:
            raise ValueError("trade exhausts available initialized liquidity")

        boundary_sqrt = Decimal(get_sqrt_ratio_at_tick(boundary_tick)) / _Q96
        if direction == "buy_weth":
            net_needed = _amount0_delta(liquidity, boundary_sqrt, sqrt_price)
        else:
            net_needed = _amount1_delta(liquidity, sqrt_price, boundary_sqrt)
        gross_needed = net_needed / (Decimal(1) - fee_fraction)

        if remaining_input_raw >= gross_needed:
            remaining_input_raw -= gross_needed
            if direction == "buy_weth":
                total_output_raw += _amount1_delta(liquidity, boundary_sqrt, sqrt_price)
                liquidity -= Decimal(tick_by_value[boundary_tick].liquidity_net)
            else:
                total_output_raw += _amount0_delta(liquidity, sqrt_price, boundary_sqrt)
                liquidity += Decimal(tick_by_value[boundary_tick].liquidity_net)
            sqrt_price = boundary_sqrt
            crossed += 1
            if liquidity <= 0 and remaining_input_raw > 0:
                raise ValueError("trade crosses into zero-liquidity range")
            continue

        net_input = remaining_input_raw * (Decimal(1) - fee_fraction)
        remaining_input_raw = Decimal(0)
        if direction == "buy_weth":
            next_sqrt = Decimal(1) / ((Decimal(1) / sqrt_price) + (net_input / liquidity))
            total_output_raw += _amount1_delta(liquidity, next_sqrt, sqrt_price)
        else:
            next_sqrt = sqrt_price + (net_input / liquidity)
            total_output_raw += _amount0_delta(liquidity, sqrt_price, next_sqrt)
        sqrt_price = next_sqrt

    final_sqrt_price_x96 = int((sqrt_price * _Q96).to_integral_value(rounding="ROUND_FLOOR"))
    final_tick = get_tick_at_sqrt_ratio(final_sqrt_price_x96)

    input_usdc = total_input_raw / _USDC_SCALE if direction == "buy_weth" else Decimal(0)
    input_weth = total_input_raw / _WETH_SCALE if direction == "sell_weth" else Decimal(0)
    output_usdc = total_output_raw / _USDC_SCALE if direction == "sell_weth" else Decimal(0)
    output_weth = total_output_raw / _WETH_SCALE if direction == "buy_weth" else Decimal(0)

    if direction == "buy_weth":
        average_price = input_usdc / output_weth
        price_impact_bps = ((average_price / mid_price) - Decimal(1)) * Decimal(10_000)
    else:
        average_price = output_usdc / input_weth
        price_impact_bps = (Decimal(1) - (average_price / mid_price)) * Decimal(10_000)

    return SwapSimulationResult(
        direction=direction,
        notional_usd=float(notional_usd),
        average_execution_price=float(average_price),
        pool_mid_price=float(mid_price),
        price_impact_bps=float(price_impact_bps),
        slippage_bps=float(price_impact_bps - Decimal(str(fee_bps))),
        input_amount_usdc=float(input_usdc),
        input_amount_weth=float(input_weth),
        output_amount_usdc=float(output_usdc),
        output_amount_weth=float(output_weth),
        final_sqrt_price_x96=final_sqrt_price_x96,
        final_tick=final_tick,
        tick_boundaries_crossed=crossed,
        snapshot_block=slot.snapshot_block,
        snapshot_timestamp=slot.snapshot_timestamp,
        date=slot.date,
    )


def simulate_swap_from_parquet(
    *,
    direction: Direction,
    notional_usd: float | Decimal,
    data_dir: Path | str = Path("data"),
    liquidity_path: Path | str | None = None,
    slot0_path: Path | str | None = None,
    snapshot_block: int | None = None,
    date: str | None = None,
    fee_bps: float | Decimal = Decimal(FEE_TIER) / Decimal(100),
) -> SwapSimulationResult:
    """Load one snapshot from Module 1 parquet files and run ``simulate_swap``."""

    if (snapshot_block is None) == (date is None):
        raise ValueError("provide exactly one of snapshot_block or date")

    import pandas as pd

    base = Path(data_dir)
    liquidity_path = Path(liquidity_path) if liquidity_path else base / "processed" / "liquidity_snapshots.parquet"
    slot0_path = Path(slot0_path) if slot0_path else base / "processed" / "slot0_snapshots.parquet"

    liquidity_df = pd.read_parquet(liquidity_path)
    slot0_df = pd.read_parquet(slot0_path)
    if snapshot_block is not None:
        liquidity_df = liquidity_df[liquidity_df["snapshot_block"] == snapshot_block]
        slot0_df = slot0_df[slot0_df["snapshot_block"] == snapshot_block]
    else:
        liquidity_df = liquidity_df[liquidity_df["date"] == date]
        slot0_df = slot0_df[slot0_df["date"] == date]

    if liquidity_df.empty:
        raise ValueError("no liquidity snapshot rows match the requested snapshot")
    if len(slot0_df) != 1:
        raise ValueError(f"expected exactly one slot0 row, found {len(slot0_df)}")

    return simulate_swap(
        liquidity_df.to_dict("records"),
        slot0_df.iloc[0].to_dict(),
        direction,
        notional_usd,
        fee_bps=fee_bps,
    )


def _normalize_liquidity_map(
    liquidity_map: Iterable[LiquidityTick | Mapping[str, object]],
) -> list[LiquidityTick]:
    ticks: list[LiquidityTick] = []
    for row in liquidity_map:
        if isinstance(row, LiquidityTick):
            ticks.append(row)
        else:
            ticks.append(
                LiquidityTick(
                    tick=int(row["tick"]),
                    liquidity_net=int(row["liquidityNet"]),
                    active_liquidity=int(row["active_liquidity"]),
                )
            )
    return sorted(ticks, key=lambda t: t.tick)


def _normalize_slot0(slot0: Slot0Snapshot | Mapping[str, object]) -> Slot0Snapshot:
    if isinstance(slot0, Slot0Snapshot):
        return slot0
    return Slot0Snapshot(
        sqrt_price_x96=int(slot0["sqrt_price_x96"]),
        current_tick=int(slot0["current_tick"]),
        price_usdc_per_weth=float(slot0["price_usdc_per_weth"]),
        snapshot_block=int(slot0["snapshot_block"]) if "snapshot_block" in slot0 else None,
        snapshot_timestamp=slot0.get("snapshot_timestamp"),
        date=str(slot0["date"]) if "date" in slot0 else None,
    )


def _active_liquidity_at(ticks: Sequence[LiquidityTick], current_tick: int) -> int:
    candidates = [t.active_liquidity for t in ticks if t.tick <= current_tick]
    if not candidates:
        raise ValueError("current_tick is below the first initialized tick")
    return candidates[-1]


def _next_initialized_tick(
    initialized_ticks: Sequence[int],
    sqrt_price: Decimal,
    direction: Direction,
) -> int | None:
    if direction == "buy_weth":
        for tick in reversed(initialized_ticks):
            if Decimal(get_sqrt_ratio_at_tick(tick)) / _Q96 < sqrt_price:
                return tick
        return None
    for tick in initialized_ticks:
        if Decimal(get_sqrt_ratio_at_tick(tick)) / _Q96 > sqrt_price:
            return tick
    return None


def _amount0_delta(liquidity: Decimal, sqrt_lower: Decimal, sqrt_upper: Decimal) -> Decimal:
    """Raw token0 amount between two prices."""

    return liquidity * ((Decimal(1) / sqrt_lower) - (Decimal(1) / sqrt_upper))


def _amount1_delta(liquidity: Decimal, sqrt_lower: Decimal, sqrt_upper: Decimal) -> Decimal:
    """Raw token1 amount between two prices."""

    return liquidity * (sqrt_upper - sqrt_lower)
