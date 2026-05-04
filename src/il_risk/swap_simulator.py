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


@dataclass(frozen=True)
class SwapState:
    direction: Direction
    notional_usd: Decimal
    slot0: Slot0Snapshot
    ticks: list[LiquidityTick]
    tick_by_value: dict[int, LiquidityTick]
    initialized_ticks: list[int]
    sqrt_price: Decimal
    mid_price: Decimal
    liquidity: Decimal
    fee_bps: Decimal
    fee_fraction: Decimal
    remaining_input_raw: Decimal
    total_input_raw: Decimal
    total_output_raw: Decimal = Decimal(0)
    tick_boundaries_crossed: int = 0


@dataclass(frozen=True)
class SwapStep:
    next_sqrt: Decimal
    output_raw: Decimal
    gross_input_used: Decimal
    crosses_tick: bool
    boundary_tick: int | None = None


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

    state = _build_initial_state(liquidity_map, slot0, direction, notional_usd, fee_bps)
    final_state = _run_swap_loop(state)
    return _build_result(final_state)


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


def _build_initial_state(
    liquidity_map: Iterable[LiquidityTick | Mapping[str, object]],
    slot0: Slot0Snapshot | Mapping[str, object],
    direction: Direction,
    notional_usd: float | Decimal,
    fee_bps: float | Decimal,
) -> SwapState:
    if direction not in ("buy_weth", "sell_weth"):
        raise ValueError("direction must be 'buy_weth' or 'sell_weth'")
    notional = Decimal(str(notional_usd))
    if notional <= 0:
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
        remaining_input_raw = notional * _USDC_SCALE
    else:
        remaining_input_raw = (notional / mid_price) * _WETH_SCALE

    return SwapState(
        direction=direction,
        notional_usd=notional,
        slot0=slot,
        ticks=ticks,
        tick_by_value={t.tick: t for t in ticks},
        initialized_ticks=[t.tick for t in ticks],
        sqrt_price=sqrt_price,
        mid_price=mid_price,
        liquidity=liquidity,
        fee_bps=Decimal(str(fee_bps)),
        fee_fraction=fee_fraction,
        remaining_input_raw=remaining_input_raw,
        total_input_raw=remaining_input_raw,
    )


def _run_swap_loop(state: SwapState) -> SwapState:
    while state.remaining_input_raw > 0:
        step = _execute_step(state)
        state = _apply_step(state, step)
    return state


def _execute_step(state: SwapState) -> SwapStep:
    boundary_tick = _next_initialized_tick(
        state.initialized_ticks,
        state.sqrt_price,
        state.direction,
    )
    if boundary_tick is None:
        raise ValueError("trade exhausts available initialized liquidity")

    boundary_sqrt = Decimal(get_sqrt_ratio_at_tick(boundary_tick)) / _Q96
    net_needed = _net_input_needed_to_boundary(state, boundary_sqrt)
    gross_needed = net_needed / (Decimal(1) - state.fee_fraction)

    if state.remaining_input_raw >= gross_needed:
        output_raw = _output_to_boundary(state, boundary_sqrt)
        return SwapStep(
            next_sqrt=boundary_sqrt,
            output_raw=output_raw,
            gross_input_used=gross_needed,
            crosses_tick=True,
            boundary_tick=boundary_tick,
        )

    net_input = state.remaining_input_raw * (Decimal(1) - state.fee_fraction)
    next_sqrt = _next_sqrt_from_input(state, net_input)
    output_raw = _output_within_interval(state, next_sqrt)
    return SwapStep(
        next_sqrt=next_sqrt,
        output_raw=output_raw,
        gross_input_used=state.remaining_input_raw,
        crosses_tick=False,
    )


def _apply_step(state: SwapState, step: SwapStep) -> SwapState:
    liquidity = state.liquidity
    crossed = state.tick_boundaries_crossed
    if step.crosses_tick:
        if step.boundary_tick is None:
            raise ValueError("crossing step is missing boundary_tick")
        liquidity = _cross_tick(state, step.boundary_tick)
        crossed += 1

    remaining_input = state.remaining_input_raw - step.gross_input_used
    if liquidity <= 0 and remaining_input > 0:
        raise ValueError("trade crosses into zero-liquidity range")

    return SwapState(
        direction=state.direction,
        notional_usd=state.notional_usd,
        slot0=state.slot0,
        ticks=state.ticks,
        tick_by_value=state.tick_by_value,
        initialized_ticks=state.initialized_ticks,
        sqrt_price=step.next_sqrt,
        mid_price=state.mid_price,
        liquidity=liquidity,
        fee_bps=state.fee_bps,
        fee_fraction=state.fee_fraction,
        remaining_input_raw=remaining_input,
        total_input_raw=state.total_input_raw,
        total_output_raw=state.total_output_raw + step.output_raw,
        tick_boundaries_crossed=crossed,
    )


def _build_result(state: SwapState) -> SwapSimulationResult:
    final_sqrt_price_x96 = int((state.sqrt_price * _Q96).to_integral_value(rounding="ROUND_FLOOR"))
    final_tick = get_tick_at_sqrt_ratio(final_sqrt_price_x96)

    input_usdc = state.total_input_raw / _USDC_SCALE if state.direction == "buy_weth" else Decimal(0)
    input_weth = state.total_input_raw / _WETH_SCALE if state.direction == "sell_weth" else Decimal(0)
    output_usdc = state.total_output_raw / _USDC_SCALE if state.direction == "sell_weth" else Decimal(0)
    output_weth = state.total_output_raw / _WETH_SCALE if state.direction == "buy_weth" else Decimal(0)

    average_price = _average_execution_price(state.direction, input_usdc, input_weth, output_usdc, output_weth)
    impact_bps = _price_impact_bps(state.direction, average_price, state.mid_price)

    return SwapSimulationResult(
        direction=state.direction,
        notional_usd=float(state.notional_usd),
        average_execution_price=float(average_price),
        pool_mid_price=float(state.mid_price),
        price_impact_bps=float(impact_bps),
        slippage_bps=float(impact_bps - state.fee_bps),
        input_amount_usdc=float(input_usdc),
        input_amount_weth=float(input_weth),
        output_amount_usdc=float(output_usdc),
        output_amount_weth=float(output_weth),
        final_sqrt_price_x96=final_sqrt_price_x96,
        final_tick=final_tick,
        tick_boundaries_crossed=state.tick_boundaries_crossed,
        snapshot_block=state.slot0.snapshot_block,
        snapshot_timestamp=state.slot0.snapshot_timestamp,
        date=state.slot0.date,
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


def _net_input_needed_to_boundary(state: SwapState, boundary_sqrt: Decimal) -> Decimal:
    if state.direction == "buy_weth":
        return _amount0_delta(state.liquidity, boundary_sqrt, state.sqrt_price)
    return _amount1_delta(state.liquidity, state.sqrt_price, boundary_sqrt)


def _output_to_boundary(state: SwapState, boundary_sqrt: Decimal) -> Decimal:
    if state.direction == "buy_weth":
        return _amount1_delta(state.liquidity, boundary_sqrt, state.sqrt_price)
    return _amount0_delta(state.liquidity, state.sqrt_price, boundary_sqrt)


def _next_sqrt_from_input(state: SwapState, net_input: Decimal) -> Decimal:
    if state.direction == "buy_weth":
        return _next_sqrt_from_amount0(state.sqrt_price, state.liquidity, net_input)
    return _next_sqrt_from_amount1(state.sqrt_price, state.liquidity, net_input)


def _output_within_interval(state: SwapState, next_sqrt: Decimal) -> Decimal:
    if state.direction == "buy_weth":
        return _amount1_delta(state.liquidity, next_sqrt, state.sqrt_price)
    return _amount0_delta(state.liquidity, state.sqrt_price, next_sqrt)


def _cross_tick(state: SwapState, boundary_tick: int) -> Decimal:
    liquidity_net = Decimal(state.tick_by_value[boundary_tick].liquidity_net)
    if state.direction == "buy_weth":
        return state.liquidity - liquidity_net
    return state.liquidity + liquidity_net


def _average_execution_price(
    direction: Direction,
    input_usdc: Decimal,
    input_weth: Decimal,
    output_usdc: Decimal,
    output_weth: Decimal,
) -> Decimal:
    if direction == "buy_weth":
        return input_usdc / output_weth
    return output_usdc / input_weth


def _price_impact_bps(
    direction: Direction,
    average_price: Decimal,
    mid_price: Decimal,
) -> Decimal:
    if direction == "buy_weth":
        return ((average_price / mid_price) - Decimal(1)) * Decimal(10_000)
    return (Decimal(1) - (average_price / mid_price)) * Decimal(10_000)


def _next_sqrt_from_amount0(
    sqrt_price: Decimal,
    liquidity: Decimal,
    amount0_raw: Decimal,
) -> Decimal:
    return Decimal(1) / ((Decimal(1) / sqrt_price) + (amount0_raw / liquidity))


def _next_sqrt_from_amount1(
    sqrt_price: Decimal,
    liquidity: Decimal,
    amount1_raw: Decimal,
) -> Decimal:
    return sqrt_price + (amount1_raw / liquidity)


def _amount0_delta(liquidity: Decimal, sqrt_lower: Decimal, sqrt_upper: Decimal) -> Decimal:
    """Raw token0 amount between two prices."""

    return liquidity * ((Decimal(1) / sqrt_lower) - (Decimal(1) / sqrt_upper))


def _amount1_delta(liquidity: Decimal, sqrt_lower: Decimal, sqrt_upper: Decimal) -> Decimal:
    """Raw token1 amount between two prices."""

    return liquidity * (sqrt_upper - sqrt_lower)
