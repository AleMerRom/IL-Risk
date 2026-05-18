"""Module 4 — synthetic Uniswap V3 LP fee income and impermanent loss."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from decimal import Decimal, getcontext
import os
from math import ceil, floor, log, sqrt
from pathlib import Path
import tempfile
from typing import Iterable

import pandas as pd
import typer

from shared.constants import FEE_TIER, Q96, TICK_SPACING
from shared.uniswap_math import MAX_TICK, MIN_TICK, get_sqrt_ratio_at_tick

RAW_PRICE_SCALE = 10**12
USDC_RAW_SCALE = 10**6
WETH_RAW_SCALE = 10**18
DEFAULT_NOTIONAL_USD = 100_000.0
DEFAULT_RESULTS_DIR = Path("data/results/module_4")
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"
DEFAULT_PROCESSED_DIR = Path("data/processed")
getcontext().prec = 80

_DECIMAL_ONE = Decimal(1)
_Q96_DECIMAL = Decimal(Q96)
_USDC_RAW_SCALE_DECIMAL = Decimal(USDC_RAW_SCALE)
_WETH_RAW_SCALE_DECIMAL = Decimal(WETH_RAW_SCALE)
_RAW_PRICE_SCALE_DECIMAL = Decimal(RAW_PRICE_SCALE)

app = typer.Typer(no_args_is_help=True, add_completion=False)

_CACHE_ROOT = Path(tempfile.gettempdir()) / "il-risk-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPLCONFIGDIR = _CACHE_ROOT / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))


@dataclass(frozen=True)
class LPPosition:
    """Synthetic LP position sized at entry."""

    position_id: str
    price_range_label: str
    character: str
    width_pct: float | None
    tick_lower: int
    tick_upper: int
    price_lower_usdc_per_weth: float
    price_upper_usdc_per_weth: float
    entry_price_usdc_per_weth: float
    entry_tick: int
    liquidity_raw: float
    initial_weth: float
    initial_usdc: float
    initial_value_usd: float


@dataclass
class _ReplayState:
    sqrt_price: Decimal
    current_tick: int
    active_liquidity: Decimal


DEFAULT_POSITION_SPECS: tuple[tuple[str, float | None, str, str], ...] = (
    ("P1", 0.001, "+/-0.1%", "Ultra-narrow, market-maker style"),
    ("P2", 0.005, "+/-0.5%", "Narrow, active LP style"),
    ("P3", 0.02, "+/-2%", "Medium, typical retail LP"),
    ("P4", 0.10, "+/-10%", "Wide, passive LP"),
    ("P5", None, "Full range", "V2-equivalent, fully passive"),
)


def build_representative_positions(
    entry_price_usdc_per_weth: float,
    entry_tick: int,
    *,
    notional_usd: float = DEFAULT_NOTIONAL_USD,
    specs: Iterable[tuple[str, float | None, str, str]] = DEFAULT_POSITION_SPECS,
) -> pd.DataFrame:
    """Construct Task 4.1's five representative LP positions.

    The token accounting is done in human units, but ``liquidity_raw`` is the
    Uniswap liquidity value that can be compared with Swap.active_liquidity.
    """

    positions = [
        _build_position(
            position_id=position_id,
            width_pct=width_pct,
            price_range_label=price_range_label,
            character=character,
            entry_price_usdc_per_weth=entry_price_usdc_per_weth,
            entry_tick=entry_tick,
            notional_usd=notional_usd,
        )
        for position_id, width_pct, price_range_label, character in specs
    ]
    return pd.DataFrame(asdict(position) for position in positions)


def run_lp_analytics(
    *,
    data_dir: Path | str = Path("data"),
    slot0_path: Path | str | None = None,
    swaps_path: Path | str | None = None,
    liquidity_path: Path | str | None = None,
    mint_burns_path: Path | str | None = None,
    positions_output_path: Path | str | None = None,
    timeseries_output_path: Path | str | None = None,
    figs_dir: Path | str = DEFAULT_FIGURES_DIR,
    notional_usd: float = DEFAULT_NOTIONAL_USD,
    write_output: bool = True,
    write_figures: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Task 4.1-4.3 tables and figures from Module 1 parquet files."""

    base = Path(data_dir)
    slot0_path = Path(slot0_path) if slot0_path else base / "processed" / "slot0_snapshots.parquet"
    swaps_path = Path(swaps_path) if swaps_path else base / "processed" / "swap_events.parquet"
    liquidity_path = (
        Path(liquidity_path)
        if liquidity_path
        else base / "processed" / "liquidity_snapshots.parquet"
    )
    mint_burns_path = (
        Path(mint_burns_path)
        if mint_burns_path
        else base / "processed" / "mint_burn_events.parquet"
    )
    positions_output_path = (
        Path(positions_output_path)
        if positions_output_path
        else DEFAULT_RESULTS_DIR / "module4_lp_positions.parquet"
    )
    timeseries_output_path = (
        Path(timeseries_output_path)
        if timeseries_output_path
        else DEFAULT_RESULTS_DIR / "module4_lp_timeseries.parquet"
    )

    slot0 = pd.read_parquet(slot0_path).sort_values("snapshot_block").reset_index(drop=True)
    swaps = pd.read_parquet(swaps_path)
    liquidity_snapshots = pd.read_parquet(liquidity_path)
    mint_burns = pd.read_parquet(mint_burns_path)
    _validate_slot0(slot0)
    _validate_swaps(swaps)
    _validate_liquidity_snapshots(liquidity_snapshots)
    _validate_mint_burns(mint_burns)

    entry = slot0.iloc[0]
    positions = build_representative_positions(
        float(entry["price_usdc_per_weth"]),
        int(entry["current_tick"]),
        notional_usd=notional_usd,
    )

    principal = compute_lp_principal_timeseries(slot0, positions)
    fees = compute_fee_income_timeseries(
        swaps,
        slot0,
        positions,
        liquidity_snapshots,
        mint_burns,
    )
    timeseries = principal.merge(
        fees,
        on=["position_id", "date", "snapshot_block", "snapshot_timestamp"],
        how="left",
        validate="one_to_one",
    )
    timeseries["daily_fee_usd"] = timeseries["daily_fee_usd"].fillna(0.0)
    timeseries["cumulative_fee_usd"] = timeseries["cumulative_fee_usd"].fillna(0.0)
    for column in [
        "daily_fee0_usdc",
        "daily_fee1_weth",
        "cumulative_fee0_usdc",
        "cumulative_fee1_weth",
    ]:
        timeseries[column] = timeseries[column].fillna(0.0)
    timeseries["net_fee_minus_il_usd"] = timeseries["cumulative_fee_usd"] - timeseries["impermanent_loss_usd"]
    timeseries = timeseries[
        [
            "position_id",
            "date",
            "snapshot_block",
            "snapshot_timestamp",
            "price_usdc_per_weth",
            "current_tick",
            "lp_weth",
            "lp_usdc",
            "lp_value_usd",
            "hodl_value_usd",
            "impermanent_loss_usd",
            "daily_fee0_usdc",
            "daily_fee1_weth",
            "daily_fee_usd",
            "cumulative_fee0_usdc",
            "cumulative_fee1_weth",
            "cumulative_fee_usd",
            "net_fee_minus_il_usd",
        ]
    ].sort_values(["position_id", "snapshot_block"])

    if write_output:
        positions_output_path.parent.mkdir(parents=True, exist_ok=True)
        timeseries_output_path.parent.mkdir(parents=True, exist_ok=True)
        positions.to_parquet(positions_output_path, index=False)
        timeseries.to_parquet(timeseries_output_path, index=False)

    if write_figures:
        plot_module4_figures(timeseries, positions=positions, figs_dir=figs_dir)

    return positions, timeseries


def compute_fee_income_timeseries(
    swaps: pd.DataFrame,
    slot0: pd.DataFrame,
    positions: pd.DataFrame,
    liquidity_snapshots: pd.DataFrame,
    mint_burns: pd.DataFrame,
    *,
    fee_rate: float = FEE_TIER / 1_000_000,
) -> pd.DataFrame:
    """Compute cumulative LP fee income with interval-level fee-growth replay.

    Swap events contain post-swap tick and liquidity.  For fees, we instead
    start from the first daily liquidity snapshot, replay Mint/Burn events to
    keep the historical liquidity map current, and split each swap across pool
    tick boundaries plus the synthetic LP range boundaries.  For each interval,
    token0/token1 fees increment fee growth per unit liquidity.  Each synthetic
    LP is treated as a counterfactual single entrant, so its fee-growth
    denominator is ``L_pool + L_position`` while it is active.
    """

    snapshots = slot0[
        ["date", "snapshot_block", "snapshot_timestamp"]
    ].sort_values("snapshot_block").reset_index(drop=True)
    start_block = int(snapshots["snapshot_block"].min())
    end_block = int(snapshots["snapshot_block"].max())

    swaps_work = swaps[
        [
            "block_number",
            "log_index",
            "amount0_raw",
            "amount1_raw",
            "sqrt_price_x96",
            "tick",
            "amount0_usdc",
            "amount1_weth",
            "price_usdc_per_weth",
            "active_liquidity",
        ]
    ].copy()
    swaps_work["block_number"] = swaps_work["block_number"].astype("int64")
    swaps_work["log_index"] = swaps_work["log_index"].astype("int64")
    swaps_work["tick"] = swaps_work["tick"].astype("int64")
    positive_liquidity = swaps_work["active_liquidity"].map(_to_decimal) > 0
    swaps_work = swaps_work[
        (swaps_work["block_number"] > start_block)
        & (swaps_work["block_number"] <= end_block)
        & positive_liquidity
    ].sort_values(["block_number", "log_index"])

    position_records = positions.to_dict("records")
    liquidity_net_by_tick, boundaries, boundary_sqrts, state = _build_fee_replay_state(
        liquidity_snapshots,
        slot0,
        position_records,
        start_block,
    )
    fee_events = _replay_fee_events(
        swaps_work,
        mint_burns,
        position_records,
        liquidity_net_by_tick,
        boundaries,
        boundary_sqrts,
        state,
        start_block=start_block,
        end_block=end_block,
        fee_rate=Decimal(str(fee_rate)),
    )

    rows = [
        _fees_at_snapshots(snapshots, fee_events, str(position["position_id"]))
        for position in position_records
    ]
    return pd.concat(rows, ignore_index=True)[
        [
            "position_id",
            "date",
            "snapshot_block",
            "snapshot_timestamp",
            "daily_fee0_usdc",
            "daily_fee1_weth",
            "daily_fee_usd",
            "cumulative_fee0_usdc",
            "cumulative_fee1_weth",
            "cumulative_fee_usd",
        ]
    ]


def _build_fee_replay_state(
    liquidity_snapshots: pd.DataFrame,
    slot0: pd.DataFrame,
    positions: list[dict],
    start_block: int,
) -> tuple[dict[int, Decimal], list[int], list[Decimal], _ReplayState]:
    snapshot_liquidity = liquidity_snapshots[
        liquidity_snapshots["snapshot_block"].astype("int64") == start_block
    ].copy()
    if snapshot_liquidity.empty:
        raise ValueError(f"no liquidity snapshot found for start block {start_block}")

    liquidity_net_by_tick = {
        int(row["tick"]): _to_decimal(row["liquidityNet"])
        for row in snapshot_liquidity.to_dict("records")
        if _to_decimal(row["liquidityNet"]) != 0
    }
    boundaries = sorted(
        set(liquidity_net_by_tick)
        | {int(position["tick_lower"]) for position in positions}
        | {int(position["tick_upper"]) for position in positions}
    )
    boundary_sqrts = [_sqrt_decimal_at_tick(tick) for tick in boundaries]

    first_slot0 = slot0.sort_values("snapshot_block").iloc[0]
    current_tick = int(first_slot0["current_tick"])
    start_rows = snapshot_liquidity.sort_values("tick")
    active_candidates = start_rows[start_rows["tick"].astype("int64") <= current_tick]
    if active_candidates.empty:
        raise ValueError("cannot infer starting active liquidity from liquidity snapshot")
    active_liquidity = _to_decimal(active_candidates.iloc[-1]["active_liquidity"])
    state = _ReplayState(
        sqrt_price=_to_decimal(first_slot0["sqrt_price_x96"]) / _Q96_DECIMAL,
        current_tick=current_tick,
        active_liquidity=active_liquidity,
    )
    return liquidity_net_by_tick, boundaries, boundary_sqrts, state


def _replay_fee_events(
    swaps: pd.DataFrame,
    mint_burns: pd.DataFrame,
    positions: list[dict],
    liquidity_net_by_tick: dict[int, Decimal],
    boundaries: list[int],
    boundary_sqrts: list[Decimal],
    state: _ReplayState,
    *,
    start_block: int,
    end_block: int,
    fee_rate: Decimal,
) -> pd.DataFrame:
    events = _fee_replay_events(swaps, mint_burns, start_block=start_block, end_block=end_block)
    rows: list[dict] = []
    for event_type, event in events:
        if event_type == "mint_burn":
            _apply_mint_burn_event(event, state, liquidity_net_by_tick, boundaries, boundary_sqrts)
            continue
        rows.extend(
            _fee_rows_for_swap(
                event,
                positions,
                liquidity_net_by_tick,
                boundaries,
                boundary_sqrts,
                state,
                fee_rate=fee_rate,
            )
        )
        # The per-swap fee path is replayed on a local copy of the state; once
        # the swap is processed we re-anchor the global state to the event's
        # recorded post-swap values so the 6-month replay tracks on-chain state
        # exactly and cannot drift from accumulated floating-point error.
        state.sqrt_price = _to_decimal(event["sqrt_price_x96"]) / _Q96_DECIMAL
        state.current_tick = int(event["tick"])
        state.active_liquidity = _to_decimal(event["active_liquidity"])

    if not rows:
        return pd.DataFrame(
            columns=["block_number", "position_id", "fee0_usdc", "fee1_weth", "fee_usd"]
        )
    return pd.DataFrame(rows)


def _fee_replay_events(
    swaps: pd.DataFrame,
    mint_burns: pd.DataFrame,
    *,
    start_block: int,
    end_block: int,
) -> list[tuple[str, dict]]:
    swap_events = [
        ("swap", row)
        for row in swaps[
            (swaps["block_number"] > start_block)
            & (swaps["block_number"] <= end_block)
        ].to_dict("records")
    ]
    mint_burn_events = [
        ("mint_burn", row)
        for row in mint_burns[
            (mint_burns["block_number"].astype("int64") > start_block)
            & (mint_burns["block_number"].astype("int64") <= end_block)
        ].to_dict("records")
    ]
    return sorted(
        swap_events + mint_burn_events,
        key=lambda item: (int(item[1]["block_number"]), int(item[1]["log_index"])),
    )


def _apply_mint_burn_event(
    event: dict,
    state: _ReplayState,
    liquidity_net_by_tick: dict[int, Decimal],
    boundaries: list[int],
    boundary_sqrts: list[Decimal],
) -> None:
    tick_lower = int(event["tick_lower"])
    tick_upper = int(event["tick_upper"])
    delta = _to_decimal(event["liquidity_delta"])
    _add_liquidity_net(liquidity_net_by_tick, boundaries, boundary_sqrts, tick_lower, delta)
    _add_liquidity_net(liquidity_net_by_tick, boundaries, boundary_sqrts, tick_upper, -delta)
    if tick_lower <= state.current_tick < tick_upper:
        state.active_liquidity += delta


def _fee_rows_for_swap(
    swap: dict,
    positions: list[dict],
    liquidity_net_by_tick: dict[int, Decimal],
    boundaries: list[int],
    boundary_sqrts: list[Decimal],
    state: _ReplayState,
    *,
    fee_rate: Decimal,
) -> list[dict]:
    amount0_raw = _to_decimal(swap["amount0_raw"])
    amount1_raw = _to_decimal(swap["amount1_raw"])
    if amount0_raw > 0:
        direction = "buy_weth"
        remaining_input = amount0_raw
        input_scale = _USDC_RAW_SCALE_DECIMAL
    elif amount1_raw > 0:
        direction = "sell_weth"
        remaining_input = amount1_raw
        input_scale = _WETH_RAW_SCALE_DECIMAL
    else:
        return []

    rows: list[dict] = []
    local_state = _ReplayState(
        sqrt_price=state.sqrt_price,
        current_tick=state.current_tick,
        active_liquidity=state.active_liquidity,
    )
    while remaining_input > 0 and local_state.active_liquidity > 0:
        boundary_tick = _next_replay_boundary(
            boundaries,
            boundary_sqrts,
            local_state.sqrt_price,
            local_state.current_tick,
            direction,
        )
        if boundary_tick is None:
            gross_used = remaining_input
            crosses_tick = False
        else:
            boundary_sqrt = _sqrt_decimal_at_tick(boundary_tick)
            net_needed = _net_input_to_boundary(local_state, boundary_sqrt, direction)
            if net_needed <= 0:
                _cross_replay_tick(local_state, liquidity_net_by_tick, boundary_tick, direction)
                continue
            gross_needed = net_needed / (_DECIMAL_ONE - fee_rate)
            crosses_tick = remaining_input >= gross_needed
            gross_used = gross_needed if crosses_tick else remaining_input

        fee_raw = gross_used * fee_rate
        if fee_raw > 0:
            fee0_raw, fee1_raw = _fee_raw_by_token(fee_raw, input_scale)
            for position in positions:
                if _position_active(position, local_state.current_tick):
                    position_liquidity = _to_decimal(position["liquidity_raw"])
                    denominator = local_state.active_liquidity + position_liquidity
                    if denominator > 0:
                        fee0_growth = fee0_raw / denominator
                        fee1_growth = fee1_raw / denominator
                        fee0_usdc = position_liquidity * fee0_growth / _USDC_RAW_SCALE_DECIMAL
                        fee1_weth = position_liquidity * fee1_growth / _WETH_RAW_SCALE_DECIMAL
                        fee_usd = fee0_usdc + fee1_weth * _human_price_from_sqrt_decimal(
                            local_state.sqrt_price
                        )
                        rows.append(
                            {
                                "block_number": int(swap["block_number"]),
                                "position_id": position["position_id"],
                                "fee0_usdc": float(fee0_usdc),
                                "fee1_weth": float(fee1_weth),
                                "fee_usd": float(fee_usd),
                            }
                        )

        remaining_input -= gross_used
        if not crosses_tick:
            break
        _cross_replay_tick(local_state, liquidity_net_by_tick, boundary_tick, direction)

    return rows


def _fees_at_snapshots(
    snapshots: pd.DataFrame,
    fee_events: pd.DataFrame,
    position_id: str,
) -> pd.DataFrame:
    position_snapshots = snapshots.copy()
    position_events = fee_events[fee_events["position_id"] == position_id]
    if position_events.empty:
        position_snapshots["cumulative_fee0_usdc"] = 0.0
        position_snapshots["cumulative_fee1_weth"] = 0.0
        position_snapshots["cumulative_fee_usd"] = 0.0
    else:
        cumulative = (
            position_events.groupby("block_number", as_index=False)[
                ["fee0_usdc", "fee1_weth", "fee_usd"]
            ]
            .sum()
            .sort_values("block_number")
        )
        cumulative["cumulative_fee0_usdc"] = cumulative["fee0_usdc"].cumsum()
        cumulative["cumulative_fee1_weth"] = cumulative["fee1_weth"].cumsum()
        cumulative["cumulative_fee_usd"] = cumulative["fee_usd"].cumsum()
        position_snapshots = pd.merge_asof(
            position_snapshots,
            cumulative[
                [
                    "block_number",
                    "cumulative_fee0_usdc",
                    "cumulative_fee1_weth",
                    "cumulative_fee_usd",
                ]
            ],
            left_on="snapshot_block",
            right_on="block_number",
            direction="backward",
        )
        for column in ["cumulative_fee0_usdc", "cumulative_fee1_weth", "cumulative_fee_usd"]:
            position_snapshots[column] = position_snapshots[column].fillna(0.0)
        position_snapshots = position_snapshots.drop(columns=["block_number"])
    position_snapshots["daily_fee0_usdc"] = position_snapshots["cumulative_fee0_usdc"].diff().fillna(
        position_snapshots["cumulative_fee0_usdc"]
    )
    position_snapshots["daily_fee1_weth"] = position_snapshots["cumulative_fee1_weth"].diff().fillna(
        position_snapshots["cumulative_fee1_weth"]
    )
    position_snapshots["daily_fee_usd"] = position_snapshots["cumulative_fee_usd"].diff().fillna(
        position_snapshots["cumulative_fee_usd"]
    )
    position_snapshots["position_id"] = position_id
    return position_snapshots


def _add_liquidity_net(
    liquidity_net_by_tick: dict[int, Decimal],
    boundaries: list[int],
    boundary_sqrts: list[Decimal],
    tick: int,
    delta: Decimal,
) -> None:
    if tick not in liquidity_net_by_tick and tick not in boundaries:
        index = bisect_left(boundaries, tick)
        boundaries.insert(index, tick)
        boundary_sqrts.insert(index, _sqrt_decimal_at_tick(tick))
    liquidity_net_by_tick[tick] = liquidity_net_by_tick.get(tick, Decimal(0)) + delta


def _next_replay_boundary(
    boundaries: list[int],
    boundary_sqrts: list[Decimal],
    sqrt_price: Decimal,
    current_tick: int,
    direction: str,
) -> int | None:
    if direction == "buy_weth":
        index = bisect_left(boundary_sqrts, sqrt_price)
        if index < len(boundary_sqrts) and boundary_sqrts[index] == sqrt_price and boundaries[index] <= current_tick:
            return boundaries[index]
        return boundaries[index - 1] if index > 0 else None
    index = bisect_left(boundary_sqrts, sqrt_price)
    if index < len(boundary_sqrts) and boundary_sqrts[index] == sqrt_price and boundaries[index] > current_tick:
        return boundaries[index]
    index = bisect_right(boundary_sqrts, sqrt_price)
    return boundaries[index] if index < len(boundaries) else None


def _cross_replay_tick(
    state: _ReplayState,
    liquidity_net_by_tick: dict[int, Decimal],
    boundary_tick: int,
    direction: str,
) -> None:
    liquidity_net = liquidity_net_by_tick.get(boundary_tick, Decimal(0))
    if direction == "buy_weth":
        state.active_liquidity -= liquidity_net
        state.current_tick = boundary_tick - 1
    else:
        state.active_liquidity += liquidity_net
        state.current_tick = boundary_tick
    state.sqrt_price = _sqrt_decimal_at_tick(boundary_tick)


def _net_input_to_boundary(state: _ReplayState, boundary_sqrt: Decimal, direction: str) -> Decimal:
    if direction == "buy_weth":
        return state.active_liquidity * ((_DECIMAL_ONE / boundary_sqrt) - (_DECIMAL_ONE / state.sqrt_price))
    return state.active_liquidity * (boundary_sqrt - state.sqrt_price)


def _fee_raw_by_token(fee_raw: Decimal, input_scale: Decimal) -> tuple[Decimal, Decimal]:
    if input_scale == _USDC_RAW_SCALE_DECIMAL:
        return fee_raw, Decimal(0)
    return Decimal(0), fee_raw


def _position_active(position: dict, tick: int) -> bool:
    return int(position["tick_lower"]) <= tick < int(position["tick_upper"])


def _sqrt_decimal_at_tick(tick: int) -> Decimal:
    return Decimal(get_sqrt_ratio_at_tick(tick)) / _Q96_DECIMAL


def _human_price_from_sqrt_decimal(sqrt_price: Decimal) -> Decimal:
    return _RAW_PRICE_SCALE_DECIMAL / (sqrt_price * sqrt_price)


def _to_decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def compute_lp_principal_timeseries(slot0: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame:
    """Compute HODL value, LP principal value, and IL at all daily snapshots."""

    rows: list[dict] = []
    for position in positions.to_dict("records"):
        sqrt_lower = _sqrt_raw_price_at_tick(int(position["tick_lower"]))
        sqrt_upper = _sqrt_raw_price_at_tick(int(position["tick_upper"]))
        liquidity = float(position["liquidity_raw"])
        initial_weth = float(position["initial_weth"])
        initial_usdc = float(position["initial_usdc"])

        for snapshot in slot0.sort_values("snapshot_block").to_dict("records"):
            price = float(snapshot["price_usdc_per_weth"])
            sqrt_price = _sqrt_raw_price_from_human_price(price)
            amount0_raw, amount1_raw = _amounts_raw_for_liquidity(
                liquidity,
                sqrt_price,
                sqrt_lower,
                sqrt_upper,
            )
            lp_usdc = amount0_raw / USDC_RAW_SCALE
            lp_weth = amount1_raw / WETH_RAW_SCALE
            lp_value = lp_usdc + lp_weth * price
            hodl_value = initial_usdc + initial_weth * price
            rows.append(
                {
                    "position_id": position["position_id"],
                    "date": snapshot["date"],
                    "snapshot_block": int(snapshot["snapshot_block"]),
                    "snapshot_timestamp": snapshot["snapshot_timestamp"],
                    "price_usdc_per_weth": price,
                    "current_tick": int(snapshot["current_tick"]),
                    "lp_weth": lp_weth,
                    "lp_usdc": lp_usdc,
                    "lp_value_usd": lp_value,
                    "hodl_value_usd": hodl_value,
                    "impermanent_loss_usd": hodl_value - lp_value,
                }
            )

    return pd.DataFrame(rows)


def plot_module4_figures(
    timeseries: pd.DataFrame,
    *,
    positions: pd.DataFrame | None = None,
    figs_dir: Path | str = DEFAULT_FIGURES_DIR,
) -> tuple[Path, Path, Path]:
    """Write Module 4 deliverable figures."""

    import matplotlib.pyplot as plt

    figs_dir = Path(figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)
    label_map = _position_label_map(positions)

    fig1 = figs_dir / "module4_cumulative_fee_income.png"
    _plot_timeseries(
        timeseries,
        y="cumulative_fee_usd",
        ylabel="Cumulative fee income (USD)",
        title="Module 4 Fig 4.1 — Cumulative fee income",
        output_path=fig1,
        label_map=label_map,
        plt=plt,
    )
    fig2 = figs_dir / "module4_impermanent_loss.png"
    _plot_timeseries(
        timeseries,
        y="impermanent_loss_usd",
        ylabel="Impermanent loss (USD)",
        title="Module 4 Fig 4.2 — Impermanent loss",
        output_path=fig2,
        label_map=label_map,
        plt=plt,
    )
    fig3 = figs_dir / "module4_net_fee_minus_il.png"
    _plot_timeseries(
        timeseries,
        y="net_fee_minus_il_usd",
        ylabel="Cumulative fee income - IL (USD)",
        title="Module 4 Fig 4.3 — Fee income net of impermanent loss",
        output_path=fig3,
        label_map=label_map,
        plt=plt,
    )

    return fig1, fig2, fig3


def _build_position(
    *,
    position_id: str,
    width_pct: float | None,
    price_range_label: str,
    character: str,
    entry_price_usdc_per_weth: float,
    entry_tick: int,
    notional_usd: float,
) -> LPPosition:
    if width_pct is None:
        tick_lower = _ceil_to_tick_spacing(MIN_TICK)
        tick_upper = _floor_to_tick_spacing(MAX_TICK)
    else:
        price_lower = entry_price_usdc_per_weth * (1.0 - width_pct)
        price_upper = entry_price_usdc_per_weth * (1.0 + width_pct)
        tick_lower = _floor_to_tick_spacing(_tick_from_human_price(price_upper))
        tick_upper = _ceil_to_tick_spacing(_tick_from_human_price(price_lower))
        if tick_lower >= entry_tick:
            tick_lower = _floor_to_tick_spacing(entry_tick - TICK_SPACING)
        if tick_upper <= entry_tick:
            tick_upper = _ceil_to_tick_spacing(entry_tick + TICK_SPACING)

    tick_lower = max(_ceil_to_tick_spacing(MIN_TICK), tick_lower)
    tick_upper = min(_floor_to_tick_spacing(MAX_TICK), tick_upper)
    if tick_lower >= tick_upper:
        raise ValueError(f"invalid range for {position_id}: [{tick_lower}, {tick_upper}]")

    sqrt_price = _sqrt_raw_price_from_human_price(entry_price_usdc_per_weth)
    sqrt_lower = _sqrt_raw_price_at_tick(tick_lower)
    sqrt_upper = _sqrt_raw_price_at_tick(tick_upper)
    amount0_per_l, amount1_per_l = _amounts_raw_for_liquidity(1.0, sqrt_price, sqrt_lower, sqrt_upper)
    value_per_l = amount0_per_l / USDC_RAW_SCALE + (amount1_per_l / WETH_RAW_SCALE) * entry_price_usdc_per_weth
    if value_per_l <= 0:
        raise ValueError(f"position {position_id} has zero entry value per unit liquidity")

    liquidity = notional_usd / value_per_l
    amount0_raw, amount1_raw = _amounts_raw_for_liquidity(liquidity, sqrt_price, sqrt_lower, sqrt_upper)
    initial_usdc = amount0_raw / USDC_RAW_SCALE
    initial_weth = amount1_raw / WETH_RAW_SCALE

    price_at_tick_lower = _human_price_at_tick(tick_lower)
    price_at_tick_upper = _human_price_at_tick(tick_upper)
    return LPPosition(
        position_id=position_id,
        price_range_label=price_range_label,
        character=character,
        width_pct=width_pct,
        tick_lower=tick_lower,
        tick_upper=tick_upper,
        price_lower_usdc_per_weth=min(price_at_tick_lower, price_at_tick_upper),
        price_upper_usdc_per_weth=max(price_at_tick_lower, price_at_tick_upper),
        entry_price_usdc_per_weth=entry_price_usdc_per_weth,
        entry_tick=entry_tick,
        liquidity_raw=liquidity,
        initial_weth=initial_weth,
        initial_usdc=initial_usdc,
        initial_value_usd=initial_usdc + initial_weth * entry_price_usdc_per_weth,
    )


def _amounts_raw_for_liquidity(
    liquidity: float,
    sqrt_price: float,
    sqrt_lower: float,
    sqrt_upper: float,
) -> tuple[float, float]:
    if sqrt_lower <= 0 or sqrt_upper <= sqrt_lower:
        raise ValueError("invalid sqrt price range")
    if sqrt_price <= sqrt_lower:
        return liquidity * (sqrt_upper - sqrt_lower) / (sqrt_lower * sqrt_upper), 0.0
    if sqrt_price >= sqrt_upper:
        return 0.0, liquidity * (sqrt_upper - sqrt_lower)
    amount0 = liquidity * (sqrt_upper - sqrt_price) / (sqrt_price * sqrt_upper)
    amount1 = liquidity * (sqrt_price - sqrt_lower)
    return amount0, amount1


def _tick_from_human_price(price_usdc_per_weth: float) -> int:
    if price_usdc_per_weth <= 0:
        raise ValueError("price must be positive")
    return int(floor(log(RAW_PRICE_SCALE / price_usdc_per_weth) / log(1.0001)))


def _human_price_at_tick(tick: int) -> float:
    return RAW_PRICE_SCALE / (1.0001**tick)


def _sqrt_raw_price_at_tick(tick: int) -> float:
    return sqrt(1.0001**tick)


def _sqrt_raw_price_from_human_price(price_usdc_per_weth: float) -> float:
    if price_usdc_per_weth <= 0:
        raise ValueError("price must be positive")
    return sqrt(RAW_PRICE_SCALE / price_usdc_per_weth)


def _floor_to_tick_spacing(tick: int) -> int:
    return floor(tick / TICK_SPACING) * TICK_SPACING


def _ceil_to_tick_spacing(tick: int) -> int:
    return ceil(tick / TICK_SPACING) * TICK_SPACING


def _position_label_map(positions: pd.DataFrame | None) -> dict[str, str]:
    if positions is None:
        return {}
    return {
        row["position_id"]: f"{row['position_id']} ({row['price_range_label']})"
        for row in positions.to_dict("records")
    }


def _plot_timeseries(
    timeseries: pd.DataFrame,
    *,
    y: str,
    ylabel: str,
    title: str,
    output_path: Path,
    label_map: dict[str, str],
    plt,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for position_id, group in timeseries.groupby("position_id", sort=True):
        group = group.sort_values("snapshot_block")
        ax.plot(
            pd.to_datetime(group["snapshot_timestamp"]),
            group[y].astype(float),
            linewidth=2,
            label=label_map.get(position_id, position_id),
        )
    ax.axhline(0, color="#333333", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Snapshot date")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _validate_slot0(slot0: pd.DataFrame) -> None:
    required = {
        "date",
        "snapshot_block",
        "snapshot_timestamp",
        "sqrt_price_x96",
        "price_usdc_per_weth",
        "current_tick",
    }
    missing = required - set(slot0.columns)
    if missing:
        raise ValueError(f"slot0 snapshots missing columns: {sorted(missing)}")
    if slot0.empty:
        raise ValueError("slot0 snapshots are empty")


def _validate_swaps(swaps: pd.DataFrame) -> None:
    required = {
        "block_number",
        "log_index",
        "amount0_raw",
        "amount1_raw",
        "sqrt_price_x96",
        "tick",
        "amount0_usdc",
        "amount1_weth",
        "price_usdc_per_weth",
        "active_liquidity",
    }
    missing = required - set(swaps.columns)
    if missing:
        raise ValueError(f"swap events missing columns: {sorted(missing)}")
    if swaps.empty:
        raise ValueError("swap events are empty")


def _validate_liquidity_snapshots(liquidity_snapshots: pd.DataFrame) -> None:
    required = {"snapshot_block", "tick", "liquidityNet", "active_liquidity"}
    missing = required - set(liquidity_snapshots.columns)
    if missing:
        raise ValueError(f"liquidity snapshots missing columns: {sorted(missing)}")
    if liquidity_snapshots.empty:
        raise ValueError("liquidity snapshots are empty")


def _validate_mint_burns(mint_burns: pd.DataFrame) -> None:
    required = {"block_number", "log_index", "tick_lower", "tick_upper", "liquidity_delta"}
    missing = required - set(mint_burns.columns)
    if missing:
        raise ValueError(f"mint/burn events missing columns: {sorted(missing)}")


def _artifact_paths(processed_dir: Path, results_dir: Path, figures_dir: Path) -> dict[str, Path]:
    return {
        "slot0": processed_dir / "slot0_snapshots.parquet",
        "swaps": processed_dir / "swap_events.parquet",
        "liquidity": processed_dir / "liquidity_snapshots.parquet",
        "mint_burns": processed_dir / "mint_burn_events.parquet",
        "positions": results_dir / "module4_lp_positions.parquet",
        "timeseries": results_dir / "module4_lp_timeseries.parquet",
        "figures": figures_dir,
    }


def _require(path: Path, label: str) -> None:
    if not path.exists():
        typer.echo(f"missing {label}: {path}", err=True)
        raise typer.Exit(1)


def _row_count(path: Path) -> int:
    return len(pd.read_parquet(path))


@app.command("check-inputs")
def check_inputs(
    processed_dir: Path = typer.Option(DEFAULT_PROCESSED_DIR, "--processed-dir"),
) -> None:
    """Check that the parquet inputs needed by Module 4 are present."""

    paths = _artifact_paths(processed_dir, DEFAULT_RESULTS_DIR, DEFAULT_FIGURES_DIR)
    for label, path in [
        ("slot0 snapshots", paths["slot0"]),
        ("swap events", paths["swaps"]),
        ("liquidity snapshots", paths["liquidity"]),
        ("mint/burn events", paths["mint_burns"]),
    ]:
        _require(path, label)
        typer.echo(f"found {label}: {path} ({_row_count(path)} rows)")


@app.command("run-all")
def run_all(
    processed_dir: Path = typer.Option(DEFAULT_PROCESSED_DIR, "--processed-dir"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    figures_dir: Path = typer.Option(DEFAULT_FIGURES_DIR, "--figures-dir"),
    notional_usd: float = typer.Option(DEFAULT_NOTIONAL_USD, "--notional-usd"),
) -> None:
    """Generate all Module 4 tables and figures from existing parquet inputs."""

    paths = _artifact_paths(processed_dir, results_dir, figures_dir)
    _require(paths["slot0"], "slot0 snapshots")
    _require(paths["swaps"], "swap events")
    _require(paths["liquidity"], "liquidity snapshots")
    _require(paths["mint_burns"], "mint/burn events")

    positions, timeseries = run_lp_analytics(
        slot0_path=paths["slot0"],
        swaps_path=paths["swaps"],
        liquidity_path=paths["liquidity"],
        mint_burns_path=paths["mint_burns"],
        positions_output_path=paths["positions"],
        timeseries_output_path=paths["timeseries"],
        figs_dir=paths["figures"],
        notional_usd=notional_usd,
    )
    typer.echo(f"wrote LP positions: {paths['positions']} ({len(positions)} rows)")
    typer.echo(f"wrote LP analytics time series: {paths['timeseries']} ({len(timeseries)} rows)")
    typer.echo(f"wrote Module 4 figures in: {paths['figures']}")


if __name__ == "__main__":
    app()
