"""Module 5 - Monte Carlo delta-hedging backtest.

The market-data collector enforces complete native Hyperliquid hourly data.
This module assumes those validated parquet files already exist, then tests a
Monte Carlo hedge rule for the Module 4 LP positions:

    target short ETH = E[LP delta at next rebalance price | current price]

The Monte Carlo rule is data-independent apart from the current price, so it
can be unit-tested without relaxing the Module 5 market-data constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil, floor, log
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from shared.constants import FEE_TIER, TICK_SPACING
from shared.uniswap_math import MAX_TICK, MIN_TICK
from module5.lp_derivatives import (
    RAW_PRICE_SCALE,
    WETH_RAW_SCALE,
    lp_delta_from_ticks,
    sqrt_raw_price_at_tick,
)

DEFAULT_POSITIONS_PATH = Path("data/results/module_4/module4_lp_positions.parquet")
DEFAULT_PRICES_PATH = Path("data/results/module_5/perp_prices.parquet")
DEFAULT_FUNDING_PATH = Path("data/results/module_5/funding_rates.parquet")
DEFAULT_RESULTS_DIR = Path("data/results/module_5")
DEFAULT_FEE_TIMESERIES_PATH = Path("data/results/module_4/module4_lp_timeseries.parquet")
DEFAULT_REBALANCE_HOURS = (1, 4, 24)
DEFAULT_NO_TRADE_BANDS = (0.05, 0.10)
DEFAULT_TRADING_FEE_RATE = 0.00045
DEFAULT_MC_PATHS = 10_000
DEFAULT_ANNUAL_VOL = 0.80
DEFAULT_ANNUAL_DRIFT = 0.0
DEFAULT_ACTIVE_RESULTS_PATH = DEFAULT_RESULTS_DIR / "active_recentered_lp_results.parquet"
DEFAULT_THRESHOLD_RESULTS_PATH = DEFAULT_RESULTS_DIR / "threshold_recentered_lp_results.parquet"
DEFAULT_ACTIVE_LP_SWAP_FEE_RATE = FEE_TIER / 1_000_000
DEFAULT_REBALANCE_GAS_USD = 25.0
DEFAULT_REBALANCE_SLIPPAGE_RATE = 0.0
DEFAULT_THRESHOLD_FRACTION = 0.50
DEFAULT_REBALANCE_COOLDOWN_HOURS = 24
DEFAULT_EXPECTED_FEE_WINDOW_HOURS = 24
DEFAULT_HEDGE_NO_TRADE_BAND_ETH = 5.0
HOURS_PER_YEAR = 365.0 * 24.0
USDC_RAW_SCALE = 10**6

_CACHE_ROOT = Path(tempfile.gettempdir()) / "il-risk-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

app = typer.Typer(no_args_is_help=True, add_completion=False)


@dataclass(frozen=True)
class MonteCarloHedgeConfig:
    """Configuration for the Monte Carlo hedge strategy."""

    rebalance_hours: tuple[int, ...] = DEFAULT_REBALANCE_HOURS
    trading_fee_rate: float = DEFAULT_TRADING_FEE_RATE
    mc_paths: int = DEFAULT_MC_PATHS
    annual_vol: float = DEFAULT_ANNUAL_VOL
    annual_drift: float = DEFAULT_ANNUAL_DRIFT
    seed: int = 5_003


def run_monte_carlo_hedge_backtest(
    prices: pd.DataFrame,
    funding: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    config: MonteCarloHedgeConfig = MonteCarloHedgeConfig(),
    fee_timeseries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return per-hour LP plus Monte Carlo hedge P&L for all positions.

    ``prices`` must contain hourly ``open_time`` and ``close`` columns.
    ``funding`` must contain hourly ``funding_time``, ``funding_rate``, and
    ``oracle_price`` columns. Funding P&L follows the project convention:
    positive funding means the short hedger receives funding.
    """

    _validate_config(config)
    market = _prepare_market_data(prices, funding)
    sorted_positions = positions.sort_values("position_id").reset_index(drop=True)
    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, object]] = []

    for position in sorted_positions.to_dict("records"):
        pid = str(position["position_id"])
        if fee_timeseries is not None:
            fee_series = _build_fee_series(fee_timeseries, pid, market["timestamp"])
        else:
            fee_series = None
        for rebalance_hours in config.rebalance_hours:
            rows.extend(
                _run_position_frequency(
                    market,
                    position,
                    rebalance_hours=rebalance_hours,
                    config=config,
                    rng=rng,
                    fee_series=fee_series,
                )
            )

    return pd.DataFrame(rows)


def _build_fee_series(
    fee_timeseries: pd.DataFrame,
    position_id: str,
    market_timestamps: pd.Series,
) -> pd.Series:
    """Forward-fill daily cumulative fee data to the hourly market timestamps.

    Returns a Series indexed by market_timestamps with cumulative fee USD values.
    If no data is found for the position, returns a Series of zeros.
    """
    pos_fees = fee_timeseries[fee_timeseries["position_id"] == position_id].copy()
    if pos_fees.empty:
        return pd.Series(0.0, index=market_timestamps)

    pos_fees["hourly_ts"] = pd.to_datetime(
        pos_fees["snapshot_timestamp"], utc=True
    ).dt.floor("h")
    daily_series = pos_fees.set_index("hourly_ts")["cumulative_fee_usd"]
    # Reindex to the hourly market timestamps then forward-fill
    combined_index = daily_series.index.union(market_timestamps)
    reindexed = daily_series.reindex(combined_index).ffill()
    result = reindexed.reindex(market_timestamps).fillna(0.0)
    result.index = market_timestamps
    return result


def run_hedge_backtest(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    rebalance_hours: tuple[int, ...] = DEFAULT_REBALANCE_HOURS,
    no_trade_bands: tuple[float, ...] = (),
    trading_fee_rate: float = DEFAULT_TRADING_FEE_RATE,
    fee_timeseries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run analytical delta hedges with fixed and no-trade-band rebalancing."""

    if any(hours <= 0 for hours in rebalance_hours):
        raise ValueError("rebalance_hours must be positive")
    if any(band < 0 for band in no_trade_bands):
        raise ValueError("no_trade_bands must be non-negative")
    if trading_fee_rate < 0:
        raise ValueError("trading_fee_rate must be non-negative")

    market = _prepare_market_data(prices, funding)
    rows: list[dict[str, object]] = []
    for position in positions.sort_values("position_id").to_dict("records"):
        pid = str(position["position_id"])
        if fee_timeseries is not None:
            fee_series = _build_fee_series(fee_timeseries, pid, market["timestamp"])
        else:
            fee_series = None
        for hours in rebalance_hours:
            rows.extend(
                _run_analytical_position_strategy(
                    market,
                    position,
                    strategy="fixed",
                    strategy_label=f"{hours}h",
                    rebalance_hours=hours,
                    no_trade_band=None,
                    trading_fee_rate=trading_fee_rate,
                    fee_series=fee_series,
                )
            )
        for band in no_trade_bands:
            rows.extend(
                _run_analytical_position_strategy(
                    market,
                    position,
                    strategy="no_trade_band",
                    strategy_label=f"band {band:.2f} ETH",
                    rebalance_hours=1,
                    no_trade_band=band,
                    trading_fee_rate=trading_fee_rate,
                    fee_series=fee_series,
                )
            )
    return pd.DataFrame(rows)


def run_active_recenter_backtest(
    positions: pd.DataFrame,
    prices: pd.DataFrame,
    funding: pd.DataFrame,
    *,
    swaps: pd.DataFrame | None = None,
    include_position_ids: tuple[str, ...] | None = None,
    include_passive: bool = True,
    include_active: bool = True,
    include_threshold: bool = False,
    hedge_modes: tuple[str, ...] = ("none", "hodl_tracking"),
    trading_fee_rate: float = DEFAULT_TRADING_FEE_RATE,
    lp_swap_fee_rate: float = DEFAULT_ACTIVE_LP_SWAP_FEE_RATE,
    gas_cost_usd: float = DEFAULT_REBALANCE_GAS_USD,
    slippage_rate: float = DEFAULT_REBALANCE_SLIPPAGE_RATE,
    fee_rate: float = DEFAULT_ACTIVE_LP_SWAP_FEE_RATE,
    adjust_fee_denominator: bool = True,
    threshold_fraction: float = DEFAULT_THRESHOLD_FRACTION,
    cooldown_hours: int = DEFAULT_REBALANCE_COOLDOWN_HOURS,
    expected_fee_window_hours: int = DEFAULT_EXPECTED_FEE_WINDOW_HOURS,
    hedge_no_trade_band_eth: float = 0.0,
) -> pd.DataFrame:
    """Backtest passive and exit-recentered LP ranges with optional HODL hedge.

    The active strategy rebuilds a finite-width range around the current hourly
    price whenever the previous range is out of range.  If ``include_threshold``
    is enabled, an additional cost-aware strategy recenters when price has
    drifted far enough from the range center, the cooldown has elapsed, and
    trailing in-range fees justify the estimated rebalance cost.  The hedge mode
    ``hodl_tracking`` implements the signed target in the report discussion:

        long hedge q_t = E_0 - Delta_LP,t

    Internally the existing project sign convention is also reported as
    ``short_size_eth = Delta_LP,t - E_0`` so positive values are short ETH and
    negative values are long ETH.

    Fee income is an hourly event-level proxy using historical swaps whose
    recorded post-swap tick lies inside the strategy's range.  It preserves the
    historical swap path and, by default, adds the synthetic LP liquidity to the
    denominator as in Module 4's adjusted fee calculation.
    """

    _validate_active_recenter_inputs(
        positions=positions,
        include_passive=include_passive,
        include_active=include_active,
        include_threshold=include_threshold,
        hedge_modes=hedge_modes,
        trading_fee_rate=trading_fee_rate,
        lp_swap_fee_rate=lp_swap_fee_rate,
        gas_cost_usd=gas_cost_usd,
        slippage_rate=slippage_rate,
        fee_rate=fee_rate,
        threshold_fraction=threshold_fraction,
        cooldown_hours=cooldown_hours,
        expected_fee_window_hours=expected_fee_window_hours,
        hedge_no_trade_band_eth=hedge_no_trade_band_eth,
    )
    market = _prepare_market_data(prices, funding)
    swap_fee_groups = _prepare_swap_fee_groups(swaps, fee_rate=fee_rate) if swaps is not None else {}

    finite_positions = _finite_width_positions(positions)
    if include_position_ids is not None:
        finite_positions = finite_positions[
            finite_positions["position_id"].astype(str).isin(set(include_position_ids))
        ]
    if finite_positions.empty:
        raise ValueError("no finite-width positions available for active recentering")

    rows: list[dict[str, object]] = []
    strategy_specs: list[tuple[str, str]] = []
    if include_passive:
        strategy_specs.append(("passive_fixed_range", "none"))
    if include_active:
        strategy_specs.append(("active_recenter_on_exit", "exit"))
    if include_threshold:
        strategy_specs.append(("active_threshold_fee_aware_recenter", "threshold"))

    for position in finite_positions.sort_values("position_id").to_dict("records"):
        for strategy, recenter_trigger in strategy_specs:
            for hedge_mode in hedge_modes:
                rows.extend(
                    _run_active_or_passive_position(
                        market,
                        position,
                        strategy=strategy,
                        recenter_trigger=recenter_trigger,
                        hedge_mode=hedge_mode,
                        swap_fee_groups=swap_fee_groups,
                        trading_fee_rate=trading_fee_rate,
                        lp_swap_fee_rate=lp_swap_fee_rate,
                        gas_cost_usd=gas_cost_usd,
                        slippage_rate=slippage_rate,
                        adjust_fee_denominator=adjust_fee_denominator,
                        threshold_fraction=threshold_fraction,
                        cooldown_hours=cooldown_hours,
                        expected_fee_window_hours=expected_fee_window_hours,
                        hedge_no_trade_band_eth=hedge_no_trade_band_eth,
                    )
                )

    return pd.DataFrame(rows)


def monte_carlo_target_short_size_eth(
    *,
    current_price: float,
    position: pd.Series | dict,
    horizon_hours: int,
    mc_paths: int,
    annual_vol: float,
    annual_drift: float = 0.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate the short ETH hedge target from simulated next-rebalance delta."""

    if rng is None:
        rng = np.random.default_rng()
    terminal_prices = simulate_gbm_terminal_prices(
        current_price=current_price,
        horizon_hours=horizon_hours,
        paths=mc_paths,
        annual_vol=annual_vol,
        annual_drift=annual_drift,
        rng=rng,
    )
    deltas = lp_delta_from_ticks(
        terminal_prices,
        float(position["liquidity_raw"]),
        int(position["tick_lower"]),
        int(position["tick_upper"]),
    )
    return float(np.mean(deltas))


def simulate_gbm_terminal_prices(
    *,
    current_price: float,
    horizon_hours: int,
    paths: int,
    annual_vol: float,
    annual_drift: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate GBM terminal prices over ``horizon_hours``."""

    if current_price <= 0:
        raise ValueError("current_price must be positive")
    if horizon_hours <= 0:
        raise ValueError("horizon_hours must be positive")
    if paths <= 0:
        raise ValueError("paths must be positive")
    if annual_vol < 0:
        raise ValueError("annual_vol must be non-negative")
    if rng is None:
        rng = np.random.default_rng()

    dt = horizon_hours / HOURS_PER_YEAR
    if annual_vol == 0:
        shock = np.zeros(paths)
    else:
        shock = rng.standard_normal(paths)
    log_return = (annual_drift - 0.5 * annual_vol**2) * dt + annual_vol * np.sqrt(dt) * shock
    return current_price * np.exp(log_return)


def write_hedge_results(results: pd.DataFrame, *, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    """Write per-step hedge results to ``hedge_results.parquet``."""

    path = results_dir / "hedge_results.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(path, index=False)
    return path


def plot_monte_carlo_hedge_results(
    results: pd.DataFrame,
    *,
    figures_dir: Path,
) -> Path:
    """Plot cumulative net LP plus hedge P&L by position and rebalance schedule."""

    if results.empty:
        raise ValueError("results cannot be empty")

    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "module5_monte_carlo_hedge_pnl.png"
    fig, ax = plt.subplots(figsize=(12, 6))

    final_rows = results.sort_values("timestamp").groupby(
        ["position_id", "rebalance_hours"], as_index=False
    ).tail(1)
    labels = {
        (str(row["position_id"]), int(row["rebalance_hours"])):
        f"{row['position_id']} / {int(row['rebalance_hours'])}h"
        for row in final_rows.to_dict("records")
    }
    for (position_id, rebalance_hours), group in results.groupby(
        ["position_id", "rebalance_hours"], sort=True
    ):
        group = group.sort_values("timestamp")
        ax.plot(
            group["timestamp"],
            group["net_lp_plus_hedge_pnl_usd"],
            linewidth=1.5,
            label=labels[(str(position_id), int(rebalance_hours))],
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    ax.set_title("Module 5 Monte Carlo LP Hedge Backtest")
    ax.set_xlabel("Time")
    ax.set_ylabel("Net LP plus hedge P&L (USD)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_hedge_results(results: pd.DataFrame, *, figures_dir: Path) -> Path:
    """Plot analytical hedge P&L by position and rebalance frequency."""

    if results.empty:
        raise ValueError("results cannot be empty")
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / "module5_hedge_results.png"
    fixed = results[results["strategy"] == "fixed"].copy()
    if fixed.empty:
        fixed = results.copy()
    positions = list(fixed["position_id"].drop_duplicates())
    fig, axes = plt.subplots(len(positions), 1, figsize=(11, 2.35 * len(positions)), sharex=True)
    axes_arr = np.atleast_1d(axes)
    for ax, position_id in zip(axes_arr, positions):
        group = fixed[fixed["position_id"] == position_id]
        for label, strategy_frame in group.groupby("strategy_label", sort=True):
            ax.plot(
                strategy_frame["timestamp"],
                strategy_frame["net_lp_plus_hedge_pnl_usd"],
                linewidth=1.6,
                label=label,
            )
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel(str(position_id))
        ax.grid(True, alpha=0.22, linestyle="--")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=8, ncol=3, framealpha=0.9)
    axes_arr[-1].set_xlabel("Time")
    fig.suptitle("LP Plus Perp Delta-Hedge P&L by Rebalance Frequency", y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_active_recenter_results(
    results: pd.DataFrame,
    *,
    figures_dir: Path,
    output_name: str = "module5_active_recenter_results.png",
) -> Path:
    """Plot terminal active/passive LP performance versus the original HODL basket."""

    if results.empty:
        raise ValueError("results cannot be empty")
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_path = figures_dir / output_name

    terminal = (
        results.sort_values("timestamp")
        .groupby(["position_id", "strategy", "hedge_mode"], as_index=False)
        .tail(1)
        .copy()
    )
    terminal["variant"] = terminal["strategy"].map(
        {
            "passive_fixed_range": "Passive",
            "active_recenter_on_exit": "Active",
            "active_threshold_fee_aware_recenter": "Threshold",
        }
    ) + terminal["hedge_mode"].map(
        {
            "none": "\nno hedge",
            "hodl_tracking": "\nHODL hedge",
        }
    )
    variant_order = [
        "Passive\nno hedge",
        "Passive\nHODL hedge",
        "Active\nno hedge",
        "Active\nHODL hedge",
        "Threshold\nno hedge",
        "Threshold\nHODL hedge",
    ]
    positions = list(terminal["position_id"].drop_duplicates())
    x = np.arange(len(positions))
    width = min(0.8 / len(variant_order), 0.19)

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, variant in enumerate(variant_order):
        group = terminal[terminal["variant"] == variant].set_index("position_id")
        values = [
            float(group.loc[position, "net_vs_hodl_usd"])
            if position in group.index
            else np.nan
            for position in positions
        ]
        ax.bar(
            x + (offset - (len(variant_order) - 1) / 2) * width,
            values,
            width=width,
            label=variant.replace("\n", ", "),
        )

    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(positions)
    ax.set_ylabel("Terminal net value vs HODL (USD)")
    ax.set_title("Active Exit-Recentering and HODL-Tracking Hedge Comparison")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"${value:,.0f}"))
    ax.grid(True, axis="y", alpha=0.22, linestyle="--")
    ax.legend(fontsize=9, ncol=2, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def compute_optional_research_extensions(
    results: pd.DataFrame,
    market: pd.DataFrame,
    *,
    large_move_quantile: float = 0.95,
) -> dict[str, object]:
    """Compute optional-extension diagnostics from completed backtests."""

    work = market.sort_values("timestamp").copy()
    work["log_return"] = np.log(work["close"]).diff()
    hourly_vol = float(work["log_return"].std(skipna=True))
    realized_vol_annualized = hourly_vol * np.sqrt(HOURS_PER_YEAR)
    threshold = float(work["log_return"].abs().quantile(large_move_quantile))

    funding = work.dropna(subset=["funding_rate", "log_return"]).copy()
    if funding.empty:
        conditional_funding = {}
    else:
        funding["large_abs_move"] = funding["log_return"].abs() >= threshold
        conditional_funding = (
            funding.groupby("large_abs_move")["funding_rate"]
            .agg(["count", "mean", "sum"])
            .rename(index={False: "normal_hours", True: "large_move_hours"})
            .to_dict(orient="index")
        )

    terminal = (
        results.sort_values("timestamp")
        .groupby(["position_id", "strategy_label"], as_index=False)
        .tail(1)
    )
    available_columns = [
        col
        for col in [
            "position_id",
            "strategy_label",
            "cumulative_trading_fee_usd",
            "cumulative_trading_fees_usd",
            "net_lp_plus_hedge_pnl_usd",
            "residual_gamma_pnl_proxy_usd",
        ]
        if col in terminal.columns
    ]

    return {
        "realized_volatility": {
            "hourly_log_return_std": hourly_vol,
            "annualized_from_hourly": float(realized_vol_annualized),
            "large_move_abs_log_return_threshold": threshold,
            "large_move_hour_count": int((work["log_return"].abs() >= threshold).sum()),
        },
        "conditional_funding": conditional_funding,
        "hedging_cost_summary": terminal[available_columns].to_dict(orient="records"),
        "lvr_note": (
            "residual_gamma_pnl_proxy_usd approximates LVR-style convexity loss "
            "with 0.5 * gamma * dP^2 between hedge updates; it is diagnostic, not "
            "a full arbitrage-flow LVR estimator."
        ),
        "options_overlay_note": (
            "P1 and P2 are the natural ETH options-overlay candidates because "
            "their gamma magnitude is highest per dollar of LP notional."
        ),
    }


def _run_active_or_passive_position(
    market: pd.DataFrame,
    initial_position: dict[str, object],
    *,
    strategy: str,
    recenter_trigger: str,
    hedge_mode: str,
    swap_fee_groups: dict[pd.Timestamp, dict[str, np.ndarray]],
    trading_fee_rate: float,
    lp_swap_fee_rate: float,
    gas_cost_usd: float,
    slippage_rate: float,
    adjust_fee_denominator: bool,
    threshold_fraction: float,
    cooldown_hours: int,
    expected_fee_window_hours: int,
    hedge_no_trade_band_eth: float,
) -> list[dict[str, object]]:
    position_id = str(initial_position["position_id"])
    width_pct = float(initial_position["width_pct"])
    current_position = _with_range_center(dict(initial_position))
    initial_usdc = float(initial_position["initial_usdc"])
    initial_weth = float(initial_position["initial_weth"])
    initial_price = float(market["close"].iloc[0])
    initial_lp_value = _lp_value_usd(initial_price, current_position)

    previous_price = initial_price
    short_size_eth = 0.0
    range_recenter_count = 0
    cumulative_lp_fee_income = 0.0
    cumulative_hedge_pnl = 0.0
    cumulative_funding_pnl = 0.0
    cumulative_perp_trading_fee = 0.0
    cumulative_lp_rebalance_cost = 0.0
    cumulative_lp_swap_fee = 0.0
    cumulative_lp_slippage_cost = 0.0
    cumulative_gas_cost = 0.0
    last_recenter_step = -10**9
    recent_fee_income: list[float] = []
    rows: list[dict[str, object]] = []

    for step, row in enumerate(market.to_dict("records")):
        timestamp = row["timestamp"]
        price = float(row["close"])
        oracle_price = float(row["oracle_price"])
        funding_rate = float(row["funding_rate"])
        price_change = 0.0 if step == 0 else price - previous_price

        hedge_pnl = -short_size_eth * price_change
        funding_pnl = short_size_eth * oracle_price * funding_rate
        cumulative_hedge_pnl += hedge_pnl
        cumulative_funding_pnl += funding_pnl

        range_active_before = _position_active_at_price(current_position, price)
        hourly_fee_income = _hourly_position_fee_income(
            swap_fee_groups,
            timestamp,
            current_position,
            adjust_fee_denominator=adjust_fee_denominator,
        )
        cumulative_lp_fee_income += hourly_fee_income
        recent_fee_income.append(hourly_fee_income)

        old_lp_usdc, old_lp_weth, pre_recenter_lp_value = _lp_amounts_usdc_weth(
            price,
            current_position,
        )
        recentered = False
        lp_rebalance_cost = 0.0
        lp_swap_fee = 0.0
        lp_slippage_cost = 0.0
        lp_gas_cost = 0.0
        lp_rebalance_swap_notional = 0.0
        range_center_price = _range_center_price(current_position)
        price_drift_from_center = abs(log(price / range_center_price))
        half_width_log = max(abs(log(1.0 - width_pct)), abs(log(1.0 + width_pct)))
        threshold_recenter_signal = (
            recenter_trigger == "threshold"
            and half_width_log > 0
            and price_drift_from_center >= threshold_fraction * half_width_log
        )
        cooldown_elapsed = (step - last_recenter_step) >= cooldown_hours
        trailing_fee_estimate = _trailing_sum(recent_fee_income, expected_fee_window_hours)
        expected_fee_filter_passed = True

        recenter_signal = recenter_trigger == "exit" and not range_active_before
        if threshold_recenter_signal:
            recenter_signal = cooldown_elapsed

        if recenter_signal:
            target_position = _build_position_for_notional(
                position_id=position_id,
                width_pct=width_pct,
                price_usdc_per_weth=price,
                notional_usd=pre_recenter_lp_value,
                template=initial_position,
            )
            target_usdc, target_weth, _ = _lp_amounts_usdc_weth(price, target_position)
            lp_rebalance_swap_notional = _token_rebalance_notional_usd(
                old_usdc=old_lp_usdc,
                old_weth=old_lp_weth,
                target_usdc=target_usdc,
                target_weth=target_weth,
                price_usdc_per_weth=price,
            )
            lp_swap_fee = lp_swap_fee_rate * lp_rebalance_swap_notional
            lp_slippage_cost = slippage_rate * lp_rebalance_swap_notional
            lp_gas_cost = gas_cost_usd
            lp_rebalance_cost = lp_swap_fee + lp_slippage_cost + lp_gas_cost
            expected_fee_filter_passed = (
                expected_fee_window_hours <= 0 or trailing_fee_estimate > lp_rebalance_cost
            )
            recenter_signal = recenter_trigger != "threshold" or expected_fee_filter_passed

        if recenter_signal:
            redeploy_notional = max(pre_recenter_lp_value - lp_rebalance_cost, 0.0)
            current_position = _build_position_for_notional(
                position_id=position_id,
                width_pct=width_pct,
                price_usdc_per_weth=price,
                notional_usd=redeploy_notional,
                template=initial_position,
            )
            recentered = True
            range_recenter_count += 1
            last_recenter_step = step
            cumulative_lp_rebalance_cost += lp_rebalance_cost
            cumulative_lp_swap_fee += lp_swap_fee
            cumulative_lp_slippage_cost += lp_slippage_cost
            cumulative_gas_cost += lp_gas_cost
            range_center_price = _range_center_price(current_position)
            price_drift_from_center = 0.0

        lp_usdc, lp_weth, lp_value = _lp_amounts_usdc_weth(price, current_position)
        target_delta = float(
            lp_delta_from_ticks(
                price,
                float(current_position["liquidity_raw"]),
                int(current_position["tick_lower"]),
                int(current_position["tick_upper"]),
            )
        )
        desired_short_size = _target_short_size_eth(
            hedge_mode=hedge_mode,
            lp_delta_eth=target_delta,
            initial_weth=initial_weth,
        )
        hedge_rebalanced = True
        if (
            hedge_mode != "none"
            and hedge_no_trade_band_eth > 0
            and step > 0
            and abs(desired_short_size - short_size_eth) < hedge_no_trade_band_eth
        ):
            target_short_size = short_size_eth
            hedge_rebalanced = False
        else:
            target_short_size = desired_short_size
        delta_change_eth = target_short_size - short_size_eth
        perp_trading_fee = trading_fee_rate * abs(delta_change_eth) * price
        short_size_eth = target_short_size
        cumulative_perp_trading_fee += perp_trading_fee
        net_hedge_pnl = cumulative_hedge_pnl + cumulative_funding_pnl - cumulative_perp_trading_fee

        hodl_value = initial_usdc + initial_weth * price
        gross_il = hodl_value - lp_value
        strategy_value = lp_value + cumulative_lp_fee_income + net_hedge_pnl
        net_vs_hodl = strategy_value - hodl_value
        residual_il = gross_il - net_hedge_pnl

        rows.append(
            {
                "timestamp": timestamp,
                "position_id": position_id,
                "price_range_label": initial_position.get("price_range_label", ""),
                "width_pct": width_pct,
                "strategy": strategy,
                "hedge_mode": hedge_mode,
                "strategy_label": _active_strategy_label(strategy, hedge_mode),
                "price_usdc_per_weth": price,
                "oracle_price": oracle_price,
                "funding_rate": funding_rate,
                "tick_lower": int(current_position["tick_lower"]),
                "tick_upper": int(current_position["tick_upper"]),
                "price_lower_usdc_per_weth": float(current_position["price_lower_usdc_per_weth"]),
                "price_upper_usdc_per_weth": float(current_position["price_upper_usdc_per_weth"]),
                "range_center_price_usdc_per_weth": range_center_price,
                "price_drift_from_center_log": price_drift_from_center,
                "threshold_recenter_signal": threshold_recenter_signal,
                "cooldown_elapsed": cooldown_elapsed,
                "trailing_fee_estimate_usd": trailing_fee_estimate,
                "expected_fee_filter_passed": expected_fee_filter_passed,
                "liquidity_raw": float(current_position["liquidity_raw"]),
                "range_active_before_recenter": range_active_before,
                "range_active_after_recenter": _position_active_at_price(current_position, price),
                "recentered": recentered,
                "range_recenter_count": range_recenter_count,
                "lp_weth": lp_weth,
                "lp_usdc": lp_usdc,
                "lp_value_usd": lp_value,
                "pre_recenter_lp_value_usd": pre_recenter_lp_value,
                "hodl_value_usd": hodl_value,
                "gross_il_usd": gross_il,
                "hourly_lp_fee_income_usd": hourly_fee_income,
                "lp_fee_income_usd": cumulative_lp_fee_income,
                "target_delta_eth": target_delta,
                "hodl_tracking_long_size_eth": initial_weth - target_delta,
                "desired_short_size_eth": desired_short_size,
                "short_size_eth": short_size_eth,
                "delta_change_eth": delta_change_eth,
                "hedge_rebalanced": hedge_rebalanced,
                "hedge_no_trade_band_eth": hedge_no_trade_band_eth,
                "hedge_pnl_usd": hedge_pnl,
                "funding_pnl_usd": funding_pnl,
                "perp_trading_fee_usd": perp_trading_fee,
                "cumulative_hedge_pnl_usd": cumulative_hedge_pnl,
                "cumulative_funding_pnl_usd": cumulative_funding_pnl,
                "cumulative_perp_trading_fee_usd": cumulative_perp_trading_fee,
                "net_hedge_pnl_usd": net_hedge_pnl,
                "lp_rebalance_swap_notional_usd": lp_rebalance_swap_notional,
                "lp_swap_fee_usd": lp_swap_fee,
                "lp_slippage_cost_usd": lp_slippage_cost,
                "gas_cost_usd": lp_gas_cost,
                "lp_rebalance_cost_usd": lp_rebalance_cost,
                "cumulative_lp_swap_fee_usd": cumulative_lp_swap_fee,
                "cumulative_lp_slippage_cost_usd": cumulative_lp_slippage_cost,
                "cumulative_gas_cost_usd": cumulative_gas_cost,
                "cumulative_lp_rebalance_cost_usd": cumulative_lp_rebalance_cost,
                "strategy_value_usd": strategy_value,
                "net_vs_hodl_usd": net_vs_hodl,
                "residual_il_usd": residual_il,
                "net_position_pnl_usd": cumulative_lp_fee_income - residual_il,
                "lp_pnl_usd": lp_value - initial_lp_value,
                "net_lp_plus_hedge_pnl_usd": lp_value - initial_lp_value + net_hedge_pnl,
            }
        )
        previous_price = price

    return rows


def _validate_active_recenter_inputs(
    *,
    positions: pd.DataFrame,
    include_passive: bool,
    include_active: bool,
    include_threshold: bool,
    hedge_modes: tuple[str, ...],
    trading_fee_rate: float,
    lp_swap_fee_rate: float,
    gas_cost_usd: float,
    slippage_rate: float,
    fee_rate: float,
    threshold_fraction: float,
    cooldown_hours: int,
    expected_fee_window_hours: int,
    hedge_no_trade_band_eth: float,
) -> None:
    _require_columns(
        positions,
        {
            "position_id",
            "width_pct",
            "tick_lower",
            "tick_upper",
            "liquidity_raw",
            "initial_usdc",
            "initial_weth",
        },
        "positions",
    )
    if not include_passive and not include_active and not include_threshold:
        raise ValueError(
            "at least one of include_passive/include_active/include_threshold must be true"
        )
    if not hedge_modes:
        raise ValueError("hedge_modes cannot be empty")
    invalid_modes = sorted(set(hedge_modes) - {"none", "hodl_tracking"})
    if invalid_modes:
        raise ValueError(f"unknown hedge modes: {invalid_modes}")
    if trading_fee_rate < 0:
        raise ValueError("trading_fee_rate must be non-negative")
    if lp_swap_fee_rate < 0:
        raise ValueError("lp_swap_fee_rate must be non-negative")
    if gas_cost_usd < 0:
        raise ValueError("gas_cost_usd must be non-negative")
    if slippage_rate < 0:
        raise ValueError("slippage_rate must be non-negative")
    if fee_rate < 0:
        raise ValueError("fee_rate must be non-negative")
    if not 0 < threshold_fraction <= 1:
        raise ValueError("threshold_fraction must be in (0, 1]")
    if cooldown_hours < 0:
        raise ValueError("cooldown_hours must be non-negative")
    if expected_fee_window_hours < 0:
        raise ValueError("expected_fee_window_hours must be non-negative")
    if hedge_no_trade_band_eth < 0:
        raise ValueError("hedge_no_trade_band_eth must be non-negative")


def _finite_width_positions(positions: pd.DataFrame) -> pd.DataFrame:
    return positions[positions["width_pct"].notna()].copy()


def _with_range_center(position: dict[str, object]) -> dict[str, object]:
    if "range_center_price_usdc_per_weth" not in position:
        position["range_center_price_usdc_per_weth"] = float(
            position["entry_price_usdc_per_weth"]
        )
    return position


def _range_center_price(position: dict[str, object]) -> float:
    return float(
        position.get(
            "range_center_price_usdc_per_weth",
            position["entry_price_usdc_per_weth"],
        )
    )


def _trailing_sum(values: list[float], window: int) -> float:
    if window <= 0:
        return 0.0
    return float(sum(values[-window:]))


def _prepare_swap_fee_groups(
    swaps: pd.DataFrame,
    *,
    fee_rate: float,
) -> dict[pd.Timestamp, dict[str, np.ndarray]]:
    time_col = _first_existing(swaps, ["block_timestamp", "timestamp", "time"])
    _require_columns(
        swaps,
        {"tick", "active_liquidity", "amount0_usdc", "amount1_weth", "price_usdc_per_weth"},
        "swaps",
    )
    work = swaps[
        [time_col, "tick", "active_liquidity", "amount0_usdc", "amount1_weth", "price_usdc_per_weth"]
    ].copy()
    work["timestamp"] = pd.to_datetime(work[time_col], utc=True).dt.floor("h")
    work["tick"] = pd.to_numeric(work["tick"], errors="raise").astype("int64")
    work["active_liquidity"] = pd.to_numeric(
        work["active_liquidity"],
        errors="coerce",
    ).astype(float)
    amount0_usdc = pd.to_numeric(work["amount0_usdc"], errors="coerce").astype(float)
    amount1_weth = pd.to_numeric(work["amount1_weth"], errors="coerce").astype(float)
    price = pd.to_numeric(work["price_usdc_per_weth"], errors="coerce").astype(float)
    input_usd = np.where(
        amount0_usdc > 0,
        amount0_usdc,
        np.where(amount1_weth > 0, amount1_weth * price, 0.0),
    )
    work["fee_usd"] = np.maximum(input_usd, 0.0) * fee_rate
    work = work[(work["fee_usd"] > 0) & (work["active_liquidity"] > 0)]

    groups: dict[pd.Timestamp, dict[str, np.ndarray]] = {}
    for timestamp, group in work.groupby("timestamp", sort=False):
        groups[pd.Timestamp(timestamp)] = {
            "tick": group["tick"].to_numpy(dtype=np.int64),
            "active_liquidity": group["active_liquidity"].to_numpy(dtype=float),
            "fee_usd": group["fee_usd"].to_numpy(dtype=float),
        }
    return groups


def _hourly_position_fee_income(
    swap_fee_groups: dict[pd.Timestamp, dict[str, np.ndarray]],
    timestamp: pd.Timestamp,
    position: dict[str, object],
    *,
    adjust_fee_denominator: bool,
) -> float:
    group = swap_fee_groups.get(pd.Timestamp(timestamp).floor("h"))
    if group is None:
        return 0.0
    tick_lower = int(position["tick_lower"])
    tick_upper = int(position["tick_upper"])
    active_mask = (group["tick"] >= tick_lower) & (group["tick"] < tick_upper)
    if not active_mask.any():
        return 0.0
    position_liquidity = float(position["liquidity_raw"])
    active_liquidity = group["active_liquidity"][active_mask]
    denominator = active_liquidity + position_liquidity if adjust_fee_denominator else active_liquidity
    valid = denominator > 0
    if not valid.any():
        return 0.0
    fee_usd = group["fee_usd"][active_mask][valid]
    share = position_liquidity / denominator[valid]
    return float(np.sum(fee_usd * share))


def _target_short_size_eth(
    *,
    hedge_mode: str,
    lp_delta_eth: float,
    initial_weth: float,
) -> float:
    if hedge_mode == "none":
        return 0.0
    if hedge_mode == "hodl_tracking":
        return lp_delta_eth - initial_weth
    raise ValueError(f"unknown hedge_mode: {hedge_mode}")


def _active_strategy_label(strategy: str, hedge_mode: str) -> str:
    strategy_label = {
        "passive_fixed_range": "Passive fixed range",
        "active_recenter_on_exit": "Active exit-recentered range",
        "active_threshold_fee_aware_recenter": "Active threshold fee-aware range",
    }.get(strategy, strategy)
    hedge_label = {
        "none": "no hedge",
        "hodl_tracking": "HODL hedge",
    }.get(hedge_mode, hedge_mode)
    return f"{strategy_label}, {hedge_label}"


def _run_analytical_position_strategy(
    market: pd.DataFrame,
    position: dict[str, object],
    *,
    strategy: str,
    strategy_label: str,
    rebalance_hours: int,
    no_trade_band: float | None,
    trading_fee_rate: float,
    fee_series: pd.Series | None = None,
) -> list[dict[str, object]]:
    initial_price = float(market["close"].iloc[0])
    initial_lp_value = _lp_value_usd(initial_price, position)
    initial_weth = float(position["initial_weth"])
    initial_usdc = float(position["initial_usdc"])
    previous_price = initial_price
    short_size_eth = 0.0
    cumulative_hedge_pnl = 0.0
    cumulative_funding_pnl = 0.0
    cumulative_trading_fee = 0.0
    cumulative_residual_gamma_proxy = 0.0
    rows: list[dict[str, object]] = []

    # Build a mapping from timestamp -> fee_income for fast lookup
    fee_lookup: dict = {}
    if fee_series is not None:
        fee_lookup = fee_series.to_dict()

    for step, row in enumerate(market.to_dict("records")):
        price = float(row["close"])
        funding_rate = float(row["funding_rate"])
        oracle_price = float(row["oracle_price"])
        target_delta = float(
            lp_delta_from_ticks(
                price,
                float(position["liquidity_raw"]),
                int(position["tick_lower"]),
                int(position["tick_upper"]),
            )
        )
        price_change = 0.0 if step == 0 else price - previous_price
        hedge_pnl = -short_size_eth * price_change
        funding_pnl = short_size_eth * oracle_price * funding_rate

        if strategy == "fixed":
            rebalanced = step % rebalance_hours == 0
        elif strategy == "no_trade_band":
            rebalanced = step == 0 or abs(target_delta - short_size_eth) >= float(no_trade_band)
        else:
            raise ValueError(f"unknown strategy: {strategy}")

        delta_change_eth = 0.0
        trading_fee = 0.0
        if rebalanced:
            delta_change_eth = target_delta - short_size_eth
            trading_fee = trading_fee_rate * abs(delta_change_eth) * price
            short_size_eth = target_delta

        gamma = _lp_gamma(price, position)
        residual_gamma_proxy = 0.5 * gamma * price_change**2
        lp_value = _lp_value_usd(price, position)
        lp_pnl = lp_value - initial_lp_value
        cumulative_hedge_pnl += hedge_pnl
        cumulative_funding_pnl += funding_pnl
        cumulative_trading_fee += trading_fee
        cumulative_residual_gamma_proxy += residual_gamma_proxy
        net_hedge_pnl = cumulative_hedge_pnl + cumulative_funding_pnl - cumulative_trading_fee

        # New derived columns
        hodl_value_usd = initial_weth * price + initial_usdc
        gross_il_usd = hodl_value_usd - lp_value
        ts_key = row["timestamp"]
        lp_fee_income_usd = float(fee_lookup.get(ts_key, 0.0))
        residual_il_usd = gross_il_usd - net_hedge_pnl
        net_position_pnl_usd = lp_fee_income_usd - residual_il_usd

        rows.append(
            {
                "timestamp": row["timestamp"],
                "position_id": position["position_id"],
                "rebalance_hours": rebalance_hours if strategy == "fixed" else pd.NA,
                "no_trade_band_eth": no_trade_band if strategy == "no_trade_band" else pd.NA,
                "strategy": strategy,
                "strategy_label": strategy_label,
                "price_usdc_per_weth": price,
                "oracle_price": oracle_price,
                "funding_rate": funding_rate,
                "target_delta_eth": target_delta,
                "short_size_eth": short_size_eth,
                "delta_change_eth": delta_change_eth,
                "rebalanced": rebalanced,
                "hedge_pnl_usd": hedge_pnl,
                "funding_pnl_usd": funding_pnl,
                "trading_fee_usd": trading_fee,
                "lp_value_usd": lp_value,
                "lp_pnl_usd": lp_pnl,
                "cumulative_hedge_pnl_usd": cumulative_hedge_pnl,
                "cumulative_funding_pnl_usd": cumulative_funding_pnl,
                "cumulative_trading_fee_usd": cumulative_trading_fee,
                "net_hedge_pnl_usd": net_hedge_pnl,
                "net_lp_plus_hedge_pnl_usd": lp_pnl + net_hedge_pnl,
                "lp_gamma_weth_per_usdc": gamma,
                "residual_gamma_pnl_proxy_usd": cumulative_residual_gamma_proxy,
                "hodl_value_usd": hodl_value_usd,
                "gross_il_usd": gross_il_usd,
                "lp_fee_income_usd": lp_fee_income_usd,
                "residual_il_usd": residual_il_usd,
                "net_position_pnl_usd": net_position_pnl_usd,
            }
        )
        previous_price = price

    return rows


def _run_position_frequency(
    market: pd.DataFrame,
    position: dict[str, object],
    *,
    rebalance_hours: int,
    config: MonteCarloHedgeConfig,
    rng: np.random.Generator,
    fee_series: pd.Series | None = None,
) -> list[dict[str, object]]:
    position_id = str(position["position_id"])
    initial_price = float(market["close"].iloc[0])
    initial_lp_value = _lp_value_usd(initial_price, position)
    initial_weth = float(position["initial_weth"])
    initial_usdc = float(position["initial_usdc"])
    previous_price = initial_price
    short_size_eth = 0.0
    cumulative_hedge_pnl = 0.0
    cumulative_funding_pnl = 0.0
    cumulative_trading_fees = 0.0
    rows: list[dict[str, object]] = []

    fee_lookup: dict = {}
    if fee_series is not None:
        fee_lookup = fee_series.to_dict()

    for step, row in enumerate(market.to_dict("records")):
        price = float(row["close"])
        funding_rate = float(row["funding_rate"])
        oracle_price = float(row["oracle_price"])
        hedge_pnl = 0.0 if step == 0 else -short_size_eth * (price - previous_price)
        funding_pnl = 0.0 if step == 0 else short_size_eth * oracle_price * funding_rate
        rebalance = step % rebalance_hours == 0
        target_short_size_eth = short_size_eth
        trading_fee = 0.0

        if rebalance:
            target_short_size_eth = monte_carlo_target_short_size_eth(
                current_price=price,
                position=position,
                horizon_hours=rebalance_hours,
                mc_paths=config.mc_paths,
                annual_vol=config.annual_vol,
                annual_drift=config.annual_drift,
                rng=rng,
            )
            delta_change_eth = target_short_size_eth - short_size_eth
            trading_fee = config.trading_fee_rate * abs(delta_change_eth) * price
            short_size_eth = target_short_size_eth
        else:
            delta_change_eth = 0.0

        lp_value = _lp_value_usd(price, position)
        lp_pnl = lp_value - initial_lp_value
        cumulative_hedge_pnl += hedge_pnl
        cumulative_funding_pnl += funding_pnl
        cumulative_trading_fees += trading_fee
        cumulative_net_hedge_pnl = (
            cumulative_hedge_pnl + cumulative_funding_pnl - cumulative_trading_fees
        )

        # New derived columns
        hodl_value_usd = initial_weth * price + initial_usdc
        gross_il_usd = hodl_value_usd - lp_value
        ts_key = row["timestamp"]
        lp_fee_income_usd = float(fee_lookup.get(ts_key, 0.0))
        residual_il_usd = gross_il_usd - cumulative_net_hedge_pnl
        net_position_pnl_usd = lp_fee_income_usd - residual_il_usd

        rows.append(
            {
                "timestamp": row["timestamp"],
                "position_id": position_id,
                "rebalance_hours": rebalance_hours,
                "strategy": "monte_carlo_expected_delta",
                "price_usdc_per_weth": price,
                "oracle_price": oracle_price,
                "funding_rate": funding_rate,
                "lp_value_usd": lp_value,
                "lp_pnl_usd": lp_pnl,
                "short_size_eth": short_size_eth,
                "target_short_size_eth": target_short_size_eth,
                "delta_change_eth": delta_change_eth,
                "rebalanced": rebalance,
                "hedge_pnl_usd": hedge_pnl,
                "funding_pnl_usd": funding_pnl,
                "trading_fee_usd": trading_fee,
                "cumulative_hedge_pnl_usd": cumulative_hedge_pnl,
                "cumulative_funding_pnl_usd": cumulative_funding_pnl,
                "cumulative_trading_fees_usd": cumulative_trading_fees,
                "cumulative_net_hedge_pnl_usd": cumulative_net_hedge_pnl,
                "net_lp_plus_hedge_pnl_usd": lp_pnl + cumulative_net_hedge_pnl,
                "mc_paths": config.mc_paths,
                "annual_vol": config.annual_vol,
                "annual_drift": config.annual_drift,
                "hodl_value_usd": hodl_value_usd,
                "gross_il_usd": gross_il_usd,
                "lp_fee_income_usd": lp_fee_income_usd,
                "residual_il_usd": residual_il_usd,
                "net_position_pnl_usd": net_position_pnl_usd,
            }
        )
        previous_price = price

    return rows


def _lp_amounts_usdc_weth(
    price_usdc_per_weth: float,
    position: pd.Series | dict,
) -> tuple[float, float, float]:
    liquidity = float(position["liquidity_raw"])
    sqrt_price = float(np.sqrt(RAW_PRICE_SCALE / price_usdc_per_weth))
    sqrt_lower = sqrt_raw_price_at_tick(int(position["tick_lower"]))
    sqrt_upper = sqrt_raw_price_at_tick(int(position["tick_upper"]))

    if sqrt_price <= sqrt_lower:
        amount0_raw = liquidity * (sqrt_upper - sqrt_lower) / (sqrt_lower * sqrt_upper)
        amount1_raw = 0.0
    elif sqrt_price >= sqrt_upper:
        amount0_raw = 0.0
        amount1_raw = liquidity * (sqrt_upper - sqrt_lower)
    else:
        amount0_raw = liquidity * (sqrt_upper - sqrt_price) / (sqrt_price * sqrt_upper)
        amount1_raw = liquidity * (sqrt_price - sqrt_lower)

    usdc = amount0_raw / USDC_RAW_SCALE
    weth = amount1_raw / WETH_RAW_SCALE
    return usdc, weth, usdc + weth * price_usdc_per_weth


def _lp_value_usd(price_usdc_per_weth: float, position: pd.Series | dict) -> float:
    return _lp_amounts_usdc_weth(price_usdc_per_weth, position)[2]


def _build_position_for_notional(
    *,
    position_id: str,
    width_pct: float,
    price_usdc_per_weth: float,
    notional_usd: float,
    template: dict[str, object],
) -> dict[str, object]:
    if price_usdc_per_weth <= 0:
        raise ValueError("price_usdc_per_weth must be positive")
    if width_pct <= 0:
        raise ValueError("width_pct must be positive")
    if notional_usd < 0:
        raise ValueError("notional_usd must be non-negative")

    current_tick = _tick_from_human_price(price_usdc_per_weth)
    price_lower = price_usdc_per_weth * (1.0 - width_pct)
    price_upper = price_usdc_per_weth * (1.0 + width_pct)
    tick_lower = _floor_to_tick_spacing(_tick_from_human_price(price_upper))
    tick_upper = _ceil_to_tick_spacing(_tick_from_human_price(price_lower))
    if tick_lower >= current_tick:
        tick_lower = _floor_to_tick_spacing(current_tick - TICK_SPACING)
    if tick_upper <= current_tick:
        tick_upper = _ceil_to_tick_spacing(current_tick + TICK_SPACING)

    tick_lower = max(_ceil_to_tick_spacing(MIN_TICK), tick_lower)
    tick_upper = min(_floor_to_tick_spacing(MAX_TICK), tick_upper)
    if tick_lower >= tick_upper:
        raise ValueError(f"invalid range for {position_id}: [{tick_lower}, {tick_upper}]")

    unit_position = {
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "liquidity_raw": 1.0,
    }
    unit_value = _lp_value_usd(price_usdc_per_weth, unit_position)
    liquidity = 0.0 if unit_value <= 0 or notional_usd == 0 else notional_usd / unit_value
    rebuilt = {
        **template,
        "position_id": position_id,
        "width_pct": width_pct,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "price_lower_usdc_per_weth": min(
            _human_price_at_tick(tick_lower),
            _human_price_at_tick(tick_upper),
        ),
        "price_upper_usdc_per_weth": max(
            _human_price_at_tick(tick_lower),
            _human_price_at_tick(tick_upper),
        ),
        "range_center_price_usdc_per_weth": price_usdc_per_weth,
        "entry_price_usdc_per_weth": price_usdc_per_weth,
        "entry_tick": current_tick,
        "liquidity_raw": liquidity,
    }
    initial_usdc, initial_weth, initial_value = _lp_amounts_usdc_weth(
        price_usdc_per_weth,
        rebuilt,
    )
    rebuilt["initial_usdc"] = initial_usdc
    rebuilt["initial_weth"] = initial_weth
    rebuilt["initial_value_usd"] = initial_value
    return rebuilt


def _position_active_at_price(position: pd.Series | dict, price_usdc_per_weth: float) -> bool:
    tick = _tick_from_human_price(price_usdc_per_weth)
    return int(position["tick_lower"]) <= tick < int(position["tick_upper"])


def _token_rebalance_notional_usd(
    *,
    old_usdc: float,
    old_weth: float,
    target_usdc: float,
    target_weth: float,
    price_usdc_per_weth: float,
) -> float:
    usdc_delta = abs(target_usdc - old_usdc)
    weth_delta_usd = abs(target_weth - old_weth) * price_usdc_per_weth
    return 0.5 * (usdc_delta + weth_delta_usd)


def _tick_from_human_price(price_usdc_per_weth: float) -> int:
    if price_usdc_per_weth <= 0:
        raise ValueError("price_usdc_per_weth must be positive")
    return int(floor(log(RAW_PRICE_SCALE / price_usdc_per_weth) / log(1.0001)))


def _human_price_at_tick(tick: int) -> float:
    return RAW_PRICE_SCALE / (1.0001**tick)


def _floor_to_tick_spacing(tick: int) -> int:
    return floor(tick / TICK_SPACING) * TICK_SPACING


def _ceil_to_tick_spacing(tick: int) -> int:
    return ceil(tick / TICK_SPACING) * TICK_SPACING


def _prepare_market_data(prices: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    price_time = _first_existing(prices, ["open_time", "timestamp", "time"])
    price_col = _first_existing(prices, ["close", "close_price", "mark_price", "price_usdc_per_weth"])
    funding_time = _first_existing(funding, ["funding_time", "timestamp", "time"])
    oracle_col = _first_existing(funding, ["oracle_price", "mark_price", "mid_price"])
    _require_columns(funding, {"funding_rate"}, "funding")
    price_work = prices[[price_time, price_col]].copy()
    price_work.columns = ["open_time", "close"]
    funding_work = funding[[funding_time, "funding_rate", oracle_col]].copy()
    funding_work.columns = ["funding_time", "funding_rate", "oracle_price"]
    price_work["timestamp"] = pd.to_datetime(price_work["open_time"], utc=True).dt.floor("h")
    funding_work["timestamp"] = pd.to_datetime(funding_work["funding_time"], utc=True).dt.floor("h")
    price_work["close"] = pd.to_numeric(price_work["close"], errors="raise").astype(float)
    funding_work["funding_rate"] = pd.to_numeric(
        funding_work["funding_rate"], errors="raise"
    ).astype(float)
    funding_work["oracle_price"] = pd.to_numeric(
        funding_work["oracle_price"], errors="raise"
    ).astype(float)
    market = price_work.merge(
        funding_work.drop(columns=["funding_time"]),
        on="timestamp",
        how="inner",
        validate="one_to_one",
    )
    if len(market) != len(price_work) or len(market) != len(funding_work):
        raise ValueError("prices and funding must have matching hourly timestamps")
    market = market.sort_values("timestamp").reset_index(drop=True)
    if market["close"].isna().any() or market["oracle_price"].isna().any():
        raise ValueError("prices and funding cannot contain missing close/oracle_price values")
    if market[["close", "oracle_price"]].le(0).any().any():
        raise ValueError("prices and oracle_price must be positive")
    diffs = market["timestamp"].diff().dropna()
    if not diffs.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("market inputs must be contiguous hourly rows")
    return market[["timestamp", "close", "funding_rate", "oracle_price"]]


def _lp_gamma(price_usdc_per_weth: float, position: pd.Series | dict) -> float:
    sqrt_price = float(np.sqrt(RAW_PRICE_SCALE / price_usdc_per_weth))
    sqrt_lower = sqrt_raw_price_at_tick(int(position["tick_lower"]))
    sqrt_upper = sqrt_raw_price_at_tick(int(position["tick_upper"]))
    if not (sqrt_lower < sqrt_price < sqrt_upper):
        return 0.0
    return -float(position["liquidity_raw"]) * sqrt_price / (
        2.0 * price_usdc_per_weth * WETH_RAW_SCALE
    )


def _validate_config(config: MonteCarloHedgeConfig) -> None:
    if not config.rebalance_hours:
        raise ValueError("rebalance_hours cannot be empty")
    if any(hours <= 0 for hours in config.rebalance_hours):
        raise ValueError("rebalance_hours must be positive")
    if config.trading_fee_rate < 0:
        raise ValueError("trading_fee_rate must be non-negative")
    if config.mc_paths <= 0:
        raise ValueError("mc_paths must be positive")
    if config.annual_vol < 0:
        raise ValueError("annual_vol must be non-negative")


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise ValueError(f"none of the expected columns exist: {candidates}")


def _parse_rebalance_hours(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise typer.BadParameter("rebalance hours must be comma-separated integers") from exc
    if not parsed:
        raise typer.BadParameter("at least one rebalance interval is required")
    if any(hours <= 0 for hours in parsed):
        raise typer.BadParameter("rebalance intervals must be positive")
    return parsed


def _parse_hedge_modes(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise typer.BadParameter("at least one hedge mode is required")
    invalid = sorted(set(parsed) - {"none", "hodl_tracking"})
    if invalid:
        raise typer.BadParameter(f"unknown hedge modes: {invalid}")
    return parsed


@app.command("run-monte-carlo")
def cmd_run_monte_carlo(
    positions_path: Path = typer.Option(DEFAULT_POSITIONS_PATH, "--positions-path"),
    prices_path: Path = typer.Option(DEFAULT_PRICES_PATH, "--prices-path"),
    funding_path: Path = typer.Option(DEFAULT_FUNDING_PATH, "--funding-path"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    rebalance_hours: str = typer.Option("1,4,24", "--rebalance-hours"),
    mc_paths: int = typer.Option(DEFAULT_MC_PATHS, "--mc-paths"),
    annual_vol: float = typer.Option(DEFAULT_ANNUAL_VOL, "--annual-vol"),
    annual_drift: float = typer.Option(DEFAULT_ANNUAL_DRIFT, "--annual-drift"),
    trading_fee_rate: float = typer.Option(DEFAULT_TRADING_FEE_RATE, "--trading-fee-rate"),
    seed: int = typer.Option(5_003, "--seed"),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
) -> None:
    """Run the Monte Carlo expected-delta hedge backtest."""

    config = MonteCarloHedgeConfig(
        rebalance_hours=_parse_rebalance_hours(rebalance_hours),
        trading_fee_rate=trading_fee_rate,
        mc_paths=mc_paths,
        annual_vol=annual_vol,
        annual_drift=annual_drift,
        seed=seed,
    )
    positions = pd.read_parquet(positions_path)
    prices = pd.read_parquet(prices_path)
    funding = pd.read_parquet(funding_path)
    results = run_monte_carlo_hedge_backtest(prices, funding, positions, config=config)
    results_path = write_hedge_results(results, results_dir=results_dir)
    typer.echo(f"wrote {results_path} ({len(results)} rows)")
    if plot:
        figure_path = plot_monte_carlo_hedge_results(
            results,
            figures_dir=results_dir / "figures",
        )
        typer.echo(f"wrote {figure_path}")


@app.command("run-active-recenter")
def cmd_run_active_recenter(
    positions_path: Path = typer.Option(DEFAULT_POSITIONS_PATH, "--positions-path"),
    prices_path: Path = typer.Option(DEFAULT_PRICES_PATH, "--prices-path"),
    funding_path: Path = typer.Option(DEFAULT_FUNDING_PATH, "--funding-path"),
    swaps_path: Path = typer.Option(Path("data/processed/swap_events.parquet"), "--swaps-path"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    hedge_modes: str = typer.Option("none,hodl_tracking", "--hedge-modes"),
    active_only: bool = typer.Option(False, "--active-only/--include-passive"),
    trading_fee_rate: float = typer.Option(DEFAULT_TRADING_FEE_RATE, "--trading-fee-rate"),
    lp_swap_fee_rate: float = typer.Option(DEFAULT_ACTIVE_LP_SWAP_FEE_RATE, "--lp-swap-fee-rate"),
    gas_cost_usd: float = typer.Option(DEFAULT_REBALANCE_GAS_USD, "--gas-cost-usd"),
    slippage_rate: float = typer.Option(DEFAULT_REBALANCE_SLIPPAGE_RATE, "--slippage-rate"),
    adjusted_fee_denominator: bool = typer.Option(
        True,
        "--adjusted-fee-denominator/--brief-fee-denominator",
    ),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
) -> None:
    """Run the active exit-recentered LP backtest and passive comparators."""

    positions = pd.read_parquet(positions_path)
    prices = pd.read_parquet(prices_path)
    funding = pd.read_parquet(funding_path)
    swaps = pd.read_parquet(swaps_path) if swaps_path.exists() else None
    if swaps is None:
        typer.echo(f"swap events not found at {swaps_path}, lp fee income will be zero")

    results = run_active_recenter_backtest(
        positions,
        prices,
        funding,
        swaps=swaps,
        include_passive=not active_only,
        include_active=True,
        hedge_modes=_parse_hedge_modes(hedge_modes),
        trading_fee_rate=trading_fee_rate,
        lp_swap_fee_rate=lp_swap_fee_rate,
        gas_cost_usd=gas_cost_usd,
        slippage_rate=slippage_rate,
        adjust_fee_denominator=adjusted_fee_denominator,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / DEFAULT_ACTIVE_RESULTS_PATH.name
    results.to_parquet(results_path, index=False)
    typer.echo(f"wrote {results_path} ({len(results)} rows)")
    if plot:
        figure_path = plot_active_recenter_results(results, figures_dir=results_dir / "figures")
        typer.echo(f"wrote {figure_path}")


@app.command("run-threshold-recenter")
def cmd_run_threshold_recenter(
    positions_path: Path = typer.Option(DEFAULT_POSITIONS_PATH, "--positions-path"),
    prices_path: Path = typer.Option(DEFAULT_PRICES_PATH, "--prices-path"),
    funding_path: Path = typer.Option(DEFAULT_FUNDING_PATH, "--funding-path"),
    swaps_path: Path = typer.Option(Path("data/processed/swap_events.parquet"), "--swaps-path"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    position_ids: str = typer.Option("P4", "--position-ids"),
    hedge_modes: str = typer.Option("none,hodl_tracking", "--hedge-modes"),
    threshold_fraction: float = typer.Option(DEFAULT_THRESHOLD_FRACTION, "--threshold-fraction"),
    cooldown_hours: int = typer.Option(DEFAULT_REBALANCE_COOLDOWN_HOURS, "--cooldown-hours"),
    expected_fee_window_hours: int = typer.Option(
        DEFAULT_EXPECTED_FEE_WINDOW_HOURS,
        "--expected-fee-window-hours",
    ),
    hedge_no_trade_band_eth: float = typer.Option(
        DEFAULT_HEDGE_NO_TRADE_BAND_ETH,
        "--hedge-no-trade-band-eth",
    ),
    trading_fee_rate: float = typer.Option(DEFAULT_TRADING_FEE_RATE, "--trading-fee-rate"),
    lp_swap_fee_rate: float = typer.Option(DEFAULT_ACTIVE_LP_SWAP_FEE_RATE, "--lp-swap-fee-rate"),
    gas_cost_usd: float = typer.Option(DEFAULT_REBALANCE_GAS_USD, "--gas-cost-usd"),
    slippage_rate: float = typer.Option(DEFAULT_REBALANCE_SLIPPAGE_RATE, "--slippage-rate"),
    adjusted_fee_denominator: bool = typer.Option(
        True,
        "--adjusted-fee-denominator/--brief-fee-denominator",
    ),
    plot: bool = typer.Option(True, "--plot/--no-plot"),
) -> None:
    """Run the cost-aware threshold recentering strategy."""

    positions = pd.read_parquet(positions_path)
    prices = pd.read_parquet(prices_path)
    funding = pd.read_parquet(funding_path)
    swaps = pd.read_parquet(swaps_path) if swaps_path.exists() else None
    if swaps is None:
        typer.echo(f"swap events not found at {swaps_path}, lp fee income will be zero")

    include_ids = tuple(part.strip() for part in position_ids.split(",") if part.strip())
    results = run_active_recenter_backtest(
        positions,
        prices,
        funding,
        swaps=swaps,
        include_position_ids=include_ids or None,
        include_passive=True,
        include_active=False,
        include_threshold=True,
        hedge_modes=_parse_hedge_modes(hedge_modes),
        trading_fee_rate=trading_fee_rate,
        lp_swap_fee_rate=lp_swap_fee_rate,
        gas_cost_usd=gas_cost_usd,
        slippage_rate=slippage_rate,
        adjust_fee_denominator=adjusted_fee_denominator,
        threshold_fraction=threshold_fraction,
        cooldown_hours=cooldown_hours,
        expected_fee_window_hours=expected_fee_window_hours,
        hedge_no_trade_band_eth=hedge_no_trade_band_eth,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / DEFAULT_THRESHOLD_RESULTS_PATH.name
    results.to_parquet(results_path, index=False)
    typer.echo(f"wrote {results_path} ({len(results)} rows)")
    if plot:
        figure_path = plot_active_recenter_results(
            results,
            figures_dir=results_dir / "figures",
            output_name="module5_threshold_recenter_results.png",
        )
        typer.echo(f"wrote {figure_path}")


@app.command("run")
def cmd_run(
    positions_path: Path = typer.Option(DEFAULT_POSITIONS_PATH, "--positions-path"),
    prices_path: Path = typer.Option(DEFAULT_PRICES_PATH, "--prices-path"),
    funding_path: Path = typer.Option(DEFAULT_FUNDING_PATH, "--funding-path"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    rebalance_hours: str = typer.Option("1,4,24", "--rebalance-hours"),
    include_no_trade_bands: bool = typer.Option(True, "--include-no-trade-bands/--fixed-only"),
    trading_fee_rate: float = typer.Option(DEFAULT_TRADING_FEE_RATE, "--trading-fee-rate"),
    fee_timeseries_path: Path = typer.Option(
        DEFAULT_FEE_TIMESERIES_PATH, "--fee-timeseries-path"
    ),
) -> None:
    """Run the analytical delta-hedging backtest."""

    positions = pd.read_parquet(positions_path)
    prices = pd.read_parquet(prices_path)
    funding = pd.read_parquet(funding_path)
    fee_timeseries: pd.DataFrame | None = None
    if fee_timeseries_path.exists():
        fee_timeseries = pd.read_parquet(fee_timeseries_path)
        typer.echo(f"loaded fee timeseries from {fee_timeseries_path} ({len(fee_timeseries)} rows)")
    else:
        typer.echo(f"fee timeseries not found at {fee_timeseries_path}, lp_fee_income_usd=0")
    bands = DEFAULT_NO_TRADE_BANDS if include_no_trade_bands else ()
    results = run_hedge_backtest(
        positions,
        prices,
        funding,
        rebalance_hours=_parse_rebalance_hours(rebalance_hours),
        no_trade_bands=bands,
        trading_fee_rate=trading_fee_rate,
        fee_timeseries=fee_timeseries,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "hedge_results.parquet"
    results.to_parquet(results_path, index=False)
    figure_path = plot_hedge_results(results, figures_dir=results_dir / "figures")
    research = compute_optional_research_extensions(
        results,
        _prepare_market_data(prices, funding),
    )
    research_path = results_dir / "optional_research_extensions.json"
    research_path.write_text(json.dumps(research, indent=2, default=str), encoding="utf-8")
    typer.echo(f"wrote {results_path} ({len(results)} rows)")
    typer.echo(f"wrote {figure_path}")
    typer.echo(f"wrote {research_path}")


if __name__ == "__main__":
    app()
