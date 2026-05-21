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
import os
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

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


def _lp_value_usd(price_usdc_per_weth: float, position: pd.Series | dict) -> float:
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
    return amount0_raw / USDC_RAW_SCALE + (amount1_raw / WETH_RAW_SCALE) * price_usdc_per_weth


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
    research = compute_optional_research_extensions(
        results,
        _prepare_market_data(prices, funding),
    )
    research_path = results_dir / "optional_research_extensions.json"
    research_path.write_text(json.dumps(research, indent=2, default=str), encoding="utf-8")
    typer.echo(f"wrote {results_path} ({len(results)} rows)")
    typer.echo(f"wrote {research_path}")


if __name__ == "__main__":
    app()
