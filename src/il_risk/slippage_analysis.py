"""Module 3 — slippage simulation grid and price-impact curve analysis."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from math import sqrt
from pathlib import Path
from typing import Iterable

import pandas as pd

from il_risk.constants import FEE_TIER
from il_risk.swap_simulator import Direction, simulate_swap

DEFAULT_TRADE_SIZES_USD = (1_000, 10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000)
DEFAULT_DIRECTIONS: tuple[Direction, Direction] = ("buy_weth", "sell_weth")


def run_simulation_grid(
    *,
    data_dir: Path | str = Path("data"),
    liquidity_path: Path | str | None = None,
    slot0_path: Path | str | None = None,
    output_path: Path | str | None = None,
    trade_sizes_usd: Iterable[float | int | Decimal] = DEFAULT_TRADE_SIZES_USD,
    directions: Iterable[Direction] = DEFAULT_DIRECTIONS,
    fee_bps: float | Decimal = Decimal(FEE_TIER) / Decimal(100),
    write_output: bool = True,
) -> pd.DataFrame:
    """Run Task 3.2's size x snapshot x direction grid.

    By default this reads Module 1's processed parquet files and writes
    ``data/processed/simulated_trades.parquet``.  Passing explicit paths makes
    the function easy to test with synthetic parquet files while real Module 1
    data is still being collected.
    """

    base = Path(data_dir)
    liquidity_path = Path(liquidity_path) if liquidity_path else base / "processed" / "liquidity_snapshots.parquet"
    slot0_path = Path(slot0_path) if slot0_path else base / "processed" / "slot0_snapshots.parquet"
    output_path = Path(output_path) if output_path else base / "processed" / "simulated_trades.parquet"

    liquidity_df = pd.read_parquet(liquidity_path)
    slot0_df = pd.read_parquet(slot0_path).sort_values("snapshot_block")
    _validate_inputs(liquidity_df, slot0_df)

    rows: list[dict] = []
    liquidity_by_block = {
        int(block): group.sort_values("tick").to_dict("records")
        for block, group in liquidity_df.groupby("snapshot_block", sort=True)
    }

    sizes = [Decimal(str(size)) for size in trade_sizes_usd]
    dirs = list(directions)
    for slot0_row in slot0_df.to_dict("records"):
        block = int(slot0_row["snapshot_block"])
        liquidity_rows = liquidity_by_block.get(block)
        if not liquidity_rows:
            raise ValueError(f"missing liquidity snapshot rows for block {block}")

        for direction in dirs:
            for notional_usd in sizes:
                result = simulate_swap(
                    liquidity_rows,
                    slot0_row,
                    direction,
                    notional_usd,
                    fee_bps=fee_bps,
                )
                rows.append(_result_row(result))

    output = pd.DataFrame(rows)
    output = output[
        [
            "snapshot_block",
            "snapshot_timestamp",
            "date",
            "direction",
            "notional_usd",
            "average_execution_price",
            "pool_mid_price",
            "price_impact_bps",
            "slippage_bps",
            "tick_boundaries_crossed",
            "input_amount_usdc",
            "input_amount_weth",
            "output_amount_usdc",
            "output_amount_weth",
            "final_sqrt_price_x96",
            "final_tick",
        ]
    ]

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(output_path, index=False)

    return output


def summarize_price_impact(
    simulated_trades: pd.DataFrame | None = None,
    *,
    data_dir: Path | str = Path("data"),
    simulated_trades_path: Path | str | None = None,
    output_path: Path | str | None = None,
    write_output: bool = True,
) -> pd.DataFrame:
    """Summarize Task 3.3 median and 10th/90th percentile impact curves."""

    base = Path(data_dir)
    simulated_trades_path = (
        Path(simulated_trades_path)
        if simulated_trades_path
        else base / "processed" / "simulated_trades.parquet"
    )
    output_path = (
        Path(output_path)
        if output_path
        else base / "processed" / "price_impact_summary.parquet"
    )

    trades = simulated_trades.copy() if simulated_trades is not None else pd.read_parquet(simulated_trades_path)
    _validate_simulated_trades(trades)

    summary = (
        trades.groupby(["direction", "notional_usd"], as_index=False)
        .agg(
            median_price_impact_bps=("price_impact_bps", "median"),
            p10_price_impact_bps=("price_impact_bps", lambda s: s.quantile(0.10)),
            p90_price_impact_bps=("price_impact_bps", lambda s: s.quantile(0.90)),
            observations=("price_impact_bps", "size"),
        )
        .sort_values(["direction", "notional_usd"])
        .reset_index(drop=True)
    )

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(output_path, index=False)

    return summary


def plot_price_impact_curves(
    summary: pd.DataFrame | None = None,
    *,
    data_dir: Path | str = Path("data"),
    summary_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Plot Task 3.3 log-log median impact curves with 10th/90th bands."""

    import matplotlib.pyplot as plt

    base = Path(data_dir)
    summary_path = (
        Path(summary_path)
        if summary_path
        else base / "processed" / "price_impact_summary.parquet"
    )
    output_path = (
        Path(output_path)
        if output_path
        else Path("figs") / "module3_price_impact_curves.png"
    )

    curve = summary.copy() if summary is not None else pd.read_parquet(summary_path)
    _validate_price_impact_summary(curve)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"buy_weth": "#1f77b4", "sell_weth": "#d62728"}
    labels = {"buy_weth": "Buy WETH", "sell_weth": "Sell WETH"}
    for direction, group in curve.groupby("direction", sort=True):
        group = group.sort_values("notional_usd")
        color = colors.get(direction, None)
        label = labels.get(direction, direction)
        x = group["notional_usd"].astype(float)
        median = group["median_price_impact_bps"].astype(float).clip(lower=1e-12)
        p10 = group["p10_price_impact_bps"].astype(float).clip(lower=1e-12)
        p90 = group["p90_price_impact_bps"].astype(float).clip(lower=1e-12)
        ax.plot(x, median, marker="o", linewidth=2, color=color, label=f"{label} median")
        ax.fill_between(x, p10, p90, color=color, alpha=0.18, label=f"{label} 10-90%")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trade size (USD notional, log scale)")
    ax.set_ylabel("Price impact (basis points, log scale)")
    ax.set_title("Uniswap V3 simulated price impact curves")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def compute_effective_spreads(
    swaps: pd.DataFrame | None = None,
    pre_swap_mid_prices: pd.DataFrame | None = None,
    *,
    data_dir: Path | str = Path("data"),
    swaps_path: Path | str | None = None,
    pre_swap_mid_prices_path: Path | str | None = None,
    output_path: Path | str | None = None,
    trade_size_buckets_usd: Iterable[float | int | Decimal] = DEFAULT_TRADE_SIZES_USD,
    write_output: bool = True,
) -> pd.DataFrame:
    """Compute Task 3.4 effective spreads for observed swaps.

    ``pre_swap_mid_prices`` must provide the pool mid-price at ``block_number - 1``
    for each observed swap block.  Keeping that lookup separate makes this
    formula testable before the archive-RPC price extraction is available.
    """

    base = Path(data_dir)
    swaps_path = Path(swaps_path) if swaps_path else base / "processed" / "swap_events.parquet"
    pre_swap_mid_prices_path = (
        Path(pre_swap_mid_prices_path)
        if pre_swap_mid_prices_path
        else base / "processed" / "swap_mid_prices.parquet"
    )
    output_path = (
        Path(output_path)
        if output_path
        else base / "processed" / "effective_spreads.parquet"
    )

    swaps_df = swaps.copy() if swaps is not None else pd.read_parquet(swaps_path)
    mids_df = (
        pre_swap_mid_prices.copy()
        if pre_swap_mid_prices is not None
        else pd.read_parquet(pre_swap_mid_prices_path)
    )
    _validate_swaps_for_effective_spread(swaps_df)
    mids_df = _normalize_pre_swap_mid_prices(mids_df)

    merged = swaps_df.merge(mids_df, on="block_number", how="left", validate="many_to_one")
    missing_mid = merged["mid_price_usdc_per_weth"].isna()
    if missing_mid.any():
        missing_blocks = sorted(merged.loc[missing_mid, "block_number"].unique().tolist())
        raise ValueError(f"missing pre-swap mid prices for blocks: {missing_blocks[:10]}")

    out = merged.copy()
    out["execution_price_usdc_per_weth"] = (
        out["amount0_usdc"].abs() / out["amount1_weth"].abs()
    )
    direction_sign = out["trade_direction"].map({"buy_weth": 1, "sell_weth": -1})
    out["direction_sign"] = direction_sign
    out["effective_spread"] = (
        2
        * out["direction_sign"]
        * (out["execution_price_usdc_per_weth"] - out["mid_price_usdc_per_weth"])
        / out["mid_price_usdc_per_weth"]
    )
    out["effective_spread_bps"] = out["effective_spread"] * 10_000
    out["size_bucket_usd"] = assign_trade_size_bucket(
        out["usd_notional"],
        trade_size_buckets_usd,
    )

    columns = [
        "block_number",
        "block_timestamp",
        "tx_hash",
        "log_index",
        "trade_direction",
        "direction_sign",
        "usd_notional",
        "size_bucket_usd",
        "execution_price_usdc_per_weth",
        "mid_price_usdc_per_weth",
        "effective_spread",
        "effective_spread_bps",
    ]
    for optional in ("date",):
        if optional in out.columns:
            columns.append(optional)
    out = out[columns].sort_values(["block_number", "log_index"]).reset_index(drop=True)

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(output_path, index=False)

    return out


def summarize_effective_spreads(
    effective_spreads: pd.DataFrame | None = None,
    *,
    data_dir: Path | str = Path("data"),
    effective_spreads_path: Path | str | None = None,
    output_path: Path | str | None = None,
    write_output: bool = True,
) -> pd.DataFrame:
    """Compute median observed effective spread by direction and size bucket."""

    base = Path(data_dir)
    effective_spreads_path = (
        Path(effective_spreads_path)
        if effective_spreads_path
        else base / "processed" / "effective_spreads.parquet"
    )
    output_path = (
        Path(output_path)
        if output_path
        else base / "processed" / "effective_spread_summary.parquet"
    )

    spreads = (
        effective_spreads.copy()
        if effective_spreads is not None
        else pd.read_parquet(effective_spreads_path)
    )
    _validate_effective_spreads(spreads)
    summary = (
        spreads.groupby(["trade_direction", "size_bucket_usd"], as_index=False)
        .agg(
            median_effective_spread_bps=("effective_spread_bps", "median"),
            p10_effective_spread_bps=("effective_spread_bps", lambda s: s.quantile(0.10)),
            p90_effective_spread_bps=("effective_spread_bps", lambda s: s.quantile(0.90)),
            observations=("effective_spread_bps", "size"),
        )
        .rename(columns={"trade_direction": "direction", "size_bucket_usd": "notional_usd"})
        .sort_values(["direction", "notional_usd"])
        .reset_index(drop=True)
    )

    if write_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(output_path, index=False)

    return summary


def plot_effective_spread_comparison(
    simulated_summary: pd.DataFrame | None = None,
    empirical_summary: pd.DataFrame | None = None,
    *,
    data_dir: Path | str = Path("data"),
    simulated_summary_path: Path | str | None = None,
    empirical_summary_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Plot Task 3.4 simulated impact vs observed effective spread by size."""

    import matplotlib.pyplot as plt

    base = Path(data_dir)
    simulated_summary_path = (
        Path(simulated_summary_path)
        if simulated_summary_path
        else base / "processed" / "price_impact_summary.parquet"
    )
    empirical_summary_path = (
        Path(empirical_summary_path)
        if empirical_summary_path
        else base / "processed" / "effective_spread_summary.parquet"
    )
    output_path = (
        Path(output_path)
        if output_path
        else Path("figs") / "module3_effective_spread_by_size.png"
    )

    simulated = (
        simulated_summary.copy()
        if simulated_summary is not None
        else pd.read_parquet(simulated_summary_path)
    )
    empirical = (
        empirical_summary.copy()
        if empirical_summary is not None
        else pd.read_parquet(empirical_summary_path)
    )
    _validate_price_impact_summary(simulated)
    _validate_effective_spread_summary(empirical)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"buy_weth": "#1f77b4", "sell_weth": "#d62728"}
    labels = {"buy_weth": "Buy WETH", "sell_weth": "Sell WETH"}
    for direction in sorted(set(simulated["direction"]) | set(empirical["direction"])):
        sim_group = simulated[simulated["direction"] == direction].sort_values("notional_usd")
        emp_group = empirical[empirical["direction"] == direction].sort_values("notional_usd")
        color = colors.get(direction, None)
        label = labels.get(direction, direction)
        if not sim_group.empty:
            ax.plot(
                sim_group["notional_usd"].astype(float),
                sim_group["median_price_impact_bps"].astype(float),
                marker="o",
                linewidth=2,
                color=color,
                label=f"{label} simulated",
            )
        if not emp_group.empty:
            ax.plot(
                emp_group["notional_usd"].astype(float),
                emp_group["median_effective_spread_bps"].astype(float),
                marker="s",
                linewidth=2,
                linestyle="--",
                color=color,
                label=f"{label} observed",
            )

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Trade size bucket (USD notional, log scale)")
    ax.set_ylabel("Basis points")
    ax.set_title("Simulated price impact vs observed effective spread")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def assign_trade_size_bucket(
    notionals: pd.Series,
    buckets_usd: Iterable[float | int | Decimal] = DEFAULT_TRADE_SIZES_USD,
) -> pd.Series:
    """Assign observed swap notionals to Task 3.2 trade-size buckets."""

    buckets = sorted(float(bucket) for bucket in buckets_usd)
    if not buckets:
        raise ValueError("at least one trade-size bucket is required")
    if len(buckets) == 1:
        return pd.Series([buckets[0]] * len(notionals), index=notionals.index, dtype="float64")

    boundaries = [0.0]
    boundaries.extend(sqrt(left * right) for left, right in zip(buckets, buckets[1:]))
    boundaries.append(float("inf"))
    return pd.cut(
        notionals.astype(float),
        bins=boundaries,
        labels=buckets,
        include_lowest=True,
        right=False,
    ).astype("float64")


def _result_row(result) -> dict:
    row = asdict(result)
    return {
        **row,
        "notional_usd": float(row["notional_usd"]),
        "tick_boundaries_crossed": int(row["tick_boundaries_crossed"]),
        "final_sqrt_price_x96": str(row["final_sqrt_price_x96"]),
    }


def _validate_inputs(liquidity_df: pd.DataFrame, slot0_df: pd.DataFrame) -> None:
    liquidity_required = {"snapshot_block", "tick", "liquidityNet", "active_liquidity"}
    slot0_required = {
        "snapshot_block",
        "snapshot_timestamp",
        "sqrt_price_x96",
        "price_usdc_per_weth",
        "current_tick",
    }
    missing_liquidity = liquidity_required - set(liquidity_df.columns)
    missing_slot0 = slot0_required - set(slot0_df.columns)
    if missing_liquidity:
        raise ValueError(f"liquidity snapshot file is missing columns: {sorted(missing_liquidity)}")
    if missing_slot0:
        raise ValueError(f"slot0 snapshot file is missing columns: {sorted(missing_slot0)}")
    duplicated_slot0 = slot0_df["snapshot_block"].duplicated()
    if duplicated_slot0.any():
        blocks = slot0_df.loc[duplicated_slot0, "snapshot_block"].tolist()
        raise ValueError(f"slot0 snapshot file has duplicate snapshot_block values: {blocks}")


def _validate_simulated_trades(trades: pd.DataFrame) -> None:
    required = {"direction", "notional_usd", "price_impact_bps"}
    missing = required - set(trades.columns)
    if missing:
        raise ValueError(f"simulated trades file is missing columns: {sorted(missing)}")
    if trades.empty:
        raise ValueError("simulated trades file is empty")


def _validate_price_impact_summary(summary: pd.DataFrame) -> None:
    required = {
        "direction",
        "notional_usd",
        "median_price_impact_bps",
        "p10_price_impact_bps",
        "p90_price_impact_bps",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"price impact summary is missing columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError("price impact summary is empty")


def _validate_swaps_for_effective_spread(swaps: pd.DataFrame) -> None:
    required = {
        "block_number",
        "block_timestamp",
        "tx_hash",
        "log_index",
        "amount0_usdc",
        "amount1_weth",
        "trade_direction",
        "usd_notional",
    }
    missing = required - set(swaps.columns)
    if missing:
        raise ValueError(f"swap events file is missing columns: {sorted(missing)}")
    if swaps.empty:
        raise ValueError("swap events file is empty")
    valid_directions = {"buy_weth", "sell_weth"}
    bad_directions = set(swaps["trade_direction"].dropna()) - valid_directions
    if bad_directions:
        raise ValueError(f"unknown trade_direction values: {sorted(bad_directions)}")
    if (swaps["amount1_weth"].astype(float).abs() == 0).any():
        raise ValueError("cannot compute execution price for swaps with zero WETH amount")


def _normalize_pre_swap_mid_prices(mid_prices: pd.DataFrame) -> pd.DataFrame:
    if "block_number" not in mid_prices.columns:
        raise ValueError("pre-swap mid price file is missing column: 'block_number'")
    price_candidates = ["mid_price_usdc_per_weth", "price_usdc_per_weth"]
    price_col = next((col for col in price_candidates if col in mid_prices.columns), None)
    if price_col is None:
        raise ValueError(
            "pre-swap mid price file must contain 'mid_price_usdc_per_weth' "
            "or 'price_usdc_per_weth'"
        )
    out = mid_prices[["block_number", price_col]].rename(
        columns={price_col: "mid_price_usdc_per_weth"}
    )
    if out["block_number"].duplicated().any():
        blocks = out.loc[out["block_number"].duplicated(), "block_number"].tolist()
        raise ValueError(f"pre-swap mid price file has duplicate block_number values: {blocks}")
    return out


def _validate_effective_spreads(spreads: pd.DataFrame) -> None:
    required = {"trade_direction", "size_bucket_usd", "effective_spread_bps"}
    missing = required - set(spreads.columns)
    if missing:
        raise ValueError(f"effective spreads file is missing columns: {sorted(missing)}")
    if spreads.empty:
        raise ValueError("effective spreads file is empty")


def _validate_effective_spread_summary(summary: pd.DataFrame) -> None:
    required = {"direction", "notional_usd", "median_effective_spread_bps"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"effective spread summary is missing columns: {sorted(missing)}")
    if summary.empty:
        raise ValueError("effective spread summary is empty")
