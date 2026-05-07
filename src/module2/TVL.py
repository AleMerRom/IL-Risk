from __future__ import annotations

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RAW_PRICE_SCALE = 10**12
USDC_RAW_SCALE = 10**6
WETH_RAW_SCALE = 10**18
TICK_SPACING = 10

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_RESULTS_DIR = Path("data/results/module_2")
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"


def run_tvl_analysis(
    processed_dir: Path | str = DEFAULT_PROCESSED_DIR,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    mint_burn = pd.read_parquet(
        processed_dir / "mint_burn_events.parquet",
        columns=["block_number", "tick_lower", "tick_upper", "liquidity_delta"],
    )
    slot0 = pd.read_parquet(processed_dir / "slot0_snapshots.parquet")

    tvl = compute_tvl_decomposition(mint_burn, slot0)
    tvl.to_parquet(results_dir / "tvl_decomposition.parquet", index=False)
    plot_tvl_decomposition(tvl, figures_dir / "fig_2_2_tvl_decomposition.png")
    return tvl


def compute_tvl_decomposition(mint_burn: pd.DataFrame, slot0: pd.DataFrame) -> pd.DataFrame:
    mint_burn = mint_burn.sort_values(["block_number"]).reset_index(drop=True)
    slot0 = slot0.sort_values("snapshot_block").reset_index(drop=True)

    active_ranges: dict[tuple[int, int], int] = {}
    rows = []
    event_index = 0

    for slot in slot0.to_dict("records"):
        snapshot_block = int(slot["snapshot_block"])
        while event_index < len(mint_burn) and int(mint_burn.at[event_index, "block_number"]) <= snapshot_block:
            event = mint_burn.iloc[event_index]
            key = (int(event["tick_lower"]), int(event["tick_upper"]))
            active_ranges[key] = active_ranges.get(key, 0) + int(event["liquidity_delta"])
            if active_ranges[key] == 0:
                del active_ranges[key]
            event_index += 1

        current_price = float(slot["price_usdc_per_weth"])
        current_sqrt_raw = _sqrt_raw_price_from_human_price(current_price)

        in_range = 0.0
        above = 0.0
        below = 0.0
        in_range_usdc = 0.0
        above_usdc = 0.0
        below_usdc = 0.0
        in_range_weth = 0.0
        above_weth = 0.0
        below_weth = 0.0

        for (tick_lower, tick_upper), liquidity_raw_int in active_ranges.items():
            liquidity_raw = float(liquidity_raw_int)
            if liquidity_raw <= 0:
                continue

            sqrt_lower = _sqrt_raw_price_at_tick(tick_lower)
            sqrt_upper = _sqrt_raw_price_at_tick(tick_upper)

            amount0_raw, amount1_raw = _amounts_raw_for_liquidity(
                liquidity_raw,
                current_sqrt_raw,
                sqrt_lower,
                sqrt_upper,
            )
            amount0_usdc = amount0_raw / USDC_RAW_SCALE
            amount1_weth = amount1_raw / WETH_RAW_SCALE
            value_usd = amount0_usdc + amount1_weth * current_price

            price_low = min(_human_price_at_tick(tick_lower), _human_price_at_tick(tick_upper))
            price_high = max(_human_price_at_tick(tick_lower), _human_price_at_tick(tick_upper))
            if price_low <= current_price <= price_high:
                in_range += value_usd
                in_range_usdc += amount0_usdc
                in_range_weth += amount1_weth
            elif price_low > current_price:
                above += value_usd
                above_usdc += amount0_usdc
                above_weth += amount1_weth
            else:
                below += value_usd
                below_usdc += amount0_usdc
                below_weth += amount1_weth

        rows.append(
            {
                "date": slot["date"],
                "snapshot_block": snapshot_block,
                "snapshot_timestamp": slot["snapshot_timestamp"],
                "current_price_usdc_per_weth": current_price,
                "in_range_tvl_usd": in_range,
                "above_range_tvl_usd": above,
                "below_range_tvl_usd": below,
                "total_tvl_usd": in_range + above + below,
                "in_range_usdc": in_range_usdc,
                "above_range_usdc": above_usdc,
                "below_range_usdc": below_usdc,
                "in_range_weth": in_range_weth,
                "above_range_weth": above_weth,
                "below_range_weth": below_weth,
                "active_range_count": sum(1 for liquidity in active_ranges.values() if liquidity > 0),
            }
        )

    return pd.DataFrame(rows)


def plot_tvl_decomposition(tvl: pd.DataFrame, output_path: Path) -> None:
    dates = pd.to_datetime(tvl["date"])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        dates,
        tvl["in_range_tvl_usd"],
        tvl["above_range_tvl_usd"],
        tvl["below_range_tvl_usd"],
        labels=["In range", "Out of range above", "Out of range below"],
        alpha=0.85,
    )
    ax.set_title("TVL Decomposition Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("TVL (USD)")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


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


def _sqrt_raw_price_at_tick(tick: int) -> float:
    return sqrt(1.0001**tick)


def _human_price_at_tick(tick: int) -> float:
    return RAW_PRICE_SCALE / (1.0001**tick)


def _sqrt_raw_price_from_human_price(price_usdc_per_weth: float) -> float:
    if price_usdc_per_weth <= 0:
        raise ValueError("price must be positive")
    return sqrt(RAW_PRICE_SCALE / price_usdc_per_weth)
