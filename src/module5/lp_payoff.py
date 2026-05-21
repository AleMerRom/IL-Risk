"""Module 5 Task 5.1.1 — LP payoff as a function of terminal ETH price (feeless)."""

from __future__ import annotations

import os
import tempfile
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

# Uniswap V3 scaling constants (same as module4)
RAW_PRICE_SCALE = 10**12   # sqrt_price = sqrt(RAW_PRICE_SCALE / price_usdc_per_weth)
USDC_RAW_SCALE = 10**6
WETH_RAW_SCALE = 10**18

N_PRICE_POINTS = 2000
DEFAULT_POSITIONS_PATH = Path("data/results/module_4/module4_lp_positions.parquet")
DEFAULT_FIGURES_DIR = Path("data/results/module_5/figures")

_CACHE_ROOT = Path(tempfile.gettempdir()) / "il-risk-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

app = typer.Typer(no_args_is_help=True, add_completion=False)

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _sqrt_raw_price_at_tick(tick: int) -> float:
    return sqrt(1.0001**tick)


def _lp_value_over_grid(
    prices: np.ndarray,
    liquidity: float,
    sqrt_lower: float,
    sqrt_upper: float,
    initial_usdc: float,
    initial_weth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised V_LP(p) and V_HODL(p) over a price grid.

    In Uniswap V3 raw sqrt-price space: sqrt_price = sqrt(RAW_PRICE_SCALE / p),
    so higher USDC/WETH price → smaller sqrt_price.  The three regions are:

        sqrt_prices <= sqrt_lower  →  above range (ETH expensive): LP holds all USDC
        sqrt_prices >= sqrt_upper  →  below range (ETH cheap):     LP holds all WETH
        otherwise                  →  in-range:                    concave mixed portfolio
    """
    sqrt_prices = np.sqrt(RAW_PRICE_SCALE / prices)

    above = sqrt_prices <= sqrt_lower   # high ETH price, LP sold WETH for USDC
    below = sqrt_prices >= sqrt_upper   # low ETH price,  LP bought WETH with USDC
    in_rng = ~above & ~below

    # Amount0 (USDC raw)
    a0_above = liquidity * (sqrt_upper - sqrt_lower) / (sqrt_lower * sqrt_upper)
    a0_in = liquidity * (sqrt_upper - sqrt_prices) / (sqrt_prices * sqrt_upper)
    a0 = np.where(above, a0_above, np.where(in_rng, a0_in, 0.0))

    # Amount1 (WETH raw)
    a1_below = liquidity * (sqrt_upper - sqrt_lower)
    a1_in = liquidity * (sqrt_prices - sqrt_lower)
    a1 = np.where(below, a1_below, np.where(in_rng, a1_in, 0.0))

    lp_value = a0 / USDC_RAW_SCALE + (a1 / WETH_RAW_SCALE) * prices
    hodl_value = initial_usdc + initial_weth * prices
    return lp_value, hodl_value


def plot_lp_payoff(
    positions: pd.DataFrame,
    *,
    output_path: Path,
    n_points: int = N_PRICE_POINTS,
) -> None:
    p0 = float(positions["entry_price_usdc_per_weth"].iloc[0])
    prices = np.linspace(0.5 * p0, 1.5 * p0, n_points)

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, row in enumerate(positions.sort_values("position_id").to_dict("records")):
        sqrt_lower = _sqrt_raw_price_at_tick(int(row["tick_lower"]))
        sqrt_upper = _sqrt_raw_price_at_tick(int(row["tick_upper"]))

        lp_val, hodl_val = _lp_value_over_grid(
            prices,
            float(row["liquidity_raw"]),
            sqrt_lower,
            sqrt_upper,
            float(row["initial_usdc"]),
            float(row["initial_weth"]),
        )

        color = _COLORS[i]
        ax.plot(
            prices, lp_val,
            color=color, linewidth=2.2, zorder=3,
            label=f"{row['position_id']} ({row['price_range_label']})",
        )
        ax.plot(
            prices, hodl_val,
            color=color, linewidth=0.9, linestyle="--", alpha=0.35, zorder=2,
        )

    # Dummy legend entry for HODL lines
    ax.plot([], [], color="gray", linewidth=0.9, linestyle="--", alpha=0.6,
            label="HODL benchmark (dashed, per position)")

    ax.axvline(p0, color="dimgray", linestyle=":", linewidth=1.5, zorder=4,
               label=f"Entry price $p_0$ = {p0:,.0f}")
    ax.axhline(100_000, color="dimgray", linestyle=":", linewidth=0.9, alpha=0.5, zorder=4)

    ax.set_xlabel("Terminal ETH price $p_T$ (USDC / WETH)", fontsize=12)
    ax.set_ylabel("Position value (USD)", fontsize=12)
    ax.set_title(
        "LP Position Payoff vs. Terminal ETH Price — Feeless (Task 5.1.1)\n"
        "Solid = $V_{LP}(p_T)$,  dashed = $V_{HODL}(p_T)$,  gap = impermanent loss",
        fontsize=11,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92)
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.set_xlim(prices[0], prices[-1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


@app.command()
def run(
    positions_path: Path = typer.Option(
        DEFAULT_POSITIONS_PATH, help="Path to module4_lp_positions.parquet"
    ),
    figures_dir: Path = typer.Option(
        DEFAULT_FIGURES_DIR, help="Output directory for figures"
    ),
) -> None:
    """Plot LP payoff as a function of terminal ETH price (Task 5.1.1)."""
    positions = pd.read_parquet(positions_path)
    plot_lp_payoff(positions, output_path=figures_dir / "module5_lp_payoff.png")


if __name__ == "__main__":
    app()
