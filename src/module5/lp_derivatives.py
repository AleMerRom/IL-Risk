"""Module 5 Task 5.1.2 — analytic LP delta.

The pool is USDC/WETH, with human price ``p`` quoted as USDC per WETH.
Following the Module 4 code, the Uniswap raw square-root price is

    s(p) = sqrt(RAW_PRICE_SCALE / p),  RAW_PRICE_SCALE = 10**12.

For liquidity ``L`` over ``[sqrt_lower, sqrt_upper]``, the LP value is:

    above range, high ETH price:  V_LP = USDC-only value
    below range, low ETH price:   V_LP = p * WETH-only amount
    in range:                    V_LP = amount0(p) + p * amount1(p)

Taking the derivative with respect to the human ETH price ``p`` gives delta
in WETH units.
"""

from __future__ import annotations

import os
from math import sqrt
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

RAW_PRICE_SCALE = 10**12
WETH_RAW_SCALE = 10**18
N_PRICE_POINTS = 2000
DEFAULT_POSITIONS_PATH = Path("data/results/module_4/module4_lp_positions.parquet")
DEFAULT_FIGURES_DIR = Path("data/results/module_5/figures")

_CACHE_ROOT = Path(tempfile.gettempdir()) / "il-risk-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

app = typer.Typer(no_args_is_help=True, add_completion=False)
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def sqrt_raw_price_from_human_price(
    price_usdc_per_weth: float | np.ndarray | pd.Series,
) -> float | np.ndarray:
    """Return Uniswap raw sqrt price ``sqrt(10**12 / p)`` for human ETH price."""

    prices = _as_float_array(price_usdc_per_weth)
    if np.any(prices <= 0):
        raise ValueError("price_usdc_per_weth must be positive")
    sqrt_prices = np.sqrt(RAW_PRICE_SCALE / prices)
    return _maybe_scalar(sqrt_prices, price_usdc_per_weth)


def sqrt_raw_price_at_tick(tick: int) -> float:
    """Return raw sqrt price at a Uniswap tick, matching Module 4/5 float math."""

    return sqrt(1.0001**tick)


def lp_delta(
    price_usdc_per_weth: float | np.ndarray | pd.Series,
    liquidity_raw: float,
    sqrt_lower: float,
    sqrt_upper: float,
) -> float | np.ndarray:
    """Analytic ``dV_LP / dp`` for a Uniswap V3 LP position.

    Cases:
    - high ETH price / above range: all USDC, delta = 0
    - low ETH price / below range: all WETH, delta = L(su - sl) / 1e18
    - in range: delta = L(s(p) - sl) / 1e18

    The in-range expression equals the LP's current WETH inventory.
    """

    _validate_inputs(liquidity_raw, sqrt_lower, sqrt_upper)
    prices = _as_float_array(price_usdc_per_weth)
    if np.any(prices <= 0):
        raise ValueError("price_usdc_per_weth must be positive")

    sqrt_prices = np.sqrt(RAW_PRICE_SCALE / prices)
    below_range = sqrt_prices >= sqrt_upper
    above_range = sqrt_prices <= sqrt_lower
    in_range = ~(below_range | above_range)

    delta = np.zeros_like(prices, dtype=float)
    delta[below_range] = liquidity_raw * (sqrt_upper - sqrt_lower) / WETH_RAW_SCALE
    delta[in_range] = liquidity_raw * (sqrt_prices[in_range] - sqrt_lower) / WETH_RAW_SCALE
    return _maybe_scalar(delta, price_usdc_per_weth)


def lp_delta_from_ticks(
    price_usdc_per_weth: float | np.ndarray | pd.Series,
    liquidity_raw: float,
    tick_lower: int,
    tick_upper: int,
) -> float | np.ndarray:
    """Convenience wrapper computing LP delta from Uniswap tick bounds."""

    return lp_delta(
        price_usdc_per_weth,
        liquidity_raw,
        sqrt_raw_price_at_tick(tick_lower),
        sqrt_raw_price_at_tick(tick_upper),
    )


def compute_position_derivatives(
    prices_usdc_per_weth: float | np.ndarray | pd.Series,
    position: pd.Series | dict,
) -> pd.DataFrame:
    """Return price and delta for one Module 4 LP position row."""

    price_array = _as_float_array(prices_usdc_per_weth)
    liquidity = float(position["liquidity_raw"])
    tick_lower = int(position["tick_lower"])
    tick_upper = int(position["tick_upper"])

    return pd.DataFrame(
        {
            "price_usdc_per_weth": price_array,
            "lp_delta_weth": lp_delta_from_ticks(price_array, liquidity, tick_lower, tick_upper),
        }
    )


def compute_all_position_derivatives(
    prices_usdc_per_weth: float | np.ndarray | pd.Series,
    positions: pd.DataFrame,
) -> pd.DataFrame:
    """Return derivative curves for every representative Module 4 position."""

    frames: list[pd.DataFrame] = []
    for position in positions.sort_values("position_id").to_dict("records"):
        frame = compute_position_derivatives(prices_usdc_per_weth, position)
        frame["position_id"] = position["position_id"]
        frame["price_range_label"] = position.get("price_range_label", "")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def plot_delta_curves(
    positions: pd.DataFrame,
    *,
    figures_dir: Path,
    n_points: int = N_PRICE_POINTS,
) -> Path:
    """Plot Module 5 Task 5.1.2 delta curves for P1-P5."""

    p0 = float(positions["entry_price_usdc_per_weth"].iloc[0])
    prices = np.linspace(0.5 * p0, 1.5 * p0, n_points)
    curves = compute_all_position_derivatives(prices, positions)
    figures_dir.mkdir(parents=True, exist_ok=True)

    delta_path = figures_dir / "module5_lp_delta.png"

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (position_id, group) in enumerate(curves.groupby("position_id", sort=True)):
        label = f"{position_id} ({group['price_range_label'].iloc[0]})"
        ax.plot(
            group["price_usdc_per_weth"],
            group["lp_delta_weth"],
            color=_COLORS[i % len(_COLORS)],
            linewidth=2.0,
            label=label,
        )
    ax.axvline(p0, color="dimgray", linestyle=":", linewidth=1.4, label=f"$p_0$ = {p0:,.0f}")
    ax.set_xlabel("ETH price (USDC / WETH)")
    ax.set_ylabel("LP delta (WETH)")
    ax.set_title("Analytical LP Delta by Position")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.legend(fontsize=9, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(delta_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return delta_path


@app.command()
def run(
    positions_path: Path = typer.Option(
        DEFAULT_POSITIONS_PATH, help="Path to module4_lp_positions.parquet"
    ),
    figures_dir: Path = typer.Option(DEFAULT_FIGURES_DIR, help="Output figure directory"),
) -> None:
    """Plot LP delta curves for the representative positions."""

    positions = pd.read_parquet(positions_path)
    delta_path = plot_delta_curves(positions, figures_dir=figures_dir)
    print(f"Saved: {delta_path}")


def _validate_inputs(liquidity_raw: float, sqrt_lower: float, sqrt_upper: float) -> None:
    if liquidity_raw < 0:
        raise ValueError("liquidity_raw must be non-negative")
    if sqrt_lower <= 0 or sqrt_upper <= sqrt_lower:
        raise ValueError("expected 0 < sqrt_lower < sqrt_upper")


def _as_float_array(value: float | np.ndarray | pd.Series) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=float))


def _maybe_scalar(result: np.ndarray, original: float | np.ndarray | pd.Series) -> float | np.ndarray:
    if np.isscalar(original):
        return float(result[0])
    return result


if __name__ == "__main__":
    app()
