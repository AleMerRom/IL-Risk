
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_RESULTS_DIR = Path("data/results/module_2")
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures"
DEFAULT_ILR_BANDS = (0.001, 0.005, 0.01, 0.02, 0.05)


def ILR(k=DEFAULT_ILR_BANDS, df=None, slot0=None):
    if df is None:
        df = pd.read_parquet("data/processed/liquidity_snapshots.parquet")
    if slot0 is None:
        slot0 = pd.read_parquet("data/processed/slot0_snapshots.parquet")

    bands = _normalize_bands(k)
    work = df.copy()
    if "price_mid" not in work.columns:
        if {"price_lower", "price_upper"}.issubset(work.columns):
            work["price_mid"] = (work["price_lower"].astype(float) + work["price_upper"].astype(float)) / 2
        elif "price" in work.columns:
            work["price_mid"] = work["price"].astype(float)
        else:
            work["price_mid"] = 10**12 / (1.0001 ** work["tick"])
    work["active_liquidity_float"] = work["active_liquidity"].astype(float)

    slot0_by_date = slot0.set_index("date")
    rows = []
    for date, snapshot in work.groupby("date", sort=True):
        current_price = float(slot0_by_date.loc[date, "price_usdc_per_weth"])
        denominator = snapshot["active_liquidity_float"].clip(lower=0).sum()
        row = {
            "date": date,
            "snapshot_block": int(slot0_by_date.loc[date, "snapshot_block"]),
            "snapshot_timestamp": slot0_by_date.loc[date, "snapshot_timestamp"],
            "current_price_usdc_per_weth": current_price,
        }
        for band in bands:
            lower_bound = current_price * (1 - band)
            upper_bound = current_price * (1 + band)
            in_band = snapshot[
                (snapshot["price_mid"] >= lower_bound) &
                (snapshot["price_mid"] <= upper_bound)
            ]
            numerator = in_band["active_liquidity_float"].clip(lower=0).sum()
            row[_band_column(band)] = numerator / denominator if denominator > 0 else np.nan
        rows.append(row)

    result = pd.DataFrame(rows)
    _save_ilr_outputs(result, bands)
    return result


def _normalize_bands(k):
    if isinstance(k, (int, float)):
        return [float(k)]
    return [float(value) for value in k]


def _band_column(band):
    return f"ilr_{band * 100:g}pct"


def _save_ilr_outputs(result, bands):
    DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(DEFAULT_RESULTS_DIR / "ilr_metrics.parquet", index=False)

    dates = pd.to_datetime(result["date"])
    fig, ax = plt.subplots(figsize=(12, 6))
    for band in bands:
        column = _band_column(band)
        ax.plot(dates, result[column], label=f"+/-{band * 100:g}%")

    ax.set_title("In-Range Liquidity Ratio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Fraction of active liquidity")
    ax.set_ylim(0, 1)
    ax.legend(title="Price band")
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(DEFAULT_FIGURES_DIR / "fig_2_3_ilr_timeseries.png", dpi=300, bbox_inches="tight")
    plt.show()
