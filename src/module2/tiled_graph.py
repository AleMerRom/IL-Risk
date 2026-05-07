import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

FIGURE_DIR = Path("data/results/module_2/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_WINDOW = 0.20
PROFILE_BINS = 100

## TASK 2.1 PART B - LIQUIDITY PROFILE EVOLUTION ACROSS ALL DAILY SNAPSHOTS

def liquidity_profile_timeseries(df, slot0):
    work = df.merge(
        slot0[['date', 'price_usdc_per_weth']],
        on='date',
        how='inner',
    ).copy()
    work['active_liquidity_float'] = work['active_liquidity'].astype(float)
    work['relative_price'] = work['price'] / work['price_usdc_per_weth'] - 1
    work = work[
        (work['relative_price'] >= -PROFILE_WINDOW) &
        (work['relative_price'] <= PROFILE_WINDOW)
    ].copy()

    bin_edges = np.linspace(-PROFILE_WINDOW, PROFILE_WINDOW, PROFILE_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    work['price_bin'] = pd.cut(
        work['relative_price'],
        bins=bin_edges,
        labels=bin_centers,
        include_lowest=True,
    )

    heatmap = (
        work
        .groupby(['date', 'price_bin'], observed=True)['active_liquidity_float']
        .mean()
        .unstack('price_bin')
        .reindex(sorted(slot0['date'].unique()))
    )
    heatmap = heatmap.reindex(columns=bin_centers).interpolate(axis=1).ffill(axis=1).bfill(axis=1)

    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(
        heatmap.to_numpy(),
        aspect='auto',
        origin='lower',
        extent=[
            -PROFILE_WINDOW * 100,
            PROFILE_WINDOW * 100,
            0,
            len(heatmap.index) - 1,
        ],
        cmap='viridis',
    )
    tick_positions = np.linspace(0, len(heatmap.index) - 1, 8).astype(int)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([heatmap.index[i] for i in tick_positions])
    ax.set_xlabel('Price distance from current pool price (%)')
    ax.set_ylabel('Snapshot date')
    ax.set_title('Liquidity Profile Evolution Across Daily Snapshots')
    fig.colorbar(image, ax=ax, label='Active Liquidity')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_2_1b_liquidity_profile_timeseries.png", dpi=300, bbox_inches="tight")
    plt.show()

def _profiles_with_relative_price(df, slot0):
    work = df.merge(
        slot0[['date', 'price_usdc_per_weth']],
        on='date',
        how='inner',
    ).copy()
    work['active_liquidity_float'] = work['active_liquidity'].astype(float)
    work['relative_price_pct'] = (work['price'] / work['price_usdc_per_weth'] - 1) * 100
    work = work[
        (work['relative_price_pct'] >= -PROFILE_WINDOW * 100) &
        (work['relative_price_pct'] <= PROFILE_WINDOW * 100)
    ].copy()
    work['snapshot_month'] = pd.to_datetime(work['date']).dt.to_period('M').astype(str)
    return work

def liquidity_profile_monthly_multiples(df, slot0):
    work = _profiles_with_relative_price(df, slot0)
    months = sorted(work['snapshot_month'].unique())

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, month in zip(axes, months):
        month_data = work[work['snapshot_month'] == month]
        for _date, group in month_data.groupby('date', sort=True):
            group = group.sort_values('relative_price_pct')
            ax.step(
                group['relative_price_pct'],
                group['active_liquidity_float'],
                where='post',
                alpha=0.18,
                linewidth=0.8,
            )
        ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
        ax.set_title(month)
        ax.set_xlabel('Price distance from current price (%)')

    axes[0].set_ylabel('Active Liquidity')
    axes[3].set_ylabel('Active Liquidity')
    fig.suptitle('Liquidity Profiles by Month')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_2_1b_monthly_multiples.png", dpi=300, bbox_inches="tight")
    plt.show()

def liquidity_profile_weekly_tiles(df, slot0):
    work = _profiles_with_relative_price(df, slot0)
    dates = sorted(work['date'].unique())[::7]
    ncols = 4
    nrows = int(np.ceil(len(dates) / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 2.2 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, d in zip(axes, dates):
        group = work[work['date'] == d].sort_values('relative_price_pct')
        ax.step(
            group['relative_price_pct'],
            group['active_liquidity_float'],
            where='post',
            linewidth=1,
        )
        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.8)
        ax.set_title(d, fontsize=8)

    for ax in axes[len(dates):]:
        ax.axis('off')

    fig.supxlabel('Price distance from current price (%)')
    fig.supylabel('Active Liquidity')
    fig.suptitle('Weekly Liquidity Profile Tiles')
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_2_1b_weekly_tiles.png", dpi=300, bbox_inches="tight")
    plt.show()

def liquidity_profile_ridgeline(df, slot0):
    work = _profiles_with_relative_price(df, slot0)
    dates = sorted(work['date'].unique())
    scale = work['active_liquidity_float'].quantile(0.95)

    fig, ax = plt.subplots(figsize=(12, 14))
    for i, d in enumerate(dates):
        group = work[work['date'] == d].sort_values('relative_price_pct')
        y = group['active_liquidity_float'] / scale + i
        ax.step(
            group['relative_price_pct'],
            y,
            where='post',
            color=plt.cm.viridis(i / max(len(dates) - 1, 1)),
            linewidth=0.6,
            alpha=0.75,
        )

    tick_positions = np.linspace(0, len(dates) - 1, 10).astype(int)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([dates[i] for i in tick_positions])
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Current Price')
    ax.set_xlabel('Price distance from current price (%)')
    ax.set_ylabel('Snapshot date')
    ax.set_title('All-Snapshot Liquidity Profile Ridgeline')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_2_1b_ridgeline.png", dpi=300, bbox_inches="tight")
    plt.show()

def liquidity_profile_raw_overlay(df, slot0):
    lower_price = slot0['price_usdc_per_weth'].min() * (1 - PROFILE_WINDOW)
    upper_price = slot0['price_usdc_per_weth'].max() * (1 + PROFILE_WINDOW)

    work = df[
        (df['price'] >= lower_price) &
        (df['price'] <= upper_price)
    ].copy()
    work['active_liquidity_float'] = work['active_liquidity'].astype(float)

    fig, ax = plt.subplots(figsize=(12, 7))
    for _date, group in work.groupby('date', sort=True):
        group = group.sort_values('price')
        ax.step(
            group['price'],
            group['active_liquidity_float'],
            where='post',
            color='tab:blue',
            alpha=0.04,
            linewidth=0.8,
        )

    for price in slot0['price_usdc_per_weth']:
        ax.axvline(x=price, color='red', alpha=0.025, linewidth=0.6)

    ax.set_xlabel('USDC per WETH')
    ax.set_ylabel('Active Liquidity')
    ax.set_title('All Daily Liquidity Profiles Overlaid')
    ax.set_xlim(lower_price, upper_price)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_2_1b_all_snapshots_raw_overlay.png", dpi=300, bbox_inches="tight")
    plt.show()