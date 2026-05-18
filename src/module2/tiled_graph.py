import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter
from pathlib import Path

FIGURE_DIR = Path("data/results/module_2/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_WINDOW = 0.20
PROFILE_BINS = 100
LIQUIDITY_SCALE = 1e18
FULL_PRICE_QUANTILE_RANGE = (0.01, 0.99)

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

def _tick_to_price(tick):
    return 10**12 / (1.0001 ** tick)

def _profiles_with_absolute_price(df, slot0):
    work = df.copy()
    if 'price' not in work.columns:
        work['price'] = work['tick'].apply(_tick_to_price)
    work = work.merge(
        slot0[['date', 'price_usdc_per_weth']],
        on='date',
        how='inner',
    )
    work['active_liquidity_float'] = work['active_liquidity'].astype(float)
    work['snapshot_month'] = pd.to_datetime(work['date']).dt.to_period('M').astype(str)
    return work

def _liquidity_heatmap_by_price(work, dates, bin_centers):
    heatmap = (
        work
        .groupby(['date', 'price_bin'], observed=True)['active_liquidity_float']
        .mean()
        .unstack('price_bin')
        .reindex(dates)
    )
    return heatmap.reindex(columns=bin_centers).interpolate(axis=1).ffill(axis=1).bfill(axis=1)

def _absolute_price_heatmap_inputs(df, slot0):
    work = _profiles_with_absolute_price(df, slot0)
    lower_price = slot0['price_usdc_per_weth'].min() * (1 - PROFILE_WINDOW)
    upper_price = slot0['price_usdc_per_weth'].max() * (1 + PROFILE_WINDOW)
    work = work[
        (work['price'] >= lower_price) &
        (work['price'] <= upper_price)
    ].copy()

    bin_edges = np.linspace(lower_price, upper_price, PROFILE_BINS + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    work['price_bin'] = pd.cut(
        work['price'],
        bins=bin_edges,
        labels=bin_centers,
        include_lowest=True,
    )
    return work, lower_price, upper_price, bin_centers

def _monthly_heatmaps(work, bin_centers):
    heatmaps = {}
    for month in sorted(work['snapshot_month'].unique()):
        month_data = work[work['snapshot_month'] == month]
        dates = sorted(month_data['date'].unique())
        heatmaps[month] = (dates, _liquidity_heatmap_by_price(month_data, dates, bin_centers))
    return heatmaps

def _positive_values(heatmaps):
    values = np.concatenate([
        heatmap.to_numpy(dtype=float).ravel()
        for _dates, heatmap in heatmaps.values()
    ])
    return values[np.isfinite(values) & (values > 0)]

def _log_norm_for_heatmaps(heatmaps, vmax_quantile):
    positive_values = _positive_values(heatmaps)
    vmin = np.quantile(positive_values, 0.01)
    vmax = np.quantile(positive_values, vmax_quantile)
    if not np.isfinite(vmin) or vmin <= 0:
        vmin = positive_values.min()
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = positive_values.max()
    return LogNorm(vmin=vmin, vmax=vmax, clip=True)

def _render_monthly_price_heatmaps(
    heatmaps,
    slot0,
    lower_price,
    upper_price,
    norm,
    title,
    colorbar_label,
    output_path,
):
    months = list(heatmaps.keys())
    ncols = 3
    nrows = int(np.ceil(len(months) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 9), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    image = None

    for i, (ax, month) in enumerate(zip(axes, months)):
        dates, heatmap = heatmaps[month]
        values = np.ma.masked_less_equal(heatmap.to_numpy(dtype=float).T, 0)
        day_numbers = pd.to_datetime(dates).day.to_numpy()

        image = ax.imshow(
            values,
            aspect='auto',
            origin='lower',
            extent=[
                day_numbers.min() - 0.5,
                day_numbers.max() + 0.5,
                lower_price,
                upper_price,
            ],
            cmap='viridis',
            norm=norm,
        )

        month_slot0 = (
            slot0[slot0['date'].isin(dates)]
            .set_index('date')
            .reindex(dates)
        )
        line = ax.plot(
            day_numbers,
            month_slot0['price_usdc_per_weth'],
            color='white',
            linewidth=1.4,
            label='Current pool price',
        )[0]
        line.set_path_effects([
            path_effects.Stroke(linewidth=2.4, foreground='black'),
            path_effects.Normal(),
        ])

        ax.set_title(month)
        if i % ncols == 0:
            ax.set_ylabel('USDC per WETH')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f'{y:,.0f}'))
        ax.set_xticks(np.linspace(day_numbers.min(), day_numbers.max(), min(5, len(day_numbers))).astype(int))

    for ax in axes[len(months):]:
        ax.axis('off')

    for ax in axes[:len(months)]:
        ax.set_xlabel('Day of month')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.935), frameon=False)
    fig.suptitle(title, y=0.98)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.17, hspace=0.45, wspace=0.16)
    cbar_ax = fig.add_axes([0.25, 0.06, 0.50, 0.02])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(colorbar_label)
    plt.savefig(FIGURE_DIR / output_path, dpi=300, bbox_inches="tight")
    plt.show()

def liquidity_profile_monthly_price_heatmaps(df, slot0, vmax_quantile=0.995):
    work, lower_price, upper_price, bin_centers = _absolute_price_heatmap_inputs(df, slot0)
    raw_heatmaps = _monthly_heatmaps(work, bin_centers)
    scaled_heatmaps = {
        month: (dates, heatmap / LIQUIDITY_SCALE)
        for month, (dates, heatmap) in raw_heatmaps.items()
    }

    _render_monthly_price_heatmaps(
        heatmaps=scaled_heatmaps,
        slot0=slot0,
        lower_price=lower_price,
        upper_price=upper_price,
        norm=_log_norm_for_heatmaps(scaled_heatmaps, vmax_quantile),
        title='Monthly Liquidity Profile Heatmaps by Actual Price',
        colorbar_label=f'Active Liquidity (L / 1e18, log scale, {vmax_quantile:.1%} cap)',
        output_path="fig_2_1b_monthly_price_heatmaps.png",
    )

def liquidity_profile_full_price_heatmap_annex(df, slot0, vmax_quantile=0.995):
    work = _profiles_with_absolute_price(df, slot0)
    positive_prices = work.loc[work['price'] > 0, 'price']
    lower_price = positive_prices.quantile(FULL_PRICE_QUANTILE_RANGE[0])
    upper_price = positive_prices.quantile(FULL_PRICE_QUANTILE_RANGE[1])
    work = work[
        (work['price'] >= lower_price) &
        (work['price'] <= upper_price)
    ].copy()

    log_edges = np.linspace(np.log10(lower_price), np.log10(upper_price), PROFILE_BINS + 1)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    work['log_price'] = np.log10(work['price'])
    work['price_bin'] = pd.cut(
        work['log_price'],
        bins=log_edges,
        labels=log_centers,
        include_lowest=True,
    )

    raw_heatmaps = _monthly_heatmaps(work, log_centers)
    scaled_heatmaps = {
        month: (dates, heatmap / LIQUIDITY_SCALE)
        for month, (dates, heatmap) in raw_heatmaps.items()
    }

    months = list(scaled_heatmaps.keys())
    ncols = 3
    nrows = int(np.ceil(len(months) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 9), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    norm = _log_norm_for_heatmaps(scaled_heatmaps, vmax_quantile)
    image = None

    for i, (ax, month) in enumerate(zip(axes, months)):
        dates, heatmap = scaled_heatmaps[month]
        values = np.ma.masked_less_equal(heatmap.to_numpy(dtype=float).T, 0)
        day_numbers = pd.to_datetime(dates).day.to_numpy()
        image = ax.imshow(
            values,
            aspect='auto',
            origin='lower',
            extent=[
                day_numbers.min() - 0.5,
                day_numbers.max() + 0.5,
                log_edges[0],
                log_edges[-1],
            ],
            cmap='viridis',
            norm=norm,
        )

        month_slot0 = (
            slot0[slot0['date'].isin(dates)]
            .set_index('date')
            .reindex(dates)
        )
        line = ax.plot(
            day_numbers,
            np.log10(month_slot0['price_usdc_per_weth']),
            color='white',
            linewidth=1.4,
            label='Current pool price',
        )[0]
        line.set_path_effects([
            path_effects.Stroke(linewidth=2.4, foreground='black'),
            path_effects.Normal(),
        ])

        ax.set_title(month)
        ax.set_xticks(np.linspace(day_numbers.min(), day_numbers.max(), min(5, len(day_numbers))).astype(int))
        ax.set_xlabel('Day of month')
        if i % ncols == 0:
            ax.set_ylabel('USDC per WETH, log scale')
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _pos: f'{10 ** y:,.0f}'))

    for ax in axes[len(months):]:
        ax.axis('off')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.935), frameon=False)
    q_low, q_high = FULL_PRICE_QUANTILE_RANGE
    fig.suptitle(
        f'Appendix: Monthly Liquidity Heatmaps Across the {q_low:.0%}-{q_high:.0%} Price Range',
        y=0.98,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.84, bottom=0.17, hspace=0.45, wspace=0.16)
    cbar_ax = fig.add_axes([0.25, 0.06, 0.50, 0.02])
    cbar = fig.colorbar(image, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(f'Active Liquidity (L / 1e18, log scale, {vmax_quantile:.1%} cap)')
    plt.savefig(
        FIGURE_DIR / "annex_fig_2_1b_full_price_monthly_heatmaps.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

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
    ax.set_ylim(0, work['active_liquidity_float'].quantile(0.995))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_2_1b_all_snapshots_raw_overlay.png", dpi=300, bbox_inches="tight")
    plt.show()
