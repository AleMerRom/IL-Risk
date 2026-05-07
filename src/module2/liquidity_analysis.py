#This file is part of Module 2: Liquidity Distribution Analysis
import pandas as pd
from module2.tiled_graph import *
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from TVL import run_tvl_analysis

FIGURE_DIR = Path("data/results/module_2/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_WINDOW = 0.20
PROFILE_BINS = 100

def main():
    df = pd.read_parquet('data/processed/liquidity_snapshots.parquet')
    slot0 = pd.read_parquet('data/processed/slot0_snapshots.parquet')
    df['price'] = df['tick'].apply(tick_to_price)

    ##TASK 2.1 PART A - LIQUIDITY PROFILES AT KEY SNAPSHOT DATES
    liquidity_snapshots(df, slot0)
    
    ##TASK 2.1 PART B - LIQUIDITY PROFILE EVOLUTION ACROSS ALL DAILY SNAPSHOTS
    liquidity_profile_timeseries(df, slot0)
    liquidity_profile_monthly_multiples(df, slot0)
    liquidity_profile_weekly_tiles(df, slot0)
    liquidity_profile_ridgeline(df, slot0)
    liquidity_profile_raw_overlay(df, slot0)
    run_tvl_analysis()


def find_snapshot_dates(df):
    start_date = df['date'].min()
    end_date = df['date'].max()
    high_volatility_date = "2026-02-06"
    return [start_date, high_volatility_date, end_date]

def get_snapshot(df, date):
    snapshot = df[df['date'] == date].copy()
    snapshot = snapshot.sort_values('price')
    return snapshot

def tick_to_price(tick):
    return 10**12 /(1.0001 ** tick)

def focus_snapshot(snapshot, current_price):
    snapshot = snapshot.copy()
    snapshot['active_liquidity_float'] = snapshot['active_liquidity'].astype(float)
    lower_bound = current_price * (1 - PROFILE_WINDOW)
    upper_bound = current_price * (1 + PROFILE_WINDOW)
    focused = snapshot[
        (snapshot['price'] >= lower_bound) &
        (snapshot['price'] <= upper_bound)
    ].copy()
    return focused

def liquidity_snapshots(df, slot0):
    dates = find_snapshot_dates(df)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=True)

    for ax, d in zip(axes, dates):
        snapshot = get_snapshot(df, d)
        current_price = slot0.loc[slot0['date'] == d, 'price_usdc_per_weth'].iloc[0]
        snapshot = focus_snapshot(snapshot, current_price)
        ax.step(
            snapshot['price'],
            snapshot['active_liquidity_float'],
            where='post',
        )
        ax.set_title(f'Liquidity Snapshot on {d}')
        ax.set_xlabel('USDC per WETH')
        ax.axvline(x=current_price, color='red', linestyle='--', label='Current Price')
        ax.legend()
    
    axes[0].set_ylabel("Active Liquidity")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "fig_2_1_liquidity_profiles.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
