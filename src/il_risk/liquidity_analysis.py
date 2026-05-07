#This file is part of Module 2: Liquidity Distribution Analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_parquet('data/processed/liquidity_snapshots.parquet')
    df['price'] = df['tick'].apply(tick_to_price)

    liquidity_snapshots(df)

    

def find_snapshot_dates(df):
    start_date = df['date'].min()
    end_date = df['date'].max()
    high_volatility_date = "2026-02-06"
    return [start_date, high_volatility_date, end_date]

def get_snapshot(df, date):
    snapshot = df[df['date'] == date]
    snapshot.sort_values(by='price', ascending=True, inplace=True)
    return snapshot

def tick_to_price(tick):
    return 1.0001 ** tick

def liquidity_snapshots(df):
    dates = find_snapshot_dates(df)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharey=True)

    for ax, d in zip(axes, dates):
        snapshot = get_snapshot(df, d)
        ax.bar(snapshot['price'], snapshot['liquidity'], width=0.0005)
        ax.set_title(f'Liquidity Snapshot on {d}')
        ax.set_xlabel('Price')
        ax.set_ylabel('Liquidity')
        current_price = tick_to_price(snapshot['tick'].iloc[0])
        ax.axvline(x=current_price, color='red', linestyle='--', label='Current Price')
    
    axes[0].set_ylabel("Active Liquidity")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

