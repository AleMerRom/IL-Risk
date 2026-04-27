from pathlib import Path

import pandas as pd
import pytest


DATA_DIR = Path("data/processed")


def test_module1_parquets_have_pdf_required_columns() -> None:
    required = {
        "swap_events.parquet": {
            "block_number",
            "block_timestamp",
            "tx_hash",
            "amount0_raw",
            "amount0_usdc",
            "amount1_raw",
            "amount1_weth",
            "sqrt_price_x96",
            "price_usdc_per_weth",
            "active_liquidity",
            "tick",
            "trade_direction",
            "usd_notional",
        },
        "mint_burn_events.parquet": {
            "block_number",
            "block_timestamp",
            "tx_hash",
            "event_type",
            "owner",
            "tick_lower",
            "tick_upper",
            "liquidity_amount_raw",
            "amount0_raw",
            "amount0_usdc",
            "amount1_raw",
            "amount1_weth",
        },
        "collect_events.parquet": {
            "block_number",
            "block_timestamp",
            "tx_hash",
            "owner",
            "recipient",
            "tick_lower",
            "tick_upper",
            "amount0_raw",
            "amount0_usdc",
            "amount1_raw",
            "amount1_weth",
        },
        "liquidity_snapshots.parquet": {
            "snapshot_block",
            "snapshot_timestamp",
            "tick",
            "liquidityNet",
            "liquidityGross",
            "active_liquidity",
            "price_lower",
            "price_upper",
        },
        "slot0_snapshots.parquet": {
            "snapshot_block",
            "snapshot_timestamp",
            "sqrt_price_x96",
            "price_usdc_per_weth",
            "current_tick",
            "observation_index",
            "unlocked",
        },
    }

    for filename, columns in required.items():
        if not (DATA_DIR / filename).exists():
            pytest.skip(f"{filename} has not been generated yet")
        df = pd.read_parquet(DATA_DIR / filename)
        assert columns.issubset(df.columns), filename
