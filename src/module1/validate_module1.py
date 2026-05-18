"""Validation helpers for Module 1 datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.constants import POOL_DEPLOYMENT_BLOCK

STUDY_START = pd.Timestamp("2025-10-01T00:00:00Z")
STUDY_END = pd.Timestamp("2026-03-31T23:59:59Z")
EXPECTED_DAYS = 182

REQUIRED_COLUMNS: dict[str, set[str]] = {
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
        "liquidity_delta",
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
    "slot0_snapshots.parquet": {
        "snapshot_block",
        "snapshot_timestamp",
        "sqrt_price_x96",
        "price_usdc_per_weth",
        "current_tick",
        "observation_index",
        "unlocked",
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
}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def validate_module1(data_dir: Path) -> list[str]:
    processed = data_dir / "processed"
    messages: list[str] = []
    frames: dict[str, pd.DataFrame] = {}

    for filename, required in REQUIRED_COLUMNS.items():
        path = processed / filename
        df = _read(path)
        frames[filename] = df
        missing = sorted(required - set(df.columns))
        if missing:
            raise AssertionError(f"{filename} missing columns: {missing}")
        messages.append(f"OK schema: {filename} ({len(df):,} rows)")

    swaps = frames["swap_events.parquet"]
    _validate_event_log_identity(swaps, "swap_events.parquet")
    if swaps["block_timestamp"].min() < STUDY_START or swaps["block_timestamp"].max() > STUDY_END:
        raise AssertionError("swap_events.parquet is not bounded to the study window")
    messages.append("OK coverage: swaps are bounded to the study window")

    mint_burn = frames["mint_burn_events.parquet"]
    if int(mint_burn["block_number"].min()) > POOL_DEPLOYMENT_BLOCK + 100_000:
        raise AssertionError("mint_burn_events.parquet does not start near pool deployment")
    if (mint_burn["log_index"] < 0).any():
        raise AssertionError("mint_burn_events.parquet contains negative log_index values")
    if mint_burn.duplicated(["tx_hash", "log_index"]).any():
        raise AssertionError("mint_burn_events.parquet contains duplicate tx_hash/log_index rows")
    if not set(mint_burn["event_type"]).issubset({"mint", "burn"}):
        raise AssertionError("mint_burn_events.parquet contains unexpected event_type values")
    bad_ticks = (
        (mint_burn["tick_lower"] >= mint_burn["tick_upper"])
        | (mint_burn["tick_lower"] % 10 != 0)
        | (mint_burn["tick_upper"] % 10 != 0)
    )
    if bad_ticks.any():
        raise AssertionError("mint_burn_events.parquet contains invalid tick ranges")
    bad_sign = (
        ((mint_burn["event_type"] == "mint") & (mint_burn["liquidity_delta"].map(int) <= 0))
        | ((mint_burn["event_type"] == "burn") & (mint_burn["liquidity_delta"].map(int) > 0))
    )
    if bad_sign.any():
        raise AssertionError("mint_burn_events.parquet has liquidity_delta signs inconsistent with event_type")
    messages.append("OK coverage: mint/burn history starts near deployment")

    collects = frames["collect_events.parquet"]
    _validate_event_log_identity(collects, "collect_events.parquet")

    slot0 = frames["slot0_snapshots.parquet"]
    if len(slot0) != EXPECTED_DAYS:
        raise AssertionError(f"slot0_snapshots.parquet should have {EXPECTED_DAYS} rows")
    seconds_after_midnight = (
        slot0["snapshot_timestamp"].dt.hour * 3600
        + slot0["snapshot_timestamp"].dt.minute * 60
        + slot0["snapshot_timestamp"].dt.second
    )
    max_seconds_from_midnight = pd.concat(
        [seconds_after_midnight, 86_400 - seconds_after_midnight],
        axis=1,
    ).min(axis=1).max()
    if max_seconds_from_midnight > 30:
        raise AssertionError("slot0 snapshots are not close to 00:00 UTC")
    messages.append("OK coverage: slot0 has 182 midnight UTC snapshots")

    liquidity = frames["liquidity_snapshots.parquet"]
    if liquidity["snapshot_block"].nunique() != EXPECTED_DAYS:
        raise AssertionError("liquidity_snapshots.parquet should have 182 distinct blocks")
    messages.append("OK coverage: liquidity has 182 distinct snapshot blocks")

    return messages


def _validate_event_log_identity(df: pd.DataFrame, filename: str) -> None:
    if (df["log_index"] < 0).any():
        raise AssertionError(f"{filename} contains negative log_index values")
    if (df["log_index"] > 1_000_000).any():
        raise AssertionError(f"{filename} contains implausibly large log_index values")
    if df.duplicated(["tx_hash", "log_index"]).any():
        raise AssertionError(f"{filename} contains duplicate tx_hash/log_index rows")


def validate_slot0_against_swaps(data_dir: Path, tolerance_bps: float = 1.0) -> pd.DataFrame:
    """Compare each slot0 snapshot price with the last observed swap before it."""

    processed = data_dir / "processed"
    slot0 = pd.read_parquet(processed / "slot0_snapshots.parquet")
    swaps = pd.read_parquet(processed / "swap_events.parquet").sort_values("block_number")
    rows = []
    for snapshot in slot0.itertuples(index=False):
        prior = swaps[swaps["block_number"] <= snapshot.snapshot_block].tail(1)
        if prior.empty:
            continue
        swap = prior.iloc[0]
        diff_bps = (
            abs(snapshot.price_usdc_per_weth - swap.price_usdc_per_weth)
            / snapshot.price_usdc_per_weth
            * 10_000
        )
        rows.append(
            {
                "snapshot_block": snapshot.snapshot_block,
                "snapshot_timestamp": snapshot.snapshot_timestamp,
                "slot0_price_usdc_per_weth": snapshot.price_usdc_per_weth,
                "last_swap_block": int(swap.block_number),
                "last_swap_price_usdc_per_weth": swap.price_usdc_per_weth,
                "absolute_diff_bps": diff_bps,
                "within_tolerance": diff_bps <= tolerance_bps,
            }
        )
    return pd.DataFrame(rows)


def volume_crosscheck(data_dir: Path) -> pd.DataFrame:
    """Aggregate decoded swaps by direction for the report volume cross-check."""

    swaps = pd.read_parquet(
        data_dir / "processed" / "swap_events.parquet",
        columns=["trade_direction", "usd_notional", "amount0_usdc"],
    )
    by_dir = (
        swaps.assign(abs_usdc=swaps["amount0_usdc"].abs())
        .groupby("trade_direction", as_index=False)
        .agg(swaps=("usd_notional", "size"), abs_usdc_notional=("abs_usdc", "sum"))
    )
    total = pd.DataFrame(
        [
            {
                "trade_direction": "total",
                "swaps": int(by_dir["swaps"].sum()),
                "abs_usdc_notional": float(by_dir["abs_usdc_notional"].sum()),
            }
        ]
    )
    return pd.concat([by_dir, total], ignore_index=True)


def validate_liquidity_ticks_against_rpc(
    data_dir: Path,
    rpc,
    *,
    snapshot_block: int | None = None,
    sample_size: int = 10,
) -> pd.DataFrame:
    """Spot-check initialized ticks against direct archive ``pool.ticks()`` calls."""

    from decimal import Decimal

    from eth_abi import decode as abi_decode
    from eth_abi import encode as abi_encode
    from eth_utils import keccak

    from shared.constants import POOL_ADDRESS
    from shared.rpc import Call

    liquidity = pd.read_parquet(data_dir / "processed" / "liquidity_snapshots.parquet")
    if snapshot_block is None:
        snapshot_block = int(liquidity["snapshot_block"].iloc[0])
    sample = (
        liquidity[liquidity["snapshot_block"] == snapshot_block]
        .sort_values("tick")
        .head(sample_size)
    )
    selector = keccak(text="ticks(int24)")[:4]
    calls = [
        Call(target=POOL_ADDRESS, data=selector + abi_encode(["int24"], [int(row.tick)]))
        for row in sample.itertuples(index=False)
    ]
    results = rpc.multicall(calls, block=snapshot_block)
    decoded = []
    for row, raw in zip(sample.itertuples(index=False), results, strict=True):
        liquidity_gross, liquidity_net, *_rest = abi_decode(
            ["uint128", "int128", "uint256", "uint256", "int56", "uint160", "uint32", "bool"],
            raw,
        )
        decoded.append(
            {
                "snapshot_block": snapshot_block,
                "tick": int(row.tick),
                "file_liquidityGross": Decimal(row.liquidityGross),
                "rpc_liquidityGross": Decimal(liquidity_gross),
                "file_liquidityNet": Decimal(row.liquidityNet),
                "rpc_liquidityNet": Decimal(liquidity_net),
                "matches": int(row.liquidityGross) == liquidity_gross
                and int(row.liquidityNet) == liquidity_net,
            }
        )
    return pd.DataFrame(decoded)
