"""Parquet schemas and append-then-compact helpers for Module 1 deliverables."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

DEC = pa.decimal128(38, 0)
DEC256 = pa.decimal256(76, 0)   # for raw uint256/uint128 fee-growth accumulators


def swap_events_schema() -> pa.Schema:
    return pa.schema(
        [
            ("block_number", pa.int64()),
            ("block_timestamp", pa.timestamp("us", tz="UTC")),
            ("tx_hash", pa.string()),
            ("log_index", pa.int32()),
            ("sender", pa.string()),
            ("recipient", pa.string()),
            ("amount0_raw", DEC),            # signed int256 as decimal
            ("amount1_raw", DEC),
            ("amount0_usdc", pa.float64()),   # human units
            ("amount1_weth", pa.float64()),
            ("sqrt_price_x96", DEC),
            ("price_usdc_per_weth", pa.float64()),
            ("active_liquidity", DEC),
            ("tick", pa.int32()),
            ("trade_direction", pa.string()),       # taker perspective: buy_weth or sell_weth
            ("usd_notional", pa.float64()),
            ("date", pa.string()),
        ]
    )


def mint_burn_events_schema() -> pa.Schema:
    return pa.schema(
        [
            ("block_number", pa.int64()),
            ("block_timestamp", pa.timestamp("us", tz="UTC")),
            ("tx_hash", pa.string()),
            ("log_index", pa.int32()),
            ("event_type", pa.string()),      # 'mint' | 'burn'
            ("owner", pa.string()),
            ("tick_lower", pa.int32()),
            ("tick_upper", pa.int32()),
            ("liquidity_delta", DEC256),
            ("liquidity_amount_raw", DEC256),
            ("amount0_raw", DEC256),
            ("amount1_raw", DEC256),
            ("amount0_usdc", pa.float64()),
            ("amount1_weth", pa.float64()),
            ("date", pa.string()),
        ]
    )


def collect_events_schema() -> pa.Schema:
    return pa.schema(
        [
            ("block_number", pa.int64()),
            ("block_timestamp", pa.timestamp("us", tz="UTC")),
            ("tx_hash", pa.string()),
            ("log_index", pa.int32()),
            ("owner", pa.string()),
            ("recipient", pa.string()),
            ("tick_lower", pa.int32()),
            ("tick_upper", pa.int32()),
            ("amount0_raw", DEC256),
            ("amount1_raw", DEC256),
            ("amount0_usdc", pa.float64()),
            ("amount1_weth", pa.float64()),
            ("date", pa.string()),
        ]
    )


def slot0_schema() -> pa.Schema:
    return pa.schema(
        [
            ("date", pa.string()),
            ("snapshot_block", pa.int64()),
            ("snapshot_timestamp", pa.timestamp("us", tz="UTC")),
            ("sqrt_price_x96", DEC),
            ("price_usdc_per_weth", pa.float64()),
            ("current_tick", pa.int32()),
            ("observation_index", pa.int32()),
            ("observation_cardinality", pa.int32()),
            ("fee_protocol", pa.int32()),
            ("unlocked", pa.bool_()),
            ("fee_growth_global_0_x128", DEC256),
            ("fee_growth_global_1_x128", DEC256),
        ]
    )

def liquidity_snapshot_schema() -> pa.Schema:
    return pa.schema(
        [
            ("date", pa.string()),
            ("snapshot_block", pa.int64()),
            ("snapshot_timestamp", pa.timestamp("us", tz="UTC")),
            ("tick", pa.int32()),
            ("liquidityNet", DEC256),
            ("liquidityGross", DEC256),
            ("active_liquidity", DEC256),
            ("price_lower", pa.float64()),
            ("price_upper", pa.float64()),
            ("fee_growth_outside_0_x128", DEC256),
            ("fee_growth_outside_1_x128", DEC256),
        ]
    )


def append_rows(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    """Append rows as a new part-file under ``path`` (a directory of parts)."""
    if not rows:
        return
    path.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(_coerce_rows(rows, schema), schema=schema)
    existing = sorted(path.glob("part-*.parquet"))
    part_path = path / f"part-{len(existing):06d}.parquet"
    pq.write_table(table, part_path, compression="zstd")


def write_rows(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    """Write rows to a specific parquet file."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(_coerce_rows(rows, schema), schema=schema)
    pq.write_table(table, path, compression="zstd")


def compact(part_dir: Path, output_file: Path, schema: pa.Schema) -> int:
    """Merge all ``part-*.parquet`` files under ``part_dir`` into ``output_file``."""
    parts = sorted(part_dir.glob("part-*.parquet"))
    return compact_files(parts, output_file, schema)


def compact_files(parts: list[Path], output_file: Path, schema: pa.Schema) -> int:
    """Merge explicit parquet part files into ``output_file``."""
    if not parts:
        return 0
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tables = [pq.read_table(p, schema=schema) for p in parts]
    merged = pa.concat_tables(tables)
    if "log_index" in merged.column_names:
        log_index = merged.column("log_index")
        if int(pc.min(log_index).as_py()) < 0:
            raise ValueError("cannot compact event parts with negative log_index values")
    pq.write_table(merged, output_file, compression="zstd")
    return merged.num_rows


def _coerce_rows(rows: list[dict], schema: pa.Schema) -> list[dict]:
    """Coerce only signed tick fields before Arrow validates int32 columns."""
    tick_names = {
        field.name
        for field in schema
        if pa.types.is_int32(field.type)
        and field.name in {"tick", "tick_lower", "tick_upper", "current_tick"}
    }
    out: list[dict] = []
    for row in rows:
        fixed = dict(row)
        log_index = fixed.get("log_index")
        if isinstance(log_index, int) and log_index < 0:
            raise ValueError(f"invalid negative log_index: {log_index}")
        if isinstance(log_index, int) and log_index > 2**31 - 1:
            raise ValueError(f"invalid oversized log_index: {log_index}")
        for name in tick_names:
            value = fixed.get(name)
            if isinstance(value, int) and value > 2**31 - 1:
                fixed[name] = value - 2**32
        out.append(fixed)
    return out
