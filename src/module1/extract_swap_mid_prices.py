"""Module 3 support — pre-swap slot0 prices for effective-spread estimates."""

from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
from eth_abi import decode as abi_decode
from eth_utils import keccak

from shared.constants import POOL_ADDRESS, Q96, TOKEN0_DECIMALS, TOKEN1_DECIMALS
from shared.rpc import RpcClient

log = logging.getLogger(__name__)

_SLOT0_SELECTOR = keccak(text="slot0()")[:4]
DEFAULT_RESULTS_DIR = Path("data/results/module_3")
_USDC_SCALE = Decimal(10) ** TOKEN0_DECIMALS
_WETH_SCALE = Decimal(10) ** TOKEN1_DECIMALS
_Q96 = Decimal(Q96)
_SIZE_BUCKET_EDGES_USD = (
    0,
    1_000,
    10_000,
    50_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    float("inf"),
)


def extract_swap_mid_prices(
    rpc: RpcClient,
    *,
    data_dir: Path,
    swap_events_path: Path | None = None,
    output_path: Path | None = None,
    limit: int | None = None,
    include_timestamps: bool = False,
    from_block: int | None = None,
    to_block: int | None = None,
    batch_size: int = 100,
    resume: bool = True,
    sample_blocks: int | None = None,
    sample_seed: int = 413,
) -> pd.DataFrame:
    """Fetch ``slot0()`` at ``block_number - 1`` for each unique swap block.

    Task 3.4 defines the mid price as the pool price at the block before each
    observed swap.  ``swap_events.parquet`` only stores post-swap event prices,
    so this support extraction performs one archive ``eth_call`` per unique swap
    block and writes ``swap_mid_prices.parquet``.

    Timestamps are disabled by default because Task 3.4 only needs
    ``block_number`` and ``mid_price_usdc_per_weth``.  Enabling timestamps adds
    an extra ``eth_getBlockByNumber`` call per unique swap block.

    Progress is written as parquet part files after every successful batch, so
    interrupted runs can resume without redoing completed blocks.

    If ``sample_blocks`` is set, blocks are selected by a reproducible
    stratified sample over day, trade direction, and USD-notional bucket.  The
    sampling unit is the swap block, because all swaps in one block share the
    same ``block_number - 1`` mid price.
    """

    swap_events_path = swap_events_path or data_dir / "processed" / "swap_events.parquet"
    output_path = output_path or DEFAULT_RESULTS_DIR / "swap_mid_prices.parquet"
    filters = []
    if from_block is not None:
        filters.append(("block_number", ">=", from_block))
    if to_block is not None:
        filters.append(("block_number", "<=", to_block))
    columns = ["block_number"]
    if sample_blocks is not None:
        columns += ["date", "trade_direction", "usd_notional"]
    swaps = pd.read_parquet(swap_events_path, columns=columns, filters=filters or None)
    if swaps.empty:
        raise ValueError(f"{swap_events_path} contains no swaps")

    all_blocks = sorted(int(block) for block in swaps["block_number"].dropna().unique())
    blocks = _select_mid_price_blocks(
        swaps,
        sample_blocks=sample_blocks,
        sample_seed=sample_seed,
    )
    if limit is not None:
        blocks = blocks[:limit]

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    parts_dir = output_path.parent / f"{output_path.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    completed_blocks = _completed_mid_price_blocks(parts_dir) if resume else set()

    total = len(blocks)
    for start in range(0, total, batch_size):
        batch_blocks = blocks[start : start + batch_size]
        pending_blocks = [block for block in batch_blocks if block not in completed_blocks]
        if not pending_blocks:
            continue
        pre_swap_blocks = [block - 1 for block in pending_blocks]
        slot0_rows = fetch_slot0_many(rpc, pre_swap_blocks, batch_size=batch_size)
        timestamps = {}
        if include_timestamps:
            timestamps = {
                block: datetime.fromtimestamp(
                    int(rpc.get_block(block)["timestamp"], 16),
                    tz=timezone.utc,
                )
                for block in pre_swap_blocks
            }
        batch_rows = []
        for block, pre_swap_block, slot0 in zip(pending_blocks, pre_swap_blocks, slot0_rows, strict=True):
            row = {
                "block_number": block,
                "pre_swap_block": pre_swap_block,
                "sqrt_price_x96": Decimal(slot0["sqrt_price_x96"]),
                "mid_price_usdc_per_weth": _price_usdc_per_weth(slot0["sqrt_price_x96"]),
                "current_tick": slot0["tick"],
            }
            if include_timestamps:
                row["pre_swap_block_timestamp"] = timestamps[pre_swap_block]
            batch_rows.append(row)
        part_path = parts_dir / f"part-{pending_blocks[0]}-{pending_blocks[-1]}.parquet"
        _write_part_atomic(pd.DataFrame(batch_rows), part_path)
        completed_blocks.update(pending_blocks)
        done = min(start + len(batch_blocks), total)
        if done == len(batch_blocks) or done % 500 == 0 or done == total:
            log.info("swap mid prices: fetched %d/%d unique blocks", done, total)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = _compact_mid_price_parts(parts_dir, blocks)
    out.to_parquet(output_path, index=False)
    _write_mid_price_metadata(
        output_path,
        total_unique_blocks=len(all_blocks),
        selected_blocks=len(blocks),
        sample_blocks=sample_blocks,
        sample_seed=sample_seed,
        from_block=from_block,
        to_block=to_block,
        include_timestamps=include_timestamps,
        batch_size=batch_size,
    )
    return out


def fetch_slot0_at_block(rpc: RpcClient, block: int) -> dict:
    """Return pool ``slot0()`` decoded at a historical block."""

    raw = rpc.call(POOL_ADDRESS, _SLOT0_SELECTOR, block=block)
    return _decode_slot0(raw)


def fetch_slot0_many(rpc: RpcClient, blocks: list[int], *, batch_size: int = 100) -> list[dict]:
    """Return pool ``slot0()`` decoded for many historical blocks."""

    if hasattr(rpc, "call_many"):
        raw_results = rpc.call_many(
            [(POOL_ADDRESS, _SLOT0_SELECTOR, block) for block in blocks],
            batch_size=batch_size,
        )
    else:
        raw_results = [rpc.call(POOL_ADDRESS, _SLOT0_SELECTOR, block=block) for block in blocks]
    return [_decode_slot0(raw) for raw in raw_results]


def _decode_slot0(raw: bytes) -> dict:
    (
        sqrt_price_x96,
        tick,
        observation_index,
        observation_cardinality,
        observation_cardinality_next,
        fee_protocol,
        unlocked,
    ) = abi_decode(["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"], raw)
    return {
        "sqrt_price_x96": sqrt_price_x96,
        "tick": tick,
        "observation_index": observation_index,
        "observation_cardinality": observation_cardinality,
        "observation_cardinality_next": observation_cardinality_next,
        "fee_protocol": fee_protocol,
        "unlocked": unlocked,
    }


def _price_weth_per_usdc(sqrt_price_x96: int) -> float:
    price_raw = (Decimal(sqrt_price_x96) / _Q96) ** 2
    return float(price_raw * (_USDC_SCALE / _WETH_SCALE))


def _price_usdc_per_weth(sqrt_price_x96: int) -> float:
    return 1.0 / _price_weth_per_usdc(sqrt_price_x96)


def _select_mid_price_blocks(
    swaps: pd.DataFrame,
    *,
    sample_blocks: int | None,
    sample_seed: int,
) -> list[int]:
    all_blocks = sorted(int(block) for block in swaps["block_number"].dropna().unique())
    if sample_blocks is None or sample_blocks >= len(all_blocks):
        return all_blocks
    if sample_blocks <= 0:
        raise ValueError("sample_blocks must be positive")

    required = {"date", "trade_direction", "usd_notional"}
    missing = required - set(swaps.columns)
    if missing:
        raise ValueError(f"stratified block sampling requires columns: {sorted(missing)}")

    frame = swaps[["block_number", "date", "trade_direction", "usd_notional"]].copy()
    frame = frame.dropna(subset=["block_number", "date", "trade_direction", "usd_notional"])
    frame["block_number"] = frame["block_number"].astype(int)
    frame["usd_notional"] = frame["usd_notional"].astype(float).clip(lower=0.0)
    frame["size_bucket"] = pd.cut(
        frame["usd_notional"],
        bins=_SIZE_BUCKET_EDGES_USD,
        include_lowest=True,
        right=False,
        labels=False,
    )
    # Use the largest swap in each block as the block's stratum representative.
    representative_idx = frame.groupby("block_number")["usd_notional"].idxmax()
    block_frame = frame.loc[
        representative_idx,
        ["block_number", "date", "trade_direction", "size_bucket"],
    ]

    strata = [
        (key, group.sort_values("block_number"))
        for key, group in block_frame.groupby(["date", "trade_direction", "size_bucket"])
    ]
    allocations = _proportional_allocations(
        [len(group) for _key, group in strata],
        sample_blocks,
    )
    sampled: list[int] = []
    for idx, (_key, group) in enumerate(strata):
        n = allocations[idx]
        if n <= 0:
            continue
        sampled.extend(
            int(block)
            for block in group.sample(n=n, random_state=sample_seed + idx)["block_number"]
        )
    return sorted(sampled)


def _proportional_allocations(counts: list[int], target: int) -> list[int]:
    if target <= 0:
        return [0 for _count in counts]
    total = sum(counts)
    if total <= target:
        return counts[:]
    raw = [target * count / total for count in counts]
    allocations = [int(value) for value in raw]

    if target >= len(counts):
        allocations = [
            max(1, allocation) if count > 0 else 0
            for allocation, count in zip(allocations, counts, strict=True)
        ]

    while sum(allocations) > target:
        candidates = [
            idx
            for idx, allocation in enumerate(allocations)
            if allocation > (1 if target >= len(counts) else 0)
        ]
        idx = min(candidates, key=lambda i: raw[i] - allocations[i])
        allocations[idx] -= 1
    while sum(allocations) < target:
        candidates = [
            idx for idx, count in enumerate(counts) if allocations[idx] < count
        ]
        idx = max(candidates, key=lambda i: raw[i] - allocations[i])
        allocations[idx] += 1
    return allocations


def _completed_mid_price_blocks(parts_dir: Path) -> set[int]:
    completed: set[int] = set()
    for path in sorted(parts_dir.glob("part-*.parquet")):
        part = pd.read_parquet(path, columns=["block_number"])
        completed.update(int(block) for block in part["block_number"])
    return completed


def _compact_mid_price_parts(parts_dir: Path, expected_blocks: list[int]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted(parts_dir.glob("part-*.parquet"))]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out["block_number"].isin(expected_blocks)]
    out = out.drop_duplicates("block_number", keep="last")
    out = out.sort_values("block_number").reset_index(drop=True)
    missing = set(expected_blocks) - set(int(block) for block in out["block_number"])
    if missing:
        sample = sorted(missing)[:10]
        raise ValueError(f"missing swap mid prices for {len(missing)} blocks, sample={sample}")
    return out


def _write_part_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.parent / f".{path.name}.tmp"
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_mid_price_metadata(
    output_path: Path,
    *,
    total_unique_blocks: int,
    selected_blocks: int,
    sample_blocks: int | None,
    sample_seed: int,
    from_block: int | None,
    to_block: int | None,
    include_timestamps: bool,
    batch_size: int,
) -> None:
    metadata = {
        "method": "all_unique_swap_blocks" if sample_blocks is None else "stratified_block_sample",
        "sampling_unit": "swap block",
        "total_unique_swap_blocks": total_unique_blocks,
        "selected_unique_swap_blocks": selected_blocks,
        "sample_blocks_requested": sample_blocks,
        "sample_seed": sample_seed if sample_blocks is not None else None,
        "strata": ["date", "trade_direction", "usd_notional_bucket"] if sample_blocks is not None else None,
        "usd_notional_bucket_edges": (
            _json_safe_bucket_edges(_SIZE_BUCKET_EDGES_USD) if sample_blocks is not None else None
        ),
        "from_block": from_block,
        "to_block": to_block,
        "include_timestamps": include_timestamps,
        "batch_size": batch_size,
        "justification": (
            "Task 3.4 accepts a subset. Blocks are sampled rather than individual swaps "
            "because all swaps in a block share the same previous-block mid price. "
            "Stratifying by day, direction, and notional bucket preserves time variation, "
            "side, and trade-size composition for the effective-spread comparison."
        )
        if sample_blocks is not None
        else "Full extraction over every unique swap block.",
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True)
    )


def _json_safe_bucket_edges(edges: Iterable[float]) -> list[float | str]:
    return ["Infinity" if edge == float("inf") else edge for edge in edges]
