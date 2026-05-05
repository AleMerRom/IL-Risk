"""Module 3 support — pre-swap slot0 prices for effective-spread estimates."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pandas as pd
from eth_abi import decode as abi_decode
from eth_utils import keccak

from il_risk.constants import POOL_ADDRESS, Q96, TOKEN0_DECIMALS, TOKEN1_DECIMALS
from il_risk.rpc.client import RpcClient

log = logging.getLogger(__name__)

_SLOT0_SELECTOR = keccak(text="slot0()")[:4]
_USDC_SCALE = Decimal(10) ** TOKEN0_DECIMALS
_WETH_SCALE = Decimal(10) ** TOKEN1_DECIMALS
_Q96 = Decimal(Q96)


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
    """

    swap_events_path = swap_events_path or data_dir / "processed" / "swap_events.parquet"
    output_path = output_path or data_dir / "processed" / "swap_mid_prices.parquet"
    filters = []
    if from_block is not None:
        filters.append(("block_number", ">=", from_block))
    if to_block is not None:
        filters.append(("block_number", "<=", to_block))
    swaps = pd.read_parquet(swap_events_path, columns=["block_number"], filters=filters or None)
    if swaps.empty:
        raise ValueError(f"{swap_events_path} contains no swaps")

    blocks = sorted(int(block) for block in swaps["block_number"].dropna().unique())
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
