"""Module 1 — Mint, Burn, and Collect event extraction.

Mint and Burn share a near-identical decoded shape (liquidity delta + token
amounts, same owner / tick range). We fetch them as separate log queries then
merge into one output, tagging ``event_type``.
"""

from __future__ import annotations

import logging
from datetime import timezone, datetime
from decimal import Decimal
from pathlib import Path

from eth_abi import decode as abi_decode

from il_risk.constants import POOL_ADDRESS
from il_risk.constants import TOKEN0_DECIMALS, TOKEN1_DECIMALS
from il_risk.events import BURN_TOPIC0, COLLECT_TOPIC0, MINT_TOPIC0
from il_risk.extract_swaps import _ts_to_date
from il_risk.block_index import BlockIndex
from il_risk.rpc import RpcClient, fetch_logs_chunked, fetch_logs_parallel
from il_risk.schemas import (
    append_rows,
    collect_events_schema,
    compact,
    mint_burn_events_schema,
)

log = logging.getLogger(__name__)
_USDC_SCALE = Decimal(10) ** TOKEN0_DECIMALS
_WETH_SCALE = Decimal(10) ** TOKEN1_DECIMALS


def _int24_from_topic(topic: str) -> int:
    """Decode int24 right-aligned in a 32-byte ABI-encoded topic."""
    raw = int(topic, 16) & ((1 << 24) - 1)  # keep only the lower 24 bits
    if raw & (1 << 23):  # sign-extend if negative
        raw -= 1 << 24
    return raw


def _decode_mint_or_burn(log_dict: dict, *, is_mint: bool) -> dict:
    topics = log_dict["topics"]
    # Mint: topics = [topic0, owner, tickLower, tickUpper]   (sender is in data)
    # Burn: topics = [topic0, owner, tickLower, tickUpper]
    owner = "0x" + topics[1][-40:]
    tick_lower = _int24_from_topic(topics[2])
    tick_upper = _int24_from_topic(topics[3])
    data = bytes.fromhex(log_dict["data"][2:])
    if is_mint:
        _sender, amount, amount0, amount1 = abi_decode(
            ["address", "uint128", "uint256", "uint256"], data
        )
    else:
        amount, amount0, amount1 = abi_decode(
            ["uint128", "uint256", "uint256"], data
        )
    return {
        "owner": owner,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "amount": amount,
        "amount0": amount0,
        "amount1": amount1,
    }


def _decode_collect(log_dict: dict) -> dict:
    topics = log_dict["topics"]
    owner = "0x" + topics[1][-40:]
    tick_lower = _int24_from_topic(topics[2])
    tick_upper = _int24_from_topic(topics[3])
    recipient, amount0, amount1 = abi_decode(
        ["address", "uint128", "uint128"], bytes.fromhex(log_dict["data"][2:])
    )
    return {
        "owner": owner,
        "recipient": recipient,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "amount0": amount0,
        "amount1": amount1,
    }


def _transform(log_dict: dict, timestamp: int, *, is_mint: bool) -> dict:
    ev = _decode_mint_or_burn(log_dict, is_mint=is_mint)
    signed_liq = ev["amount"] if is_mint else -ev["amount"]
    return {
        "block_number": int(log_dict["blockNumber"], 16),
        "block_timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc),
        "tx_hash": log_dict["transactionHash"],
        "log_index": int(log_dict["logIndex"], 16),
        "event_type": "mint" if is_mint else "burn",
        "owner": ev["owner"],
        "tick_lower": ev["tick_lower"],
        "tick_upper": ev["tick_upper"],
        "liquidity_delta": Decimal(signed_liq),
        "liquidity_amount_raw": Decimal(ev["amount"]),
        "amount0_raw": Decimal(ev["amount0"]),
        "amount1_raw": Decimal(ev["amount1"]),
        "amount0_usdc": float(Decimal(ev["amount0"]) / _USDC_SCALE),
        "amount1_weth": float(Decimal(ev["amount1"]) / _WETH_SCALE),
        "date": _ts_to_date(timestamp),
    }


def _transform_collect(log_dict: dict, timestamp: int) -> dict:
    ev = _decode_collect(log_dict)
    return {
        "block_number": int(log_dict["blockNumber"], 16),
        "block_timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc),
        "tx_hash": log_dict["transactionHash"],
        "log_index": int(log_dict["logIndex"], 16),
        "owner": ev["owner"],
        "recipient": ev["recipient"],
        "tick_lower": ev["tick_lower"],
        "tick_upper": ev["tick_upper"],
        "amount0_raw": Decimal(ev["amount0"]),
        "amount1_raw": Decimal(ev["amount1"]),
        "amount0_usdc": float(Decimal(ev["amount0"]) / _USDC_SCALE),
        "amount1_weth": float(Decimal(ev["amount1"]) / _WETH_SCALE),
        "date": _ts_to_date(timestamp),
    }


def _extract_one(
    rpc: RpcClient,
    topic0: str,
    is_mint: bool,
    from_block: int,
    to_block: int,
    parts_dir: Path,
    checkpoint: Path,
    block_index: BlockIndex,
    workers: int = 1,
    chunk_blocks: int | None = None,
) -> int:
    def on_batch(logs: list[dict], lo: int, hi: int) -> None:
        if not logs:
            return
        ts_cache: dict[int, int] = {}
        for lg in logs:
            b = int(lg["blockNumber"], 16)
            if b not in ts_cache:
                ts_cache[b] = block_index._ts_of(b)  # noqa: SLF001
        rows = [
            _transform(lg, ts_cache[int(lg["blockNumber"], 16)], is_mint=is_mint) for lg in logs
        ]
        append_rows(parts_dir, rows, mint_burn_events_schema())
        log.info("%s %d..%d: %d rows", "mint" if is_mint else "burn", lo, hi, len(rows))

    if workers > 1:
        name = "mint_events" if is_mint else "burn_events"
        return fetch_logs_parallel(
            rpc,
            POOL_ADDRESS,
            [topic0],
            from_block,
            to_block,
            on_batch,
            progress_dir=checkpoint.parent / f"{name}_parallel",
            workers=workers,
            chunk_blocks=chunk_blocks,
        )
    return fetch_logs_chunked(
        rpc, POOL_ADDRESS, [topic0], from_block, to_block, on_batch, checkpoint_path=checkpoint
    )


def extract_mints_burns(
    rpc: RpcClient,
    from_block: int,
    to_block: int,
    out_dir: Path,
    *,
    workers: int = 1,
    chunk_blocks: int | None = None,
) -> int:
    parts_dir = out_dir / "raw" / "mint_burn_events_parts"
    block_index = BlockIndex(rpc, cache_path=out_dir / "checkpoints" / "block_index.sqlite")
    total = 0
    total += _extract_one(
        rpc, MINT_TOPIC0, True, from_block, to_block, parts_dir,
        out_dir / "checkpoints" / "mint_events.json", block_index, workers, chunk_blocks,
    )
    total += _extract_one(
        rpc, BURN_TOPIC0, False, from_block, to_block, parts_dir,
        out_dir / "checkpoints" / "burn_events.json", block_index, workers, chunk_blocks,
    )
    return total


def compact_mints_burns(out_dir: Path) -> int:
    return compact(
        out_dir / "raw" / "mint_burn_events_parts",
        out_dir / "processed" / "mint_burn_events.parquet",
        mint_burn_events_schema(),
    )


def extract_collects(
    rpc: RpcClient,
    from_block: int,
    to_block: int,
    out_dir: Path,
    *,
    workers: int = 1,
    chunk_blocks: int | None = None,
) -> int:
    parts_dir = out_dir / "raw" / "collect_events_parts"
    checkpoint = out_dir / "checkpoints" / "collect_events.json"
    block_index = BlockIndex(rpc, cache_path=out_dir / "checkpoints" / "block_index.sqlite")

    def on_batch(logs: list[dict], lo: int, hi: int) -> None:
        if not logs:
            return
        ts_cache: dict[int, int] = {}
        for lg in logs:
            block = int(lg["blockNumber"], 16)
            if block not in ts_cache:
                ts_cache[block] = block_index._ts_of(block)  # noqa: SLF001
        rows = [_transform_collect(lg, ts_cache[int(lg["blockNumber"], 16)]) for lg in logs]
        append_rows(parts_dir, rows, collect_events_schema())
        log.info("collects %d..%d: %d rows", lo, hi, len(rows))

    if workers > 1:
        return fetch_logs_parallel(
            rpc,
            POOL_ADDRESS,
            [COLLECT_TOPIC0],
            from_block,
            to_block,
            on_batch,
            progress_dir=checkpoint.parent / "collect_events_parallel",
            workers=workers,
            chunk_blocks=chunk_blocks,
        )
    return fetch_logs_chunked(
        rpc, POOL_ADDRESS, [COLLECT_TOPIC0], from_block, to_block, on_batch, checkpoint_path=checkpoint
    )


def compact_collects(out_dir: Path) -> int:
    return compact(
        out_dir / "raw" / "collect_events_parts",
        out_dir / "processed" / "collect_events.parquet",
        collect_events_schema(),
    )
