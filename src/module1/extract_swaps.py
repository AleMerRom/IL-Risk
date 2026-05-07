"""Module 1 — Swap event ETL to ``swap_events.parquet``."""

from __future__ import annotations

import logging
from datetime import timezone, datetime
from decimal import Decimal
from pathlib import Path

from eth_abi import decode as abi_decode

from shared.constants import POOL_ADDRESS, TOKEN0_DECIMALS, TOKEN1_DECIMALS
from shared.uniswap_events import SWAP_TOPIC0
from shared.block_index import BlockIndex
from shared.rpc import fetch_logs_chunked, fetch_logs_parallel
from shared.schemas import append_rows, compact, swap_events_schema, write_rows
from shared.rpc import RpcClient

log = logging.getLogger(__name__)

_USDC_SCALE = Decimal(10) ** TOKEN0_DECIMALS
_WETH_SCALE = Decimal(10) ** TOKEN1_DECIMALS
_Q96 = Decimal(2) ** 96


def _decode_swap(log_dict: dict) -> dict:
    topics = log_dict["topics"]
    sender = "0x" + topics[1][-40:]
    recipient = "0x" + topics[2][-40:]
    data = bytes.fromhex(log_dict["data"][2:])
    amount0, amount1, sqrt_price_x96, liquidity, tick = abi_decode(
        ["int256", "int256", "uint160", "uint128", "int24"], data
    )
    return {
        "sender": sender,
        "recipient": recipient,
        "amount0": amount0,
        "amount1": amount1,
        "sqrt_price_x96": sqrt_price_x96,
        "liquidity": liquidity,
        "tick": tick,
    }


def _price_weth_per_usdc(sqrt_price_x96: int) -> float:
    # sqrtPriceX96 = sqrt(token1/token0) * 2^96 in raw units.
    # price_raw = token1/token0 = WETH_raw / USDC_raw.
    # human WETH per USDC = price_raw * 10^(USDC_dec - WETH_dec)
    price_raw = (Decimal(sqrt_price_x96) / _Q96) ** 2
    return float(price_raw * (_USDC_SCALE / _WETH_SCALE))


def _price_usdc_per_weth(sqrt_price_x96: int) -> float:
    return 1.0 / _price_weth_per_usdc(sqrt_price_x96)


def _transform(log_dict: dict, timestamp: int) -> dict:
    ev = _decode_swap(log_dict)
    price_usdc_per_weth = _price_usdc_per_weth(ev["sqrt_price_x96"])
    amount0_usdc = float(Decimal(ev["amount0"]) / _USDC_SCALE)
    amount1_weth = float(Decimal(ev["amount1"]) / _WETH_SCALE)
    block = int(log_dict["blockNumber"], 16)
    date_str = _ts_to_date(timestamp)
    return {
        "block_number": block,
        "block_timestamp": datetime.fromtimestamp(timestamp, tz=timezone.utc),
        "tx_hash": log_dict["transactionHash"],
        "log_index": int(log_dict["logIndex"], 16),
        "sender": ev["sender"],
        "recipient": ev["recipient"],
        "amount0_raw": Decimal(ev["amount0"]),
        "amount1_raw": Decimal(ev["amount1"]),
        "amount0_usdc": amount0_usdc,
        "amount1_weth": amount1_weth,
        "sqrt_price_x96": Decimal(ev["sqrt_price_x96"]),
        "price_usdc_per_weth": price_usdc_per_weth,
        "active_liquidity": Decimal(ev["liquidity"]),
        "tick": ev["tick"],
        "trade_direction": "buy_weth" if ev["amount0"] > 0 else "sell_weth",
        "usd_notional": abs(amount0_usdc),
        "date": date_str,
    }


def _ts_to_date(ts: int) -> str:
    from datetime import timezone, datetime

    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def extract_swaps(
    rpc: RpcClient,
    from_block: int,
    to_block: int,
    *,
    out_dir: Path,
    parts_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    workers: int = 1,
    chunk_blocks: int | None = None,
) -> int:
    """Fetch all Swap events in range and write to partitioned parquet parts."""
    parts_dir = parts_dir or out_dir / "raw" / "swap_events_parts"
    checkpoint = checkpoint_path or out_dir / "checkpoints" / "swap_events.json"
    block_index = BlockIndex(rpc, cache_path=out_dir / "checkpoints" / "block_index.sqlite")

    def on_batch(logs: list[dict], lo: int, hi: int) -> None:
        if not logs:
            return
        # Resolve timestamps once per unique block.
        blocks = {int(lg["blockNumber"], 16) for lg in logs}
        ts_by_block = block_index.ts_many(blocks)
        rows = [_transform(lg, ts_by_block[int(lg["blockNumber"], 16)]) for lg in logs]
        if workers > 1:
            write_rows(parts_dir / f"part-{lo}-{hi}.parquet", rows, swap_events_schema())
        else:
            append_rows(parts_dir, rows, swap_events_schema())
        log.info("swaps %d..%d: %d rows", lo, hi, len(rows))

    if workers > 1:
        return fetch_logs_parallel(
            rpc,
            POOL_ADDRESS,
            [SWAP_TOPIC0],
            from_block,
            to_block,
            on_batch,
            progress_dir=out_dir / "checkpoints" / "swap_events_parallel",
            workers=workers,
            chunk_blocks=chunk_blocks,
        )
    return fetch_logs_chunked(
        rpc,
        POOL_ADDRESS,
        [SWAP_TOPIC0],
        from_block,
        to_block,
        on_batch,
        checkpoint_path=checkpoint,
        initial_chunk=chunk_blocks,
    )


def compact_swaps(out_dir: Path) -> int:
    return compact(
        out_dir / "raw" / "swap_events_parts",
        out_dir / "processed" / "swap_events.parquet",
        swap_events_schema(),
    )
