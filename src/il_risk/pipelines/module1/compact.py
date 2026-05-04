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
) -> pd.DataFrame:
    """Fetch ``slot0()`` at ``block_number - 1`` for each unique swap block.

    Task 3.4 defines the mid price as the pool price at the block before each
    observed swap.  ``swap_events.parquet`` only stores post-swap event prices,
    so this support extraction performs one archive ``eth_call`` per unique swap
    block and writes ``swap_mid_prices.parquet``.

    Timestamps are disabled by default because Task 3.4 only needs
    ``block_number`` and ``mid_price_usdc_per_weth``.  Enabling timestamps adds
    an extra ``eth_getBlockByNumber`` call per unique swap block.
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

    rows = []
    total = len(blocks)
    for i, block in enumerate(blocks, start=1):
        pre_swap_block = block - 1
        slot0 = fetch_slot0_at_block(rpc, pre_swap_block)
        row = {
            "block_number": block,
            "pre_swap_block": pre_swap_block,
            "sqrt_price_x96": Decimal(slot0["sqrt_price_x96"]),
            "mid_price_usdc_per_weth": _price_usdc_per_weth(slot0["sqrt_price_x96"]),
            "current_tick": slot0["tick"],
        }
        if include_timestamps:
            row["pre_swap_block_timestamp"] = datetime.fromtimestamp(
                int(rpc.get_block(pre_swap_block)["timestamp"], 16),
                tz=timezone.utc,
            )
        rows.append(row)
        if i == 1 or i % 500 == 0 or i == total:
            log.info("swap mid prices: fetched %d/%d unique blocks", i, total)

    out = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    return out


def fetch_slot0_at_block(rpc: RpcClient, block: int) -> dict:
    """Return pool ``slot0()`` decoded at a historical block."""

    raw = rpc.call(POOL_ADDRESS, _SLOT0_SELECTOR, block=block)
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
