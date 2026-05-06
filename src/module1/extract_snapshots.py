"""Module 1 — daily slot0 and liquidity-map snapshots.

Two reconstruction paths that produce the same parquet schema:

- **Path A (archive):** scan ``tickBitmap`` words over the full range at the
  daily block, enumerate initialized ticks, then batch ``ticks(tick)`` via
  Multicall3 to get ``liquidityNet``, ``liquidityGross``, and both
  ``feeGrowthOutside_*_X128`` values.
- **Path B (event-replay):** walk Mint/Burn events up to the daily block and
  maintain a per-tick running sum of liquidityNet / liquidityGross. No fee-
  growth information is produced here — those fields are null in Path B rows.

Selection is automatic based on ``RpcClient.supports_archive``; callers can
override by calling ``snapshot_path_a`` / ``snapshot_path_b`` directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timezone, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pyarrow.parquet as pq
from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from eth_utils import keccak

from shared.constants import POOL_ADDRESS, TICK_SPACING
from shared.constants import TOKEN0_DECIMALS, TOKEN1_DECIMALS
from shared.uniswap_math import MAX_TICK, MIN_TICK
from shared.block_index import BlockIndex
from shared.schemas import append_rows, compact, liquidity_snapshot_schema, slot0_schema
from shared.rpc import Call, RpcClient

log = logging.getLogger(__name__)

_TICK_BITMAP_SELECTOR = keccak(text="tickBitmap(int16)")[:4]
_TICKS_SELECTOR = keccak(text="ticks(int24)")[:4]
_SLOT0_SELECTOR = keccak(text="slot0()")[:4]
_FGG0_SELECTOR = keccak(text="feeGrowthGlobal0X128()")[:4]
_FGG1_SELECTOR = keccak(text="feeGrowthGlobal1X128()")[:4]

# The full valid word-position range for this pool's tick spacing.
_MIN_WORD = (MIN_TICK // TICK_SPACING) >> 8
_MAX_WORD = (MAX_TICK // TICK_SPACING) >> 8

_MULTICALL_BATCH_BITMAP = 100
_MULTICALL_BATCH_TICKS = 100
_TOKEN_SCALE = Decimal(10) ** Decimal(TOKEN1_DECIMALS - TOKEN0_DECIMALS)
_Q96 = Decimal(2) ** 96
_USDC_SCALE = Decimal(10) ** TOKEN0_DECIMALS
_WETH_SCALE = Decimal(10) ** TOKEN1_DECIMALS


@dataclass
class TickDetails:
    tick: int
    liquidity_gross: int
    liquidity_net: int
    fee_growth_outside_0_x128: int
    fee_growth_outside_1_x128: int


# ---------------------------------------------------------------- Path A: archive

def _encode_bitmap_call(word_pos: int) -> bytes:
    return _TICK_BITMAP_SELECTOR + abi_encode(["int16"], [word_pos])


def _encode_ticks_call(tick: int) -> bytes:
    return _TICKS_SELECTOR + abi_encode(["int24"], [tick])


def _decode_bitmap(raw: bytes) -> int:
    (word,) = abi_decode(["uint256"], raw)
    return word


def _decode_tick_details(raw: bytes, tick: int) -> TickDetails:
    (
        liquidity_gross,
        liquidity_net,
        fg0,
        fg1,
        _tick_cum_outside,
        _secs_per_liq_outside,
        _secs_outside,
        _initialized,
    ) = abi_decode(
        ["uint128", "int128", "uint256", "uint256", "int56", "uint160", "uint32", "bool"],
        raw,
    )
    return TickDetails(
        tick=tick,
        liquidity_gross=liquidity_gross,
        liquidity_net=liquidity_net,
        fee_growth_outside_0_x128=fg0,
        fee_growth_outside_1_x128=fg1,
    )


def _enumerate_initialized_ticks(rpc: RpcClient, block: int) -> list[int]:
    """Scan every word in the valid range; return the ticks with a set bit."""
    words = list(range(_MIN_WORD, _MAX_WORD + 1))
    ticks: list[int] = []
    for i in range(0, len(words), _MULTICALL_BATCH_BITMAP):
        batch = words[i : i + _MULTICALL_BATCH_BITMAP]
        calls = [Call(target=POOL_ADDRESS, data=_encode_bitmap_call(w)) for w in batch]
        results = rpc.multicall(calls, block=block)
        for word_pos, raw in zip(batch, results, strict=True):
            word = _decode_bitmap(raw)
            if word == 0:
                continue
            for bit in range(256):
                if word & (1 << bit):
                    compressed_tick = (word_pos << 8) | bit
                    # Handle two's complement for negative wordPos (int16 was encoded signed above,
                    # but compressed_tick derived from (word_pos << 8 | bit) is already correct
                    # because word_pos retains its sign in Python int).
                    ticks.append(compressed_tick * TICK_SPACING)
    return sorted(ticks)


def _fetch_tick_details(rpc: RpcClient, block: int, ticks: list[int]) -> list[TickDetails]:
    out: list[TickDetails] = []
    for i in range(0, len(ticks), _MULTICALL_BATCH_TICKS):
        batch = ticks[i : i + _MULTICALL_BATCH_TICKS]
        calls = [Call(target=POOL_ADDRESS, data=_encode_ticks_call(t)) for t in batch]
        results = rpc.multicall(calls, block=block)
        out.extend(_decode_tick_details(r, t) for t, r in zip(batch, results, strict=True))
    return out


def snapshot_path_a(rpc: RpcClient, block: int) -> list[TickDetails]:
    ticks = _enumerate_initialized_ticks(rpc, block)
    log.info("path-A: %d initialized ticks at block %d", len(ticks), block)
    return _fetch_tick_details(rpc, block, ticks)


# ---------------------------------------------------------------- Path B: event-replay

def snapshot_path_b(mint_burn_parquet: Path, snapshot_block: int) -> list[TickDetails]:
    """Replay the ``mint_burn_events`` parquet up to ``snapshot_block`` inclusive."""
    table = pq.read_table(
        mint_burn_parquet,
        columns=["block_number", "event_type", "tick_lower", "tick_upper", "liquidity_delta"],
        filters=[("block_number", "<=", snapshot_block)],
    )
    positions: dict[int, dict[str, int]] = {}
    for row in table.to_pylist():
        delta = int(row["liquidity_delta"])
        lower, upper = row["tick_lower"], row["tick_upper"]
        for t, net_sign in ((lower, +1), (upper, -1)):
            bucket = positions.setdefault(t, {"net": 0, "gross": 0})
            bucket["net"] += net_sign * delta
            bucket["gross"] += abs(delta) if row["event_type"] == "mint" else -abs(delta)
    return [
        TickDetails(
            tick=t,
            liquidity_gross=b["gross"],
            liquidity_net=b["net"],
            fee_growth_outside_0_x128=0,
            fee_growth_outside_1_x128=0,
        )
        for t, b in sorted(positions.items())
        if b["gross"] != 0 or b["net"] != 0
    ]


# ---------------------------------------------------------------- snapshot writer

def _snapshot_ts(day: date) -> int:
    """The PDF asks for the block closest to 00:00 UTC for each snapshot day."""
    return int(datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc).timestamp())


def _price_usdc_per_weth_at_tick(tick: int) -> float:
    return float(_TOKEN_SCALE / (Decimal("1.0001") ** Decimal(tick)))


def _price_weth_per_usdc(sqrt_price_x96: int) -> float:
    price_raw = (Decimal(sqrt_price_x96) / _Q96) ** 2
    return float(price_raw * (_USDC_SCALE / _WETH_SCALE))


def _price_usdc_per_weth(sqrt_price_x96: int) -> float:
    return 1.0 / _price_weth_per_usdc(sqrt_price_x96)


def _fetch_slot0(rpc: RpcClient, block: int) -> dict:
    results = rpc.multicall(
        [
            Call(POOL_ADDRESS, _SLOT0_SELECTOR),
            Call(POOL_ADDRESS, _FGG0_SELECTOR),
            Call(POOL_ADDRESS, _FGG1_SELECTOR),
        ],
        block=block,
    )
    (
        sqrt_price_x96,
        tick,
        obs_idx,
        obs_card,
        _obs_card_next,
        fee_protocol,
        unlocked,
    ) = abi_decode(["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"], results[0])
    (fgg0,) = abi_decode(["uint256"], results[1])
    (fgg1,) = abi_decode(["uint256"], results[2])
    return {
        "sqrt_price_x96": sqrt_price_x96,
        "tick": tick,
        "observation_index": obs_idx,
        "observation_cardinality": obs_card,
        "fee_protocol": fee_protocol,
        "unlocked": unlocked,
        "fee_growth_global_0_x128": fgg0,
        "fee_growth_global_1_x128": fgg1,
    }


def extract_slot0_daily(rpc: RpcClient, start: date, end: date, out_dir: Path) -> int:
    block_index = BlockIndex(rpc, cache_path=out_dir / "checkpoints" / "block_index.sqlite")
    parts_dir = out_dir / "raw" / "slot0_snapshots_parts"
    rows: list[dict] = []
    day = start
    while day <= end:
        block = block_index.closest_block_at_timestamp(_snapshot_ts(day))
        ts = block_index._ts_of(block)  # noqa: SLF001
        slot0 = _fetch_slot0(rpc, block)
        rows.append(
            {
                "date": day.isoformat(),
                "snapshot_block": block,
                "snapshot_timestamp": datetime.fromtimestamp(ts, tz=timezone.utc),
                "sqrt_price_x96": Decimal(slot0["sqrt_price_x96"]),
                "price_usdc_per_weth": _price_usdc_per_weth(slot0["sqrt_price_x96"]),
                "current_tick": slot0["tick"],
                "observation_index": slot0["observation_index"],
                "observation_cardinality": slot0["observation_cardinality"],
                "fee_protocol": slot0["fee_protocol"],
                "unlocked": slot0["unlocked"],
                "fee_growth_global_0_x128": Decimal(slot0["fee_growth_global_0_x128"]),
                "fee_growth_global_1_x128": Decimal(slot0["fee_growth_global_1_x128"]),
            }
        )
        log.info("slot0 %s @ block %d: tick=%d", day.isoformat(), block, slot0["tick"])
        day += timedelta(days=1)
    append_rows(parts_dir, rows, slot0_schema())
    return len(rows)


def compact_slot0(out_dir: Path) -> int:
    return compact(
        out_dir / "raw" / "slot0_snapshots_parts",
        out_dir / "processed" / "slot0_snapshots.parquet",
        slot0_schema(),
    )


def extract_liquidity_snapshots_daily(
    rpc: RpcClient,
    start: date,
    end: date,
    out_dir: Path,
    *,
    force_path_b: bool = False,
    mint_burn_parquet: Path | None = None,
) -> int:
    parts_dir = out_dir / "raw" / "liquidity_snapshots_parts"

    use_path_b = force_path_b or not (rpc.has_archive_url or rpc.supports_archive)
    slot0_schedule = _load_slot0_snapshot_schedule(out_dir / "processed" / "slot0_snapshots.parquet")
    if slot0_schedule:
        log.info("using %d slot0 snapshot blocks for liquidity snapshots", len(slot0_schedule))
    if use_path_b:
        if mint_burn_parquet is None:
            mint_burn_parquet = out_dir / "processed" / "mint_burn_events.parquet"
        if not mint_burn_parquet.exists():
            raise FileNotFoundError(
                f"Path B requires {mint_burn_parquet} — run `python scripts/data_extraction.py extract mints-burns` first"
            )

    block_index = None
    if slot0_schedule is None:
        block_index = BlockIndex(rpc, cache_path=out_dir / "checkpoints" / "block_index.sqlite")

    day = start
    total_rows = 0
    while day <= end:
        if slot0_schedule is not None:
            if day not in slot0_schedule:
                raise KeyError(f"{day.isoformat()} missing from slot0 snapshot schedule")
            block, snapshot_timestamp = slot0_schedule[day]
        else:
            assert block_index is not None
            block = block_index.closest_block_at_timestamp(_snapshot_ts(day))
            ts = block_index._ts_of(block)  # noqa: SLF001
            snapshot_timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
        if use_path_b:
            details = snapshot_path_b(mint_burn_parquet, block)  # type: ignore[arg-type]
        else:
            details = snapshot_path_a(rpc, block)

        rows = []
        active_liquidity = 0
        for d in sorted(details, key=lambda x: x.tick):
            active_liquidity += d.liquidity_net
            rows.append({
                "date": day.isoformat(),
                "snapshot_block": block,
                "snapshot_timestamp": snapshot_timestamp,
                "tick": d.tick,
                "liquidityNet": Decimal(d.liquidity_net),
                "liquidityGross": Decimal(d.liquidity_gross),
                "active_liquidity": Decimal(active_liquidity),
                "price_lower": _price_usdc_per_weth_at_tick(d.tick),
                "price_upper": _price_usdc_per_weth_at_tick(d.tick + TICK_SPACING),
                "fee_growth_outside_0_x128": Decimal(d.fee_growth_outside_0_x128),
                "fee_growth_outside_1_x128": Decimal(d.fee_growth_outside_1_x128),
            })
        append_rows(parts_dir, rows, liquidity_snapshot_schema())
        total_rows += len(rows)
        log.info("liquidity %s @ block %d: %d ticks", day.isoformat(), block, len(rows))
        day += timedelta(days=1)
    return total_rows


def _load_slot0_snapshot_schedule(path: Path) -> dict[date, tuple[int, datetime]] | None:
    """Load the validated slot0 daily block schedule, if available.

    Event-replay liquidity snapshots should use the exact same snapshot blocks
    as ``slot0_snapshots.parquet``. That avoids a second RPC-dependent midnight
    block lookup and prevents tiny block mismatches between the two outputs.
    """
    if not path.exists():
        return None
    table = pq.read_table(path, columns=["date", "snapshot_block", "snapshot_timestamp"])
    schedule: dict[date, tuple[int, datetime]] = {}
    for row in table.to_pylist():
        day = date.fromisoformat(row["date"])
        ts = row["snapshot_timestamp"]
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        schedule[day] = (int(row["snapshot_block"]), ts)
    return schedule


def compact_liquidity_snapshots(out_dir: Path) -> int:
    return compact(
        out_dir / "raw" / "liquidity_snapshots_parts",
        out_dir / "processed" / "liquidity_snapshots.parquet",
        liquidity_snapshot_schema(),
    )
