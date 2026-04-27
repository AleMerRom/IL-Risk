"""Timestamp ↔ block-number resolver with SQLite-backed memoization."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from il_risk.rpc import RpcClient


class BlockIndex:
    def __init__(self, rpc: RpcClient, cache_path: Path | None = None):
        self._rpc = rpc
        self._cache_path = cache_path or Path("data/checkpoints/block_index.sqlite")
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS block_ts "
                "(block INTEGER PRIMARY KEY, timestamp INTEGER NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON block_ts(timestamp)")

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self._cache_path, timeout=30.0)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _ts_of(self, block: int) -> int:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT timestamp FROM block_ts WHERE block = ?", (block,)
            ).fetchone()
        if row is not None:
            return row[0]
        blk = self._rpc.get_block(block)
        ts = int(blk["timestamp"], 16) if isinstance(blk["timestamp"], str) else int(blk["timestamp"])
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO block_ts(block, timestamp) VALUES (?, ?)", (block, ts)
            )
        return ts

    def block_at_timestamp(self, target_ts: int) -> int:
        """Largest block number with ``timestamp <= target_ts``."""
        lo = 1
        hi = self._rpc.get_block_number()
        if target_ts <= self._ts_of(lo):
            return lo
        if target_ts >= self._ts_of(hi):
            return hi
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._ts_of(mid) <= target_ts:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def first_block_at_or_after(self, target_ts: int) -> int:
        """Smallest block number with ``timestamp >= target_ts``."""
        block = self.block_at_timestamp(target_ts)
        if self._ts_of(block) >= target_ts:
            return block
        return block + 1

    def closest_block_at_timestamp(self, target_ts: int) -> int:
        """Block whose timestamp is closest to ``target_ts``."""
        before = self.block_at_timestamp(target_ts)
        after = before + 1
        head = self._rpc.get_block_number()
        if after > head:
            return before
        before_delta = abs(self._ts_of(before) - target_ts)
        after_delta = abs(self._ts_of(after) - target_ts)
        return after if after_delta < before_delta else before
