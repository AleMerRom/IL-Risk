"""Generic Ethereum JSON-RPC client with multi-endpoint rate-limit cycling.

Vendor-neutral by design: speaks only standard ``eth_*`` methods so any
endpoint (paid, free, self-hosted, local Anvil fork) works.  No
``alchemy_*``, ``trace_*``, or provider-proprietary extensions.

Multi-endpoint failover
-----------------------
``RpcConfig`` accepts a primary ``url`` plus optional ``fallback_urls``. When
an endpoint returns HTTP 429, the client waits a fixed cooldown and retries at
a steadier pace. No exponential backoff is used.

Per-endpoint rate limiting (``rate_limit_rps``) throttles each URL
independently, so the effective throughput scales with pool size.

Cache
-----
Responses are keyed on ``(chain_id, method, block, target, calldata)`` and
stored in SQLite.  Only immutable results are cached (``eth_call`` at a
specific historic block, finalised ``eth_getBlockByNumber``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from il_risk.constants import MULTICALL3_ADDRESS

log = logging.getLogger(__name__)

BatchCallback = Callable[[list[dict], int, int], None]


# ------------------------------------------------------------------ exceptions


class RpcError(Exception):
    """Non-retryable JSON-RPC error from the endpoint."""


class RpcTransientError(Exception):
    """Retryable transport or server error (5xx, timeout, overloaded)."""


class RateLimitError(RpcTransientError):
    """HTTP 429 or rate-limit JSON error — triggers URL cycling."""


class TooManyResultsError(RpcError):
    """eth_getLogs exceeded the provider's result cap; caller should halve the range."""


# ------------------------------------------------------------------ data classes


@dataclass
class Call:
    target: str   # 0x-prefixed address
    data: bytes   # ABI-encoded calldata


@dataclass
class RpcConfig:
    url: str
    fallback_urls: list[str] = field(default_factory=list)
    log_url: str | None = None
    log_fallback_urls: list[str] = field(default_factory=list)
    archive_url: str | None = None
    max_logs_per_request: int = 10_000
    initial_log_chunk_blocks: int = 2_000
    rate_limit_rps: float | None = None
    rate_limit_cooldown: float = 5.0
    rate_limit_retries: int = 3
    retry_attempts: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 4.0
    request_headers: dict[str, str] = field(default_factory=dict)
    request_timeout_seconds: float = 30.0
    cache_path: Path | None = None  # defaults to data/checkpoints/rpc_cache.sqlite

    @classmethod
    def from_env(cls) -> RpcConfig:
        url = _validated_url(os.environ.get("RPC_URL"), "RPC_URL")
        if not url:
            raise RuntimeError("RPC_URL is not set — see .env.example")

        fallback_str = os.environ.get("RPC_FALLBACK_URLS", "")
        fallbacks = [
            _validated_url(u.strip(), "RPC_FALLBACK_URLS")
            for u in fallback_str.split(",")
            if u.strip()
        ]
        log_fallback_str = os.environ.get("LOG_RPC_FALLBACK_URLS", "")
        log_fallbacks = [
            _validated_url(u.strip(), "LOG_RPC_FALLBACK_URLS")
            for u in log_fallback_str.split(",")
            if u.strip()
        ]

        return cls(
            url=url,
            fallback_urls=fallbacks,
            log_url=_validated_url(os.environ.get("LOG_RPC_URL"), "LOG_RPC_URL"),
            log_fallback_urls=log_fallbacks,
            archive_url=_validated_url(os.environ.get("ARCHIVE_RPC_URL"), "ARCHIVE_RPC_URL"),
            max_logs_per_request=int(os.environ.get("RPC_MAX_LOGS_PER_REQUEST") or 10_000),
            initial_log_chunk_blocks=int(os.environ.get("RPC_INITIAL_LOG_CHUNK_BLOCKS") or 500),
            rate_limit_rps=float(os.environ["RPC_RATE_LIMIT_RPS"])
            if os.environ.get("RPC_RATE_LIMIT_RPS")
            else 3.0,
            rate_limit_cooldown=float(os.environ.get("RPC_RATE_LIMIT_COOLDOWN") or 5.0),
            rate_limit_retries=int(os.environ.get("RPC_RATE_LIMIT_RETRIES") or 3),
        )


# ------------------------------------------------------------------ internals


class _RpcCache:
    """SQLite-backed read-through cache for immutable RPC responses."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(key TEXT PRIMARY KEY, value BLOB NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def get(self, key: str) -> bytes | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM cache WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    def put(self, key: str, value: bytes) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value)
            )


class _RateLimiter:
    def __init__(self, rps: float | None):
        self._min_interval = 1.0 / rps if rps else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._min_interval == 0.0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
            self._next_allowed = max(now, self._next_allowed) + self._min_interval


class _UrlRotator:
    """Tracks endpoint health and cycles past rate-limited URLs with fixed cooldowns."""

    def __init__(self, urls: list[str], cooldown: float = 5.0):
        if not urls:
            raise ValueError("at least one URL required")
        self._urls = list(urls)
        self._cooldown = cooldown
        self._blocked_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def current(self) -> str:
        """Return the first available URL (not cooling-down)."""
        with self._lock:
            return self._pick_available()

    def _pick_available(self) -> str:
        now = time.monotonic()
        for url in self._urls:
            if self._blocked_until.get(url, 0.0) <= now:
                return url
        # All cooling-down — return soonest to recover.
        return min(self._urls, key=lambda u: self._blocked_until.get(u, 0.0))

    def mark_rate_limited(self, url: str) -> str:
        """Block *url* briefly; return the next available URL."""
        with self._lock:
            self._blocked_until[url] = time.monotonic() + self._cooldown
            log.debug("%s fixed cooldown %.1fs", _url_label(url), self._cooldown)
            return self._pick_available()

    def mark_success(self, url: str) -> None:
        """Reset back-off counter for a URL that just returned a good response."""
        with self._lock:
            self._blocked_until.pop(url, None)

    def seconds_until_any_available(self) -> float:
        """Seconds to sleep until at least one URL exits cooldown (0 if already available)."""
        with self._lock:
            now = time.monotonic()
            for url in self._urls:
                if self._blocked_until.get(url, 0.0) <= now:
                    return 0.0
            soonest = min(self._blocked_until.get(u, 0.0) for u in self._urls)
            return max(0.0, soonest - now)

    def all_urls(self) -> list[str]:
        return list(self._urls)


def _url_label(url: str) -> str:
    """Short human-readable label for a URL (hostname only)."""
    try:
        return urlparse(url).netloc
    except Exception:
        return url[:40]


def _validated_url(value: str | None, env_name: str) -> str | None:
    """Return a cleaned HTTP(S) URL or raise a helpful config error."""
    if value is None:
        return None
    url = value.strip()
    if not url:
        return None
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RuntimeError(
            f"{env_name} must be a full http(s) URL, got {url!r}. "
            "Check .env for placeholders or missing https://."
        )
    return url


def _hex_to_int(h: str) -> int:
    return int(h, 16) if isinstance(h, str) else h


def _int_to_hex(i: int) -> str:
    return hex(i)


def _block_to_rpc(block: int | str) -> str:
    if isinstance(block, int):
        return _int_to_hex(block)
    if block in ("latest", "earliest", "pending", "safe", "finalized"):
        return block
    raise ValueError(f"bad block reference: {block!r}")


def _validate_log_response(result: Any, url: str) -> None:
    """Reject malformed provider log payloads before ETL writes them to disk."""
    if not isinstance(result, list):
        raise RpcTransientError(f"malformed eth_getLogs result from {_url_label(url)}")
    for log_item in result:
        if not isinstance(log_item, dict):
            raise RpcTransientError(f"malformed eth_getLogs entry from {_url_label(url)}")
        log_index = log_item.get("logIndex")
        if not isinstance(log_index, str):
            raise RpcTransientError(f"missing logIndex in eth_getLogs response from {_url_label(url)}")
        try:
            value = int(log_index, 16)
        except ValueError as exc:
            raise RpcTransientError(
                f"bad logIndex {log_index!r} in eth_getLogs response from {_url_label(url)}"
            ) from exc
        # A mainnet block cannot contain anywhere near this many logs. Values
        # like 0xfffffffc are provider-corrupt sentinels, not Ethereum log ids.
        if value > 1_000_000:
            raise RpcTransientError(
                f"implausible logIndex {log_index} from {_url_label(url)}"
            )


# ------------------------------------------------------------------ client


class RpcClient:
    """Standards-only JSON-RPC client with transparent multi-endpoint failover."""

    def __init__(self, config: RpcConfig, *, data_dir: Path | None = None):
        self._cfg = config
        all_urls = [config.url] + list(config.fallback_urls)
        log_urls = [config.log_url or config.url] + list(
            config.log_fallback_urls or config.fallback_urls
        )
        self._rotator = _UrlRotator(
            all_urls,
            cooldown=config.rate_limit_cooldown,
        )
        self._log_rotator = _UrlRotator(log_urls, cooldown=config.rate_limit_cooldown)

        # One rate-limiter per URL so each endpoint's quota is respected
        # independently — cycling to a fresh URL gets a fresh token bucket.
        limiter_urls = all_urls + log_urls
        if config.archive_url:
            limiter_urls.append(config.archive_url)
        self._limiters: dict[str, _RateLimiter] = {
            url: _RateLimiter(config.rate_limit_rps) for url in sorted(set(limiter_urls))
        }

        self._session = requests.Session()
        self._session.headers["content-type"] = "application/json"
        if config.request_headers:
            self._session.headers.update(config.request_headers)

        # Archive session shares headers but uses a dedicated connection pool.
        self._archive_session = requests.Session()
        self._archive_session.headers["content-type"] = "application/json"
        if config.request_headers:
            self._archive_session.headers.update(config.request_headers)

        cache_path = config.cache_path or (
            (data_dir or Path("data")) / "checkpoints" / "rpc_cache.sqlite"
        )
        self._cache = _RpcCache(cache_path)
        self._chain_id: int | None = None
        self._supports_archive: bool | None = None

    # ------------------------------------------------------------------ transport

    def _post_one(
        self,
        url: str,
        session: requests.Session,
        payload: dict[str, Any] | list[dict[str, Any]],
    ) -> Any:
        """Single HTTP POST to *url*.

        Retries on generic transient errors (5xx, timeout) with fixed
        backoff via tenacity.  Does NOT retry on ``RateLimitError`` — that is
        handled by the caller which cycles to a different endpoint.
        """
        limiter = self._limiters.get(url) or _RateLimiter(None)
        limiter.wait()

        @retry(
            retry=(
                retry_if_exception_type(RpcTransientError)
                & retry_if_not_exception_type(RateLimitError)
            ),
            wait=wait_fixed(self._cfg.retry_min_wait),
            stop=stop_after_attempt(self._cfg.retry_attempts),
            reraise=True,
        )
        def _do() -> Any:
            try:
                resp = session.post(
                    url, json=payload, timeout=self._cfg.request_timeout_seconds
                )
            except requests.RequestException as exc:
                raise RpcTransientError(f"transport error: {exc}") from exc
            if resp.status_code == 429:
                raise RateLimitError(f"HTTP 429 from {_url_label(url)}: {resp.text[:200]}")
            if resp.status_code in (401, 403):
                # Auth/access denial on this endpoint — cycle to next provider.
                raise RateLimitError(
                    f"HTTP {resp.status_code} from {_url_label(url)}: {resp.text[:200]}"
                )
            if 500 <= resp.status_code < 600:
                # Transient server error — tenacity will retry on the same URL
                # briefly before we consider cycling away.
                raise RpcTransientError(
                    f"HTTP {resp.status_code} from {_url_label(url)}: {resp.text[:200]}"
                )
            if resp.status_code == 400:
                body = resp.text[:400]
                if "up to a 10 block range" in body.lower():
                    raise TooManyResultsError(body)
                # Provider capacity limits (e.g. Alchemy free-tier block-range
                # restriction) — treat as a cycle signal so the rotator moves
                # to the next endpoint rather than hard-failing.
                _cap_hints = ("block range", "free tier", "your plan", "upgrade", "compute unit")
                if any(h in body.lower() for h in _cap_hints):
                    raise RateLimitError(f"provider capacity limit on {_url_label(url)}: {body[:200]}")
                raise RpcError(f"HTTP 400 from {_url_label(url)}: {body}")
            if resp.status_code >= 400:
                raise RpcError(
                    f"HTTP {resp.status_code} from {_url_label(url)}: {resp.text[:200]}"
                )
            try:
                return resp.json()
            except ValueError as exc:
                raise RpcTransientError(
                    f"invalid JSON from {_url_label(url)}: {resp.text[:200]}"
                ) from exc

        return _do()

    def _parse_rpc_result(self, data: Any, url: str) -> Any:
        """Parse JSON-RPC envelope; raise typed exceptions for error responses."""
        if not isinstance(data, dict):
            raise RpcTransientError(f"malformed JSON-RPC response from {_url_label(url)}")
        if "error" not in data:
            if "result" not in data:
                raise RpcTransientError(f"missing JSON-RPC result from {_url_label(url)}")
            return data["result"]
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        low = msg.lower()
        if any(
            h in low
            for h in (
                "too many requests",
                "rate limit",
                "request limit",
                "compute units",
                "daily limit",
                "quota",
                "throttle",
            )
        ):
            raise RateLimitError(f"{_url_label(url)}: {msg}")
        if any(h in low for h in ("query returned more than", "response size",
                                   "block range", "range too large", "exceeds maximum")):
            raise TooManyResultsError(msg)
        transient_hints = (
            "timeout",
            "temporarily",
            "temporary",
            "overloaded",
            "upstream",
            "unreachable",
            "connection",
            "bad gateway",
            "gateway timeout",
            "service unavailable",
            "try again",
            "cannot parse json-rpc response",
        )
        if any(h in low for h in transient_hints):
            raise RpcTransientError(msg)
        raise RpcError(msg)

    def _rpc(self, method: str, params: list[Any], *, archive: bool = False, logs: bool = False) -> Any:
        """Execute a JSON-RPC call, cycling endpoints on rate-limit errors.

        Archive calls always use ``config.archive_url`` if set (never rotated).
        Non-archive calls rotate through the URL pool on ``RateLimitError``.
        If every URL is cooling-down, sleeps until the soonest one recovers.
        """
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

        if archive and self._cfg.archive_url:
            for attempt in range(1, self._cfg.retry_attempts + 1):
                try:
                    data = self._post_one(self._cfg.archive_url, self._archive_session, payload)
                    return self._parse_rpc_result(data, self._cfg.archive_url)
                except RateLimitError:
                    if attempt >= self._cfg.retry_attempts:
                        raise
                    log.warning(
                        "archive endpoint rate-limited; sleeping %.1fs before retry %d/%d",
                        self._cfg.rate_limit_cooldown,
                        attempt + 1,
                        self._cfg.retry_attempts,
                    )
                    time.sleep(self._cfg.rate_limit_cooldown)
                except RpcTransientError as exc:
                    if attempt >= self._cfg.retry_attempts:
                        raise
                    log.warning(
                        "archive endpoint transient error (%s); retry %d/%d after %.1fs",
                        exc,
                        attempt + 1,
                        self._cfg.retry_attempts,
                        self._cfg.retry_min_wait,
                    )
                    time.sleep(self._cfg.retry_min_wait)

        rotator = self._log_rotator if logs else self._rotator
        tried_urls: set[str] = set()
        url = rotator.current()
        rate_limit_attempts = 0

        while True:
            try:
                data = self._post_one(url, self._session, payload)
                result = self._parse_rpc_result(data, url)
                if logs and method == "eth_getLogs":
                    _validate_log_response(result, url)
                rotator.mark_success(url)
                return result
            except RateLimitError as exc:
                rate_limit_attempts += 1
                log.warning(
                    "rate-limited on %s (%s); fixed cooldown attempt %d/%d",
                    _url_label(url),
                    exc,
                    rate_limit_attempts,
                    self._cfg.rate_limit_retries,
                )
                if rate_limit_attempts >= self._cfg.rate_limit_retries:
                    raise
                tried_urls.add(url)
                next_url = rotator.mark_rate_limited(url)
                url = self._handle_exhausted(rotator, tried_urls, next_url, exc)
            except RpcTransientError as exc:
                # Tenacity retries on same URL exhausted; cycle to next endpoint.
                log.warning("transient error on %s after retries — cycling", _url_label(url))
                tried_urls.add(url)
                next_url = rotator.mark_rate_limited(url)
                url = self._handle_exhausted(rotator, tried_urls, next_url, exc)

    def _rpc_batch(
        self,
        method: str,
        params_list: list[list[Any]],
        *,
        archive: bool = False,
    ) -> list[Any]:
        """Execute a JSON-RPC batch."""
        if not params_list:
            return []

        payload = [
            {"jsonrpc": "2.0", "id": idx, "method": method, "params": params}
            for idx, params in enumerate(params_list)
        ]

        if archive and self._cfg.archive_url:
            for attempt in range(1, self._cfg.retry_attempts + 1):
                try:
                    data = self._post_one(self._cfg.archive_url, self._archive_session, payload)
                    if not isinstance(data, list):
                        raise RpcTransientError(
                            f"malformed JSON-RPC batch from {_url_label(self._cfg.archive_url)}"
                        )
                    by_id = {item.get("id"): item for item in data if isinstance(item, dict)}
                    if len(by_id) != len(payload):
                        raise RpcTransientError(
                            f"incomplete JSON-RPC batch from {_url_label(self._cfg.archive_url)}"
                        )
                    return [
                        self._parse_rpc_result(by_id[idx], self._cfg.archive_url)
                        for idx in range(len(payload))
                    ]
                except RateLimitError:
                    if attempt >= self._cfg.retry_attempts:
                        raise
                    log.warning(
                        "archive endpoint rate-limited on batch; sleeping %.1fs before retry %d/%d",
                        self._cfg.rate_limit_cooldown,
                        attempt + 1,
                        self._cfg.retry_attempts,
                    )
                    time.sleep(self._cfg.rate_limit_cooldown)
                except RpcTransientError as exc:
                    if attempt >= self._cfg.retry_attempts:
                        raise
                    log.warning(
                        "archive endpoint transient batch error (%s); retry %d/%d after %.1fs",
                        exc,
                        attempt + 1,
                        self._cfg.retry_attempts,
                        self._cfg.retry_min_wait,
                    )
                    time.sleep(self._cfg.retry_min_wait)

        rotator = self._rotator
        tried_urls: set[str] = set()
        url = rotator.current()
        rate_limit_attempts = 0

        while True:
            try:
                data = self._post_one(url, self._session, payload)
                if not isinstance(data, list):
                    raise RpcTransientError(f"malformed JSON-RPC batch from {_url_label(url)}")
                by_id = {item.get("id"): item for item in data if isinstance(item, dict)}
                if len(by_id) != len(payload):
                    raise RpcTransientError(f"incomplete JSON-RPC batch from {_url_label(url)}")
                result = [self._parse_rpc_result(by_id[idx], url) for idx in range(len(payload))]
                rotator.mark_success(url)
                return result
            except RateLimitError as exc:
                rate_limit_attempts += 1
                log.warning(
                    "rate-limited on %s batch (%s); fixed cooldown attempt %d/%d",
                    _url_label(url),
                    exc,
                    rate_limit_attempts,
                    self._cfg.rate_limit_retries,
                )
                if rate_limit_attempts >= self._cfg.rate_limit_retries:
                    raise
                tried_urls.add(url)
                next_url = rotator.mark_rate_limited(url)
                url = self._handle_exhausted(rotator, tried_urls, next_url, exc)
            except RpcTransientError as exc:
                log.warning("transient batch error on %s after retries — cycling", _url_label(url))
                tried_urls.add(url)
                next_url = rotator.mark_rate_limited(url)
                url = self._handle_exhausted(rotator, tried_urls, next_url, exc)

    def _handle_exhausted(
        self, rotator: _UrlRotator, tried_urls: set[str], next_url: str, exc: Exception
    ) -> str:
        """Handle the case where all URLs in the current cycle have been tried."""
        if next_url not in tried_urls:
            return next_url
        # All endpoints failed in this cycle.
        wait_t = rotator.seconds_until_any_available()
        if wait_t <= 0:
            # Zero cooldown (test mode) or already recovered — raise to avoid busy loop.
            raise exc
        log.warning(
            "all %d RPC endpoint(s) unavailable; sleeping %.1fs",
            len(rotator.all_urls()),
            wait_t,
        )
        time.sleep(wait_t)
        tried_urls.clear()
        return rotator.current()

    # ------------------------------------------------------------------ public methods

    def chain_id(self) -> int:
        if self._chain_id is None:
            if self._cfg.archive_url:
                self._chain_id = _hex_to_int(self._rpc("eth_chainId", [], archive=True))
            else:
                self._chain_id = _hex_to_int(self._rpc("eth_chainId", []))
        return self._chain_id

    def get_block_number(self) -> int:
        return _hex_to_int(self._rpc("eth_blockNumber", []))

    def get_block(self, number: int | str, *, full_transactions: bool = False) -> dict:
        block_ref = _block_to_rpc(number)
        cacheable = isinstance(number, int)
        key = (
            self._cache_key("eth_getBlockByNumber", block_ref, "", full_transactions)
            if cacheable
            else None
        )
        if key is not None:
            hit = self._cache.get(key)
            if hit is not None:
                return json.loads(hit)
        result = self._rpc("eth_getBlockByNumber", [block_ref, full_transactions])
        if key is not None and result is not None:
            self._cache.put(key, json.dumps(result).encode())
        return result

    def get_blocks(
        self,
        numbers: Iterable[int],
        *,
        full_transactions: bool = False,
        batch_size: int = 100,
    ) -> dict[int, dict]:
        """Fetch many blocks, using the cache and JSON-RPC batch requests."""
        unique = sorted(set(numbers))
        out: dict[int, dict] = {}
        missing: list[int] = []
        for number in unique:
            block_ref = _block_to_rpc(number)
            key = self._cache_key("eth_getBlockByNumber", block_ref, "", full_transactions)
            hit = self._cache.get(key)
            if hit is None:
                missing.append(number)
            else:
                out[number] = json.loads(hit)

        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            params = [[_block_to_rpc(number), full_transactions] for number in batch]
            results = self._rpc_batch("eth_getBlockByNumber", params)
            for number, result in zip(batch, results, strict=True):
                if result is None:
                    raise RpcTransientError(f"missing block {number}")
                out[number] = result
                key = self._cache_key(
                    "eth_getBlockByNumber",
                    _block_to_rpc(number),
                    "",
                    full_transactions,
                )
                self._cache.put(key, json.dumps(result).encode())
        return out

    def call(self, to: str, data: bytes, block: int | str = "latest") -> bytes:
        block_ref = _block_to_rpc(block)
        is_historical = isinstance(block, int)
        key = (
            self._cache_key("eth_call", block_ref, to.lower(), data.hex())
            if is_historical
            else None
        )
        if key is not None:
            hit = self._cache.get(key)
            if hit is not None:
                return hit
        params = [{"to": to, "data": "0x" + data.hex()}, block_ref]
        result_hex: str = self._rpc("eth_call", params, archive=is_historical)
        result = bytes.fromhex(result_hex[2:])
        if key is not None:
            self._cache.put(key, result)
        return result

    def call_many(
        self,
        calls: Iterable[tuple[str, bytes, int | str]],
        *,
        batch_size: int = 100,
    ) -> list[bytes]:
        """Execute many ``eth_call`` requests, preserving input order."""

        call_list = list(calls)
        out: list[bytes | None] = [None] * len(call_list)
        missing: list[tuple[int, str, bytes, int | str, str | None]] = []
        for idx, (to, data, block) in enumerate(call_list):
            block_ref = _block_to_rpc(block)
            key = (
                self._cache_key("eth_call", block_ref, to.lower(), data.hex())
                if isinstance(block, int)
                else None
            )
            if key is not None:
                hit = self._cache.get(key)
                if hit is not None:
                    out[idx] = hit
                    continue
            missing.append((idx, to, data, block, key))

        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            params = [
                [{"to": to, "data": "0x" + data.hex()}, _block_to_rpc(block)]
                for _idx, to, data, block, _key in batch
            ]
            archive = all(isinstance(block, int) for _idx, _to, _data, block, _key in batch)
            result_hexes = self._rpc_batch("eth_call", params, archive=archive)
            for (idx, _to, _data, _block, key), result_hex in zip(batch, result_hexes, strict=True):
                result = bytes.fromhex(result_hex[2:])
                out[idx] = result
                if key is not None:
                    self._cache.put(key, result)

        final: list[bytes] = []
        for idx, result in enumerate(out):
            if result is None:
                raise RpcTransientError(f"missing eth_call result at batch position {idx}")
            final.append(result)
        return final

    def get_logs(
        self,
        address: str,
        topics: list[str | None | list[str]],
        from_block: int,
        to_block: int,
    ) -> list[dict]:
        # Use archive=False so eth_getLogs cycles through all URLs in the
        # rotation.  Historical logs are available on any full node; routing
        # them to the fixed archive URL would lock onto Alchemy free-tier
        # which caps eth_getLogs at 10 blocks per request.
        params = [
            {
                "address": address,
                "topics": topics,
                "fromBlock": _int_to_hex(from_block),
                "toBlock": _int_to_hex(to_block),
            }
        ]
        return self._rpc("eth_getLogs", params, archive=False, logs=True)

    def multicall(
        self,
        calls: list[Call],
        *,
        block: int | str = "latest",
        allow_failure: bool = False,
    ) -> list[bytes]:
        """Aggregate3 via Multicall3.  Standard ``eth_call``, works on any endpoint."""
        selector = bytes.fromhex("82ad56cb")  # aggregate3((address,bool,bytes)[])
        tuples = [(c.target, allow_failure, c.data) for c in calls]
        encoded_args = abi_encode(["(address,bool,bytes)[]"], [tuples])
        data = selector + encoded_args
        result = self.call(MULTICALL3_ADDRESS, data, block=block)
        (decoded,) = abi_decode(["(bool,bytes)[]"], result)
        out: list[bytes] = []
        for success, ret in decoded:
            if not success and not allow_failure:
                raise RpcError("multicall call failed")
            out.append(ret)
        return out

    # ------------------------------------------------------------------ introspection

    @property
    def supports_archive(self) -> bool:
        """True if the active endpoint can serve ``eth_call`` at historical blocks."""
        if self._supports_archive is None:
            self._supports_archive = self._probe_archive()
        return self._supports_archive

    @property
    def has_archive_url(self) -> bool:
        """True when a dedicated archive endpoint is configured."""
        return bool(self._cfg.archive_url)

    def _probe_archive(self) -> bool:
        try:
            head = self.get_block_number()
        except RpcError:
            return False
        probe_block = max(1, head - 1_000)
        try:
            self.call(
                MULTICALL3_ADDRESS,
                bytes.fromhex("3408e470"),  # getChainId()
                block=probe_block,
            )
            return True
        except (RpcError, RpcTransientError) as exc:
            log.info("archive probe failed (%s) — Path B event-replay will be used", exc)
            return False

    # ------------------------------------------------------------------ helpers

    def _cache_key(self, method: str, block: str, target: str, payload: Any) -> str:
        chain_id = self.chain_id()
        raw = f"{chain_id}|{method}|{block}|{target}|{payload}".encode()
        return hashlib.sha256(raw).hexdigest()


def _read_checkpoint(path: Path) -> int | None:
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("last_block_completed")


def _write_checkpoint(path: Path, last_block: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"last_block_completed": last_block}))
    tmp.replace(path)


def fetch_logs_chunked(
    rpc: RpcClient,
    address: str,
    topics: Iterable[str | None | list[str]],
    from_block: int,
    to_block: int,
    on_batch: BatchCallback,
    *,
    checkpoint_path: Path | None = None,
    initial_chunk: int | None = None,
    min_chunk: int = 1,
) -> int:
    """Fetch logs over a range with adaptive chunks and checkpointed progress."""

    topics_list = list(topics)
    initial = initial_chunk or rpc._cfg.initial_log_chunk_blocks  # noqa: SLF001
    chunk = initial
    ceiling = initial
    resume_from = _read_checkpoint(checkpoint_path) if checkpoint_path else None
    start = max(from_block, (resume_from + 1) if resume_from is not None else from_block)
    total_blocks = to_block - from_block + 1
    started_at = time.monotonic()
    total = 0
    cursor = start
    if start > to_block:
        log.info("logs already complete for %d..%d", from_block, to_block)
        return 0
    log.info(
        "log scan starting: blocks %d..%d, resume at %d, chunk=%d",
        from_block,
        to_block,
        start,
        chunk,
    )
    while cursor <= to_block:
        window_end = min(cursor + chunk - 1, to_block)
        try:
            logs = rpc.get_logs(address, topics_list, cursor, window_end)
        except RateLimitError as exc:
            old_chunk = chunk
            if chunk > min_chunk:
                chunk = max(min_chunk, chunk // 2)
                ceiling = min(ceiling, chunk)
            log.warning(
                "provider throttled block window %d..%d (%s); chunk %d -> %d; sleeping %.1fs",
                cursor,
                window_end,
                exc,
                old_chunk,
                chunk,
                rpc._cfg.rate_limit_cooldown,  # noqa: SLF001
            )
            time.sleep(rpc._cfg.rate_limit_cooldown)  # noqa: SLF001
            continue
        except TooManyResultsError:
            if chunk <= min_chunk:
                raise
            old_chunk = chunk
            chunk = max(min_chunk, chunk // 2)
            ceiling = chunk
            log.info(
                "provider returned too many logs for %d..%d; chunk %d -> %d and retrying",
                cursor,
                window_end,
                old_chunk,
                chunk,
            )
            continue
        except RpcTransientError as exc:
            log.warning(
                "transient RPC failure for window %d..%d (%s); sleeping %.1fs",
                cursor,
                window_end,
                exc,
                rpc._cfg.rate_limit_cooldown,  # noqa: SLF001
            )
            time.sleep(rpc._cfg.rate_limit_cooldown)  # noqa: SLF001
            continue
        on_batch(logs, cursor, window_end)
        total += len(logs)
        if checkpoint_path is not None:
            _write_checkpoint(checkpoint_path, window_end)
        blocks_done = window_end - from_block + 1
        pct = min(100.0, blocks_done / total_blocks * 100)
        elapsed = max(0.001, time.monotonic() - started_at)
        blocks_per_sec = max(0.001, (window_end - start + 1) / elapsed)
        remaining_blocks = max(0, to_block - window_end)
        eta_seconds = int(remaining_blocks / blocks_per_sec)
        log.info(
            "progress blocks %d/%d (%.2f%%), window %d..%d, logs this=%d, logs total=%d, chunk=%d, eta=%s",
            blocks_done,
            total_blocks,
            pct,
            cursor,
            window_end,
            len(logs),
            total,
            chunk,
            _format_eta(eta_seconds),
        )
        cursor = window_end + 1
        if chunk < ceiling:
            chunk = min(chunk * 2, ceiling)
    return total


def fetch_logs_parallel(
    rpc: RpcClient,
    address: str,
    topics: Iterable[str | None | list[str]],
    from_block: int,
    to_block: int,
    on_batch: BatchCallback,
    *,
    progress_dir: Path,
    workers: int = 4,
    chunk_blocks: int | None = None,
    min_chunk: int = 10,
) -> int:
    """Fetch logs with independent block chunks and resumable completion markers."""

    topics_list = list(topics)
    chunk = chunk_blocks or rpc._cfg.initial_log_chunk_blocks  # noqa: SLF001
    progress_dir.mkdir(parents=True, exist_ok=True)
    total_blocks = to_block - from_block + 1
    started_at = time.monotonic()
    total_logs = 0
    completed_blocks = 0
    completed_chunks = 0

    def marker(lo: int, hi: int) -> Path:
        return progress_dir / f"done-{lo}-{hi}.json"

    def make_chunks(lo: int, hi: int, size: int) -> list[tuple[int, int]]:
        return [(start, min(start + size - 1, hi)) for start in range(lo, hi + 1, size)]

    pending_chunks = [c for c in make_chunks(from_block, to_block, chunk) if not marker(*c).exists()]
    skipped_blocks = total_blocks - sum(hi - lo + 1 for lo, hi in pending_chunks)
    completed_blocks += skipped_blocks
    log.info(
        "parallel log scan starting: blocks %d..%d, chunk=%d, workers=%d, pending_chunks=%d, skipped_blocks=%d",
        from_block,
        to_block,
        chunk,
        workers,
        len(pending_chunks),
        skipped_blocks,
    )

    def fetch_window(lo: int, hi: int) -> tuple[int, int, list[dict]]:
        return lo, hi, rpc.get_logs(address, topics_list, lo, hi)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}

        def submit_more() -> None:
            while pending_chunks and len(futures) < workers:
                lo, hi = pending_chunks.pop(0)
                futures[pool.submit(fetch_window, lo, hi)] = (lo, hi)

        submit_more()
        while futures:
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                lo, hi = futures.pop(fut)
                try:
                    _lo, _hi, logs = fut.result()
                except TooManyResultsError:
                    if hi - lo + 1 <= min_chunk:
                        raise
                    mid = (lo + hi) // 2
                    pending_chunks.insert(0, (mid + 1, hi))
                    pending_chunks.insert(0, (lo, mid))
                    log.info("split busy block window %d..%d into %d..%d and %d..%d", lo, hi, lo, mid, mid + 1, hi)
                    continue
                except RateLimitError as exc:
                    pending_chunks.append((lo, hi))
                    log.warning(
                        "throttled window %d..%d (%s); requeue after %.1fs",
                        lo,
                        hi,
                        exc,
                        rpc._cfg.rate_limit_cooldown,  # noqa: SLF001
                    )
                    time.sleep(rpc._cfg.rate_limit_cooldown)  # noqa: SLF001
                    continue
                except RpcTransientError as exc:
                    pending_chunks.append((lo, hi))
                    log.warning(
                        "transient RPC failure for window %d..%d (%s); requeue after %.1fs",
                        lo,
                        hi,
                        exc,
                        rpc._cfg.rate_limit_cooldown,  # noqa: SLF001
                    )
                    time.sleep(rpc._cfg.rate_limit_cooldown)  # noqa: SLF001
                    continue

                on_batch(logs, lo, hi)
                marker(lo, hi).write_text(json.dumps({"from_block": lo, "to_block": hi, "logs": len(logs)}))
                total_logs += len(logs)
                completed_chunks += 1
                completed_blocks += hi - lo + 1
                elapsed = max(0.001, time.monotonic() - started_at)
                blocks_per_sec = max(0.001, (completed_blocks - skipped_blocks) / elapsed)
                remaining_blocks = max(0, total_blocks - completed_blocks)
                log.info(
                    "progress blocks %d/%d (%.2f%%), window %d..%d, logs this=%d, logs total=%d, chunks done=%d, eta=%s",
                    completed_blocks,
                    total_blocks,
                    min(100.0, completed_blocks / total_blocks * 100),
                    lo,
                    hi,
                    len(logs),
                    total_logs,
                    completed_chunks,
                    _format_eta(int(remaining_blocks / blocks_per_sec)),
                )
            submit_more()
    return total_logs


def _format_eta(seconds: int) -> str:
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"
