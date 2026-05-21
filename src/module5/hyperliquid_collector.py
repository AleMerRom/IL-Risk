"""Module 5 Task 5.2 - Hyperliquid ETH perp market-data collection.

The project asks for hourly ETH perpetual candles and hourly funding rates
for the full study window (2025-10-01 to 2026-03-31, ~4 368 hourly rows).

Hyperliquid's public ``POST https://api.hyperliquid.xyz/info`` endpoint
requires no authentication.  The ``candleSnapshot`` type returns up to 5 000
candles per call, but the rolling retention window can make early study-period
rows unavailable.  The official path intentionally fails when Hyperliquid
does not provide full hourly coverage; it does not substitute another venue's
spot or perp data.
The ``fundingHistory`` type returns up to 500 records per call; the
collector paginates automatically.

``fundingHistory`` does not expose an oracle price field, so the official path
cannot produce the complete funding P&L input required by Task 5.2.  It fails
rather than filling oracle prices with candle closes.

Set ``QUICKNODE_SQL_API_KEY`` to use QuickNode SQL Explorer, or set one of
``ALLIUM_API_KEY`` for Allium Explorer, or set one of ``ZEROXARCHIVE_API_KEY`` /
``0XARCHIVE_API_KEY`` / ``OXARCHIVE_API_KEY`` / ``OXA_API_KEY`` to use the
0xArchive Python SDK (``pip install oxarchive``) which provides native
Hyperliquid candles and hourly oracle/mark prices. Free-tier keys are available
at https://0xarchive.io.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import logging
import os
from pathlib import Path
import time as time_module
from typing import Any, Literal

try:
    import oxarchive as _oxarchive
except ImportError:  # pragma: no cover
    _oxarchive = None  # type: ignore[assignment]

import pandas as pd
import requests
from dotenv import load_dotenv
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)
import typer

DEFAULT_START = "2025-10-01"
DEFAULT_END = "2026-03-31"
DEFAULT_COIN = "ETH"
DEFAULT_INTERVAL = "1h"
DEFAULT_RESULTS_DIR = Path("data/results/module_5")
DEFAULT_HYPERLIQUID_API_URL = "https://api.hyperliquid.xyz"
QUICKNODE_SQL_URL = "https://api.quicknode.com/sql/rest/v1/query"
QUICKNODE_CLUSTER_ID = "hyperliquid-core-mainnet"
ALLIUM_API_URL = "https://api.allium.so/api/v1"
REQUEST_TIMEOUT_SECONDS = 30.0
MAX_OFFICIAL_PAGE_ROWS = 500

MarketDataSource = Literal["auto", "allium", "quicknode", "0xarchive", "official"]

log = logging.getLogger(__name__)
app = typer.Typer(no_args_is_help=True, add_completion=False)


class HyperliquidTransientError(RuntimeError):
    """Retryable Hyperliquid/SQL transport or server error."""


@dataclass(frozen=True)
class CollectionWindow:
    start_date: date
    end_date: date
    start_ms: int
    end_ms: int


class HyperliquidInfoClient:
    """Small client for Hyperliquid's public ``POST /info`` API."""

    def __init__(
        self,
        *,
        api_url: str = DEFAULT_HYPERLIQUID_API_URL,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    @retry(
        retry=retry_if_exception_type(HyperliquidTransientError),
        wait=wait_fixed(1),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def post_info(self, payload: dict[str, Any]) -> Any:
        try:
            response = self.session.post(
                f"{self.api_url}/info",
                json=payload,
                timeout=self.timeout_seconds,
                headers={"Content-Type": "application/json"},
            )
        except requests.RequestException as exc:
            raise HyperliquidTransientError(str(exc)) from exc

        if response.status_code == 429 or 500 <= response.status_code < 600:
            raise HyperliquidTransientError(
                f"Hyperliquid /info returned HTTP {response.status_code}: {response.text[:200]}"
            )
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as exc:
            raise HyperliquidTransientError("Hyperliquid /info returned invalid JSON") from exc


class QuickNodeSqlClient:
    """REST client for QuickNode SQL Explorer."""

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str = QUICKNODE_SQL_URL,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.endpoint = endpoint
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    @retry(
        retry=retry_if_exception_type(HyperliquidTransientError),
        wait=wait_fixed(2),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def query(self, sql: str) -> list[dict[str, Any]]:
        try:
            response = self.session.post(
                self.endpoint,
                json={"query": sql, "clusterId": QUICKNODE_CLUSTER_ID},
                timeout=self.timeout_seconds,
                headers={
                    "accept": "application/json",
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                },
            )
        except requests.RequestException as exc:
            raise HyperliquidTransientError(str(exc)) from exc

        if response.status_code == 429 or 500 <= response.status_code < 600:
            raise HyperliquidTransientError(
                f"QuickNode SQL returned HTTP {response.status_code}: {response.text[:300]}"
            )
        if response.status_code >= 400:
            raise RuntimeError(
                f"QuickNode SQL returned HTTP {response.status_code}: {response.text[:300]}. "
                "The configured API key may not include SQL Explorer access on the current plan."
            )
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise RuntimeError(f"QuickNode SQL error: {payload['error']}")
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"QuickNode SQL response missing data list: {payload}")
        return data


class AlliumExplorerClient:
    """REST client for Allium Explorer saved-query execution."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str = ALLIUM_API_URL,
        timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()

    def query(self, sql: str, *, title: str, limit: int = 250_000) -> list[dict[str, Any]]:
        query_id = self._create_query(sql, title=title, limit=limit)
        run_id = self._run_query(query_id)
        self._wait_for_success(run_id)
        return self._fetch_results(run_id)

    def _headers(self) -> dict[str, str]:
        return {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        for attempt in range(8):
            try:
                response = self.session.request(
                    method,
                    url,
                    timeout=self.timeout_seconds,
                    **kwargs,
                )
            except requests.RequestException as exc:
                if attempt == 7:
                    raise HyperliquidTransientError(str(exc)) from exc
                time_module.sleep(2**attempt)
                continue

            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == 7:
                    raise RuntimeError(
                        f"Allium API returned HTTP {response.status_code}: {response.text[:300]}"
                    )
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2**attempt, 60)
                time_module.sleep(delay)
                continue
            return response
        raise RuntimeError("unreachable Allium retry state")

    def _create_query(self, sql: str, *, title: str, limit: int) -> str:
        response = self._request(
            "POST",
            f"{self.api_url}/explorer/queries",
            headers=self._headers(),
            json={"title": title, "config": {"sql": sql, "limit": limit}},
        )
        response.raise_for_status()
        payload = response.json()
        query_id = payload.get("query_id") or payload.get("id")
        if not query_id:
            raise RuntimeError(f"Allium create query response missing query_id: {payload}")
        return str(query_id)

    def _run_query(self, query_id: str) -> str:
        response = self._request(
            "POST",
            f"{self.api_url}/explorer/queries/{query_id}/run-async",
            headers=self._headers(),
            json={"parameters": {}, "run_config": {}},
        )
        response.raise_for_status()
        payload = response.json()
        run_id = payload.get("run_id")
        if not run_id:
            raise RuntimeError(f"Allium run query response missing run_id: {payload}")
        return str(run_id)

    def _wait_for_success(self, run_id: str, *, timeout_seconds: int = 600) -> None:
        deadline = time_module.monotonic() + timeout_seconds
        while time_module.monotonic() < deadline:
            response = self._request(
                "GET",
                f"{self.api_url}/explorer/query-runs/{run_id}/status",
                headers={"X-API-KEY": self.api_key},
            )
            response.raise_for_status()
            payload = response.json()
            status = payload if isinstance(payload, str) else payload.get("status")
            if status == "success":
                return
            if status in {"failed", "canceled"}:
                raise RuntimeError(f"Allium query {run_id} ended with status={status}")
            time_module.sleep(3)
        raise TimeoutError(f"Allium query {run_id} did not complete within {timeout_seconds}s")

    def _fetch_results(self, run_id: str) -> list[dict[str, Any]]:
        response = self._request(
            "GET",
            f"{self.api_url}/explorer/query-runs/{run_id}/results",
            headers={"X-API-KEY": self.api_key},
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"Allium query results missing data list: {payload}")
        return data


def _make_oxarchive_client(api_key: str) -> Any:
    if _oxarchive is None:
        raise RuntimeError(
            "oxarchive package is required for source='0xarchive'. "
            "Install it with: pip install oxarchive"
        )
    return _oxarchive.Client(api_key=api_key)


def _collection_window(from_date: str, to_date: str) -> CollectionWindow:
    start_day = _parse_date(from_date)
    end_day = _parse_date(to_date)
    if end_day < start_day:
        raise ValueError("--to must be on or after --from")
    start_dt = datetime.combine(start_day, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(end_day, time.max, tzinfo=timezone.utc)
    return CollectionWindow(
        start_date=start_day,
        end_date=end_day,
        start_ms=_to_ms(start_dt),
        end_ms=_to_ms(end_dt),
    )


def _monthly_ms_windows(window: CollectionWindow) -> list[tuple[int, int]]:
    """Return month-bounded UTC windows, each small enough for 1000 hourly rows."""

    out: list[tuple[int, int]] = []
    start = datetime.fromtimestamp(window.start_ms / 1000, tz=timezone.utc)
    final = datetime.fromtimestamp(window.end_ms / 1000, tz=timezone.utc)
    current = start
    while current <= final:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        end = min(next_month - timedelta(milliseconds=1), final)
        out.append((_to_ms(current), _to_ms(end)))
        current = next_month
    return out


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _to_utc_timestamp(value: Any) -> pd.Series:
    s = pd.to_datetime(value, unit="ms", utc=True)
    # Normalise to microsecond resolution so merge keys match regardless of
    # whether the other side was parsed from ISO strings (us) or ms integers.
    return s.astype("datetime64[us, UTC]")


def _ts_from_ms(value: int) -> pd.Timestamp:
    return pd.Timestamp(value, unit="ms", tz="UTC")


def _sql_dt(value_ms: int) -> str:
    return _ts_from_ms(value_ms).strftime("%Y-%m-%d %H:%M:%S")


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("wrote %s (%d rows)", path, len(df))


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _setup_logging() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _zeroxarchive_api_key_from_env() -> str | None:
    return (
        os.environ.get("ZEROXARCHIVE_API_KEY")
        or os.environ.get("0XARCHIVE_API_KEY")
        or os.environ.get("ZEROARCHIVE_API_KEY")
        or os.environ.get("OXARCHIVE_API_KEY")
        or os.environ.get("OXA_API_KEY")
    )


def _parse_source(value: str) -> MarketDataSource:
    if value not in {"auto", "allium", "quicknode", "0xarchive", "official"}:
        raise typer.BadParameter("source must be one of: auto, allium, quicknode, 0xarchive, official")
    return value  # type: ignore[return-value]


if __name__ == "__main__":
    app()
