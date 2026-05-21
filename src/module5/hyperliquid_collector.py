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


def collect_perp_prices(
    *,
    coin: str = DEFAULT_COIN,
    interval: str = DEFAULT_INTERVAL,
    from_date: str = DEFAULT_START,
    to_date: str = DEFAULT_END,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    source: MarketDataSource = "auto",
    api_url: str = DEFAULT_HYPERLIQUID_API_URL,
    allium_api_key: str | None = None,
    quicknode_api_key: str | None = None,
    zeroxarchive_api_key: str | None = None,
) -> pd.DataFrame:
    """Collect hourly OHLCV candles and write ``perp_prices.parquet``."""

    window = _collection_window(from_date, to_date)
    if interval != "1h":
        raise ValueError("Task 5.2 requires hourly candles; only interval='1h' is supported")

    allium_api_key = allium_api_key or os.environ.get("ALLIUM_API_KEY")
    quicknode_api_key = quicknode_api_key or os.environ.get("QUICKNODE_SQL_API_KEY")
    zeroxarchive_api_key = zeroxarchive_api_key or _zeroxarchive_api_key_from_env()
    # Allium asset contexts are reliable for funding/oracle joins but can have
    # sparse hourly price-context coverage. In auto mode, prefer candle-native
    # price sources and reserve Allium prices for explicit source='allium'.
    use_allium = source == "allium"
    use_zeroxarchive = source == "0xarchive" or (
        source == "auto" and not use_allium and zeroxarchive_api_key
    )
    use_quicknode = source == "quicknode" or (
        source == "auto" and not use_allium and not use_zeroxarchive and quicknode_api_key
    )

    if use_allium:
        if not allium_api_key:
            raise RuntimeError("source='allium' requires ALLIUM_API_KEY")
        df = _collect_prices_allium(
            AlliumExplorerClient(api_key=allium_api_key),
            coin=coin,
            window=window,
        )
        source_name = "allium_explorer"
    elif use_quicknode:
        if not quicknode_api_key:
            raise RuntimeError("source='quicknode' requires QUICKNODE_SQL_API_KEY")
        df = _collect_prices_quicknode(
            QuickNodeSqlClient(api_key=quicknode_api_key),
            coin=coin,
            window=window,
        )
        source_name = "quicknode_sql"
    elif use_zeroxarchive:
        if not zeroxarchive_api_key:
            raise RuntimeError("source='0xarchive' requires ZEROXARCHIVE_API_KEY")
        df = _collect_prices_zeroxarchive(
            _make_oxarchive_client(zeroxarchive_api_key),
            coin=coin,
            interval=interval,
            window=window,
        )
        source_name = "0xarchive"
    else:
        df = _collect_prices_official(
            HyperliquidInfoClient(api_url=api_url),
            coin=coin,
            interval=interval,
            window=window,
        )
        source_name = "hyperliquid_info"

    if "source" not in df.columns:
        df["source"] = source_name
    df = _normalise_prices(df, coin=coin, interval=interval, window=window)
    _validate_hourly_coverage(df, window, time_column="open_time", label="perp_prices")
    path = results_dir / "perp_prices.parquet"
    _write_parquet(df, path)
    _write_metadata(
        path.with_suffix(".metadata.json"),
        {
            "coin": coin,
            "interval": interval,
            "from": from_date,
            "to": to_date,
            "source": source_name,
            "rows": int(len(df)),
        },
    )
    return df


def collect_funding_rates(
    *,
    coin: str = DEFAULT_COIN,
    from_date: str = DEFAULT_START,
    to_date: str = DEFAULT_END,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    source: MarketDataSource = "auto",
    api_url: str = DEFAULT_HYPERLIQUID_API_URL,
    allium_api_key: str | None = None,
    quicknode_api_key: str | None = None,
    zeroxarchive_api_key: str | None = None,
) -> pd.DataFrame:
    """Collect hourly funding rates and write ``funding_rates.parquet``.

    The official Hyperliquid ``fundingHistory`` endpoint does not include
    oracle prices.  Use ``source='quicknode'`` when Task 5.2 requires the
    native oracle price for funding P&L.
    """

    window = _collection_window(from_date, to_date)
    allium_api_key = allium_api_key or os.environ.get("ALLIUM_API_KEY")
    quicknode_api_key = quicknode_api_key or os.environ.get("QUICKNODE_SQL_API_KEY")
    zeroxarchive_api_key = zeroxarchive_api_key or _zeroxarchive_api_key_from_env()
    use_hybrid = source == "auto" and allium_api_key and zeroxarchive_api_key
    use_allium = source == "allium" or (source == "auto" and allium_api_key and not use_hybrid)
    use_quicknode = source == "quicknode" or (
        source == "auto" and not use_hybrid and not use_allium and quicknode_api_key
    )
    use_zeroxarchive = source == "0xarchive" or (
        source == "auto" and not use_hybrid and not use_allium and not use_quicknode and zeroxarchive_api_key
    )

    if use_hybrid:
        df = _collect_funding_zeroxarchive_allium(
            zeroxarchive_client=_make_oxarchive_client(zeroxarchive_api_key),
            allium_client=AlliumExplorerClient(api_key=allium_api_key),
            coin=coin,
            window=window,
        )
        source_name = "0xarchive_allium"
    elif use_allium:
        if not allium_api_key:
            raise RuntimeError("source='allium' requires ALLIUM_API_KEY")
        df = _collect_funding_allium(
            AlliumExplorerClient(api_key=allium_api_key),
            coin=coin,
            window=window,
        )
        source_name = "allium_explorer"
    elif use_quicknode:
        if not quicknode_api_key:
            raise RuntimeError("source='quicknode' requires QUICKNODE_SQL_API_KEY")
        df = _collect_funding_quicknode(
            QuickNodeSqlClient(api_key=quicknode_api_key),
            coin=coin,
            window=window,
        )
        source_name = "quicknode_sql"
    elif use_zeroxarchive:
        if not zeroxarchive_api_key:
            raise RuntimeError("source='0xarchive' requires ZEROXARCHIVE_API_KEY")
        df = _collect_funding_zeroxarchive(
            _make_oxarchive_client(zeroxarchive_api_key),
            coin=coin,
            window=window,
        )
        source_name = "0xarchive"
    else:
        df = _collect_funding_official(
            HyperliquidInfoClient(api_url=api_url),
            coin=coin,
            window=window,
        )
        source_name = "hyperliquid_info"

    df["source"] = source_name
    df = _normalise_funding(df, coin=coin, window=window)

    _validate_hourly_coverage(df, window, time_column="funding_time", label="funding_rates")
    _validate_oracle_coverage(df, source_name=source_name)
    path = results_dir / "funding_rates.parquet"
    _write_parquet(df, path)
    oracle_source = (
        source_name
        if use_hybrid or use_allium or use_quicknode or use_zeroxarchive
        else "unavailable"
    )
    _write_metadata(
        path.with_suffix(".metadata.json"),
        {
            "coin": coin,
            "from": from_date,
            "to": to_date,
            "source": source_name,
            "oracle_price_source": oracle_source,
            "rows": int(len(df)),
            "sign_convention": (
                "funding_pnl_short = position_size_ETH * oracle_price * funding_rate; "
                "positive funding_rate means short receives funding"
            ),
        },
    )
    return df


def _collect_prices_official(
    client: HyperliquidInfoClient,
    *,
    coin: str,
    interval: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    payload = {
        "type": "candleSnapshot",
        "req": {
            "coin": coin,
            "interval": interval,
            "startTime": window.start_ms,
            "endTime": window.end_ms,
        },
    }
    rows = client.post_info(payload)
    if not isinstance(rows, list):
        raise RuntimeError(f"unexpected candleSnapshot response: {rows}")
    return pd.DataFrame(rows)


def _collect_funding_official(
    client: HyperliquidInfoClient,
    *,
    coin: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    start_ms = window.start_ms
    while start_ms <= window.end_ms:
        payload = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": start_ms,
            "endTime": window.end_ms,
        }
        page = client.post_info(payload)
        if not isinstance(page, list):
            raise RuntimeError(f"unexpected fundingHistory response: {page}")
        if not page:
            break
        rows.extend(page)
        last_time = int(page[-1]["time"])
        next_start = last_time + 1
        if next_start <= start_ms:
            raise RuntimeError("fundingHistory pagination did not advance")
        start_ms = next_start
        if len(page) < MAX_OFFICIAL_PAGE_ROWS:
            break
    return pd.DataFrame(rows)


def _collect_prices_quicknode(
    client: QuickNodeSqlClient,
    *,
    coin: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start_ms, end_ms in _monthly_ms_windows(window):
        sql = f"""
        SELECT
            coin,
            hour,
            toFloat64(open) AS open,
            toFloat64(high) AS high,
            toFloat64(low) AS low,
            toFloat64(close) AS close,
            toFloat64(volume) AS volume,
            toUInt64(trade_count) AS num_trades
        FROM hyperliquid_market_volume_hourly
        WHERE coin = {_sql_string(coin)}
          AND hour >= toDateTime({_sql_string(_sql_dt(start_ms))}, 'UTC')
          AND hour <= toDateTime({_sql_string(_sql_dt(end_ms))}, 'UTC')
        ORDER BY hour ASC
        """
        rows.extend(client.query(sql))
    return pd.DataFrame(rows)


def _collect_prices_allium(
    client: AlliumExplorerClient,
    *,
    coin: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for start_ms, end_ms in _monthly_ms_windows(window):
        sql = f"""
        SELECT
            coin,
            date_trunc('hour', timestamp) AS hour,
            argMin(mid_price, timestamp) AS open,
            max(mid_price) AS high,
            min(mid_price) AS low,
            argMax(mid_price, timestamp) AS close,
            0 AS volume,
            0 AS num_trades
        FROM hyperliquid.raw.perpetual_market_asset_contexts
        WHERE coin = {_sql_string(coin)}
          AND timestamp >= {_sql_string(_sql_dt(start_ms))}
          AND timestamp <= {_sql_string(_sql_dt(end_ms))}
          AND mid_price IS NOT NULL
        GROUP BY coin, hour
        ORDER BY hour ASC
        """
        rows.extend(
            client.query(
                sql,
                title=f"IL Risk {coin} Allium hourly prices {_sql_dt(start_ms)}",
            )
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["source"] = "allium_explorer"
    return df


def _collect_prices_zeroxarchive(
    client: Any,
    *,
    coin: str,
    interval: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    start_str = _ts_from_ms(window.start_ms).strftime("%Y-%m-%dT%H:%M:%S")
    # Add 1 h so the SDK's exclusive end includes the final candle of the window.
    end_str = (_ts_from_ms(window.end_ms) + pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
    candles = []
    cursor = None
    while True:
        kwargs: dict[str, Any] = {"start": start_str, "end": end_str, "interval": interval}
        if cursor is not None:
            kwargs["cursor"] = cursor
        resp = client.hyperliquid.candles.history(coin, **kwargs)
        candles.extend(resp.data)
        if resp.next_cursor is None:
            break
        cursor = resp.next_cursor

    if not candles:
        return pd.DataFrame()

    rows = [
        {
            "coin": coin,
            "hour": c.timestamp,
            "open": float(c.open),
            "high": float(c.high),
            "low": float(c.low),
            "close": float(c.close),
            "volume": float(c.volume),
            "num_trades": int(c.trade_count),
            "source": "0xarchive",
        }
        for c in candles
    ]
    return pd.DataFrame(rows)


def _collect_funding_quicknode(
    client: QuickNodeSqlClient,
    *,
    coin: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    # The official funding endpoint gives the settled hourly funding record.
    # QuickNode is used here to attach the corresponding hourly oracle price,
    # which Task 5.2 needs for short-position funding P&L.
    funding = _collect_funding_official(
        HyperliquidInfoClient(),
        coin=coin,
        window=window,
    )
    oracle_rows: list[dict[str, Any]] = []
    for start_ms, end_ms in _monthly_ms_windows(window):
        sql = f"""
        SELECT
            coin,
            toStartOfHour(polled_at) AS hour,
            avg(toFloat64(oracle_px)) AS oracle_price,
            avg(toFloat64(mark_px)) AS mark_price,
            avg(toFloat64(mid_px)) AS mid_price
        FROM hyperliquid_perpetual_market_contexts
        WHERE coin = {_sql_string(coin)}
          AND polled_at >= toDateTime({_sql_string(_sql_dt(start_ms))}, 'UTC')
          AND polled_at <= toDateTime({_sql_string(_sql_dt(end_ms))}, 'UTC')
        GROUP BY coin, hour
        ORDER BY hour ASC
        """
        oracle_rows.extend(client.query(sql))
    oracle = pd.DataFrame(oracle_rows)
    if funding.empty:
        return funding
    funding["funding_hour"] = _to_utc_timestamp(funding["time"]).dt.floor("h")
    if not oracle.empty:
        oracle["funding_hour"] = pd.to_datetime(oracle["hour"], utc=True)
        funding = funding.merge(
            oracle[["funding_hour", "oracle_price", "mark_price", "mid_price"]],
            on="funding_hour",
            how="left",
            validate="many_to_one",
        )
    return funding.drop(columns=["funding_hour"])


def _collect_funding_allium(
    client: AlliumExplorerClient,
    *,
    coin: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    # Use official settled funding rates, then attach hourly native Allium
    # oracle/mark/mid context from Hyperliquid asset contexts.
    funding = _collect_funding_official(HyperliquidInfoClient(), coin=coin, window=window)
    if funding.empty:
        return funding

    oracle_rows: list[dict[str, Any]] = []
    for start_ms, end_ms in _monthly_ms_windows(window):
        sql = f"""
        SELECT
            coin,
            date_trunc('hour', timestamp) AS hour,
            avg(oracle_price) AS oracle_price,
            avg(mark_price) AS mark_price,
            avg(mid_price) AS mid_price
        FROM hyperliquid.raw.perpetual_market_asset_contexts
        WHERE coin = {_sql_string(coin)}
          AND timestamp >= {_sql_string(_sql_dt(start_ms))}
          AND timestamp <= {_sql_string(_sql_dt(end_ms))}
          AND oracle_price IS NOT NULL
        GROUP BY coin, hour
        ORDER BY hour ASC
        """
        oracle_rows.extend(
            client.query(
                sql,
                title=f"IL Risk {coin} Allium hourly oracle {_sql_dt(start_ms)}",
            )
        )

    oracle = pd.DataFrame(oracle_rows)
    funding["funding_hour"] = _to_utc_timestamp(funding["time"]).dt.floor("h")
    if not oracle.empty:
        oracle["funding_hour"] = pd.to_datetime(oracle["hour"], utc=True).dt.floor("h")
        funding = funding.merge(
            oracle[["funding_hour", "oracle_price", "mark_price", "mid_price"]],
            on="funding_hour",
            how="left",
            validate="many_to_one",
        )
    funding["source"] = "allium_explorer"
    return funding.drop(columns=["funding_hour"])


def _collect_funding_zeroxarchive_allium(
    *,
    zeroxarchive_client: Any,
    allium_client: AlliumExplorerClient,
    coin: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    funding = _collect_funding_zeroxarchive(zeroxarchive_client, coin=coin, window=window)
    missing_before = int(funding["oracle_price"].isna().sum()) if "oracle_price" in funding else len(funding)
    if missing_before == 0:
        funding["source"] = "0xarchive"
        return funding

    allium = _collect_funding_allium(allium_client, coin=coin, window=window)
    allium_fill = allium[["time", "oracle_price", "mark_price", "mid_price"]].copy()
    allium_fill["funding_hour"] = _to_utc_timestamp(allium_fill["time"]).dt.floor("h")
    allium_fill = allium_fill.drop(columns=["time"]).rename(
        columns={
            "oracle_price": "allium_oracle_price",
            "mark_price": "allium_mark_price",
            "mid_price": "allium_mid_price",
        }
    )
    funding["funding_hour"] = _to_utc_timestamp(funding["time"]).dt.floor("h")
    funding = funding.merge(allium_fill, on="funding_hour", how="left", validate="many_to_one")
    for base, fill in [
        ("oracle_price", "allium_oracle_price"),
        ("mark_price", "allium_mark_price"),
        ("mid_price", "allium_mid_price"),
    ]:
        if base not in funding:
            funding[base] = pd.NA
        funding[base] = funding[base].fillna(funding[fill])
    missing_after = int(funding["oracle_price"].isna().sum())
    log.info(
        "filled %d of %d missing 0xArchive oracle rows from Allium",
        missing_before - missing_after,
        missing_before,
    )
    funding["source"] = "0xarchive_allium"
    return funding.drop(
        columns=[
            "funding_hour",
            "allium_oracle_price",
            "allium_mark_price",
            "allium_mid_price",
        ]
    )


def _collect_funding_zeroxarchive(
    client: Any,
    *,
    coin: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    # Use official Hyperliquid fundingHistory for settled rates (complete 4 368-row coverage).
    funding = _collect_funding_official(HyperliquidInfoClient(), coin=coin, window=window)
    if funding.empty:
        return funding

    # Fetch hourly-averaged oracle/mark prices from 0xArchive.
    start_str = _ts_from_ms(window.start_ms).strftime("%Y-%m-%dT%H:%M:%S")
    end_str = (_ts_from_ms(window.end_ms) + pd.Timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
    price_snapshots = []
    cursor = None
    while True:
        kwargs: dict[str, Any] = {"start": start_str, "end": end_str, "interval": "1h"}
        if cursor is not None:
            kwargs["cursor"] = cursor
        resp = client.hyperliquid.get_price_history(coin, **kwargs)
        price_snapshots.extend(resp.data)
        if resp.next_cursor is None:
            break
        cursor = resp.next_cursor

    if price_snapshots:
        oracle_df = pd.DataFrame(
            [
                {
                    "funding_hour": p.timestamp,
                    "oracle_price": float(p.oracle_price),
                    "mark_price": float(p.mark_price),
                    "mid_price": float(p.mid_price),
                }
                for p in price_snapshots
            ]
        )
        oracle_df["funding_hour"] = pd.to_datetime(oracle_df["funding_hour"], utc=True).dt.floor("h")
        funding["funding_hour"] = _to_utc_timestamp(funding["time"]).dt.floor("h")
        funding = funding.merge(oracle_df, on="funding_hour", how="left", validate="many_to_one")
        funding = funding.drop(columns=["funding_hour"])

        missing = int(funding["oracle_price"].isna().sum())
        if missing:
            log.warning(
                "0xArchive oracle_price missing for %d of %d funding rows; "
                "leaving gaps for a native secondary source.",
                missing,
                len(funding),
            )

    funding["source"] = "0xarchive"
    return funding


def _normalise_prices(
    df: pd.DataFrame,
    *,
    coin: str,
    interval: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("no candle rows returned")

    work = df.copy()
    if "t" in work.columns:
        work["open_time"] = _to_utc_timestamp(work["t"])
        work["close_time"] = _to_utc_timestamp(work["T"])
        work["coin"] = work.get("s", coin)
        work["interval"] = work.get("i", interval)
        rename = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "num_trades"}
        work = work.rename(columns=rename)
    else:
        work["open_time"] = pd.to_datetime(work["hour"], utc=True)
        work["close_time"] = work["open_time"] + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
        work["interval"] = interval

    for column in ["open", "high", "low", "close", "volume"]:
        work[column] = pd.to_numeric(work[column], errors="raise").astype(float)
    work["num_trades"] = pd.to_numeric(work.get("num_trades", 0), errors="coerce").fillna(0).astype("int64")
    work["coin"] = work["coin"].fillna(coin).astype(str)
    work["interval"] = work["interval"].fillna(interval).astype(str)
    work["date"] = work["open_time"].dt.strftime("%Y-%m-%d")

    out = work[
        [
            "coin",
            "interval",
            "open_time",
            "close_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "num_trades",
            "date",
            "source",
        ]
    ].drop_duplicates(subset=["coin", "interval", "open_time"])
    out = out[
        (out["open_time"] >= _ts_from_ms(window.start_ms))
        & (out["open_time"] <= _ts_from_ms(window.end_ms))
    ].sort_values("open_time").reset_index(drop=True)

    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("perp_prices contains non-positive OHLC prices")
    if (out["high"] < out[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("perp_prices contains high below open/close/low")
    if (out["low"] > out[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("perp_prices contains low above open/close/high")
    return out


def _normalise_funding(
    df: pd.DataFrame,
    *,
    coin: str,
    window: CollectionWindow,
) -> pd.DataFrame:
    if df.empty:
        raise ValueError("no funding rows returned")
    work = df.copy()
    work["funding_time"] = _to_utc_timestamp(work["time"])
    work["funding_rate"] = pd.to_numeric(work["fundingRate"], errors="raise").astype(float)
    work["premium"] = pd.to_numeric(work.get("premium"), errors="coerce").astype(float)
    work["coin"] = work["coin"].fillna(coin).astype(str)
    if "oracle_price" not in work:
        work["oracle_price"] = pd.NA
    if "mark_price" not in work:
        work["mark_price"] = pd.NA
    if "mid_price" not in work:
        work["mid_price"] = pd.NA
    for column in ["oracle_price", "mark_price", "mid_price"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["date"] = work["funding_time"].dt.strftime("%Y-%m-%d")
    out = work[
        [
            "coin",
            "funding_time",
            "funding_rate",
            "premium",
            "oracle_price",
            "mark_price",
            "mid_price",
            "date",
            "source",
        ]
    ].drop_duplicates(subset=["coin", "funding_time"])
    out = out[
        (out["funding_time"] >= _ts_from_ms(window.start_ms))
        & (out["funding_time"] <= _ts_from_ms(window.end_ms))
    ].sort_values("funding_time").reset_index(drop=True)
    if out["funding_rate"].abs().gt(0.04).any():
        raise ValueError("funding_rates contains absolute hourly funding rate above Hyperliquid cap")
    return out


def _validate_hourly_coverage(
    df: pd.DataFrame,
    window: CollectionWindow,
    *,
    time_column: str,
    label: str,
) -> None:
    start = _ts_from_ms(window.start_ms).floor("h")
    end = _ts_from_ms(window.end_ms).floor("h")
    expected = pd.date_range(start, end, freq="h", tz="UTC")
    actual = pd.DatetimeIndex(df[time_column]).floor("h").drop_duplicates()
    missing = expected.difference(actual)
    if len(missing):
        preview = ", ".join(ts.isoformat() for ts in missing[:5])
        raise ValueError(
            f"{label} missing {len(missing)} hourly rows over requested window; first gaps: {preview}. "
            "Use a native historical provider with full coverage; do not substitute another venue."
        )


def _validate_oracle_coverage(df: pd.DataFrame, *, source_name: str) -> None:
    missing = int(df["oracle_price"].isna().sum())
    if missing:
        raise ValueError(
            f"funding_rates missing oracle_price for {missing} rows from {source_name}. "
            "Use a native Hyperliquid historical source with full oracle_px coverage, "
            "or leave the dataset ungenerated."
        )


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


@app.command("collect-prices")
def cmd_collect_prices(
    from_date: str = typer.Option(DEFAULT_START, "--from"),
    to_date: str = typer.Option(DEFAULT_END, "--to"),
    coin: str = typer.Option(DEFAULT_COIN, "--coin"),
    interval: str = typer.Option(DEFAULT_INTERVAL, "--interval"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    source: str = typer.Option("auto", "--source", help="auto, allium, quicknode, 0xarchive, or official"),
    api_url: str = typer.Option(DEFAULT_HYPERLIQUID_API_URL, "--api-url"),
) -> None:
    """Collect hourly ETH perp OHLCV candles."""

    _setup_logging()
    df = collect_perp_prices(
        coin=coin,
        interval=interval,
        from_date=from_date,
        to_date=to_date,
        results_dir=results_dir,
        source=_parse_source(source),
        api_url=api_url,
    )
    typer.echo(f"wrote {results_dir / 'perp_prices.parquet'} ({len(df)} rows)")


@app.command("collect-funding")
def cmd_collect_funding(
    from_date: str = typer.Option(DEFAULT_START, "--from"),
    to_date: str = typer.Option(DEFAULT_END, "--to"),
    coin: str = typer.Option(DEFAULT_COIN, "--coin"),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR, "--results-dir"),
    source: str = typer.Option("auto", "--source", help="auto, allium, quicknode, 0xarchive, or official"),
    api_url: str = typer.Option(DEFAULT_HYPERLIQUID_API_URL, "--api-url"),
) -> None:
    """Collect hourly ETH perp funding rates."""

    _setup_logging()
    df = collect_funding_rates(
        coin=coin,
        from_date=from_date,
        to_date=to_date,
        results_dir=results_dir,
        source=_parse_source(source),
        api_url=api_url,
    )
    typer.echo(f"wrote {results_dir / 'funding_rates.parquet'} ({len(df)} rows)")


def _parse_source(value: str) -> MarketDataSource:
    if value not in {"auto", "allium", "quicknode", "0xarchive", "official"}:
        raise typer.BadParameter("source must be one of: auto, allium, quicknode, 0xarchive, official")
    return value  # type: ignore[return-value]


if __name__ == "__main__":
    app()
