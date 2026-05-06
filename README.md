# FIN-413 — FABDL Final Project (2026)

Repository for the EPFL FIN-413 “Financial applications of blockchains and distributed ledgers” final project.

## Pool under study (fixed)
- **Protocol**: Uniswap V3
- **Pair**: USDC / WETH
- **Fee tier**: 0.05% (tick spacing = 10)
- **Pool address**: `0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640`
- **Network**: Ethereum Mainnet
- **Token0**: USDC (6 decimals)
- **Token1**: WETH (18 decimals)

## Study window
Chosen Module 1 window: **2025-10-01 00:00:00 UTC → 2026-03-31 23:59:59 UTC**.

Daily snapshots are taken at the block closest to **00:00 UTC** for each date in this window.

## Required deliverables (from the PDF)
You must submit:
- **Codebase**: a well-documented Python repository covering all 5 modules
- **Report**: a research report answering all questions, with fully-labeled figures

Important constraint:
- **All on-chain data must be extracted directly via RPC calls to an Ethereum archive node**. The grader will verify direct on-chain calls.

## Data layout
`data/processed/` contains only the canonical Module 1 source datasets used by later modules:
- `swap_events.parquet` (Module 1)
- `mint_burn_events.parquet` (Module 1; full history since deployment)
- `collect_events.parquet` (Module 1 support file for Module 4 fee analysis)
- `liquidity_snapshots.parquet` (Module 1; daily tick-level liquidity map)
- `slot0_snapshots.parquet` (Module 1; daily slot0 at snapshot blocks)

Generated deliverables and support tables live under `data/results/module_*`:
- `data/results/module_1/`: optional validation and report-ready check tables.
- `data/results/module_3/`: `simulated_trades.parquet`, price-impact summaries, effective-spread outputs, and Module 3 figures.
- `data/results/module_4/`: synthetic LP position tables, fee/IL/net P&L time series, and Module 4 figures.
- `data/results/module_5/`: reserved for `perp_prices.parquet`, `funding_rates.parquet`, `hedge_results.parquet`, and Module 5 figures.

Old raw part files, checkpoints, previous archives, and one-off exports are kept locally under `data/_archive/`.

## Repo structure
The code is organized around the deliverable filenames from the project PDF.  The repo is meant to
be easy to inspect for grading, not packaged as a reusable library.

```
.
├── data/
│   ├── processed/           # canonical Module 1 source parquet files
│   ├── results/             # generated module deliverables and figures
│   └── _archive/            # local backups, raw parts, caches, old exports
├── scripts/                 # optional convenience wrappers
├── src/
│   ├── module1/             # data_extraction.py and extraction helpers
│   ├── module2/             # liquidity_analysis.py
│   ├── module3/             # swap_simulator.py and slippage_analysis.py
│   ├── module4/             # lp_analytics.py
│   ├── module5/             # hedge_backtest.py
│   └── shared/              # constants, RPC, schemas, Uniswap math/events
└── tests/                   # schema checks / validations
```

Guiding rule:
- Deliverable files keep their PDF names inside the corresponding `src/module*/` folder.
- Small shared helpers stay in `src/shared/` only when they avoid meaningful duplication.
- Run commands with `PYTHONPATH=src`.

## Module 1 extraction

Create a local `.env` from `.env.example` and set `RPC_URL`. Set `ARCHIVE_RPC_URL` when the primary
endpoint is not archive-capable.

Run the full Module 1 refresh:

```bash
PYTHONPATH=src python src/module1/data_extraction.py extract all
```

Useful individual commands:

```bash
PYTHONPATH=src python src/module1/data_extraction.py archive-current
PYTHONPATH=src python src/module1/data_extraction.py extract mints-burns
PYTHONPATH=src python src/module1/data_extraction.py extract collects
PYTHONPATH=src python src/module1/data_extraction.py extract swaps
PYTHONPATH=src python src/module1/data_extraction.py extract slot0
PYTHONPATH=src python src/module1/data_extraction.py extract liquidity-snapshots
PYTHONPATH=src python src/module1/data_extraction.py extract swap-mid-prices
PYTHONPATH=src python src/module1/data_extraction.py validate
```

Older normalized drafts and raw extraction leftovers are archived under:

```text
data/_archive/
```

## Module 3 simulation (slippage grid)
The standalone simulator lives in `src/module3/swap_simulator.py`; the Module 3 workflow lives in
`src/module3/slippage_analysis.py`.

To generate `simulated_trades.parquet` from the Module 1 snapshot Parquets:

```bash
PYTHONPATH=src python src/module3/slippage_analysis.py run-all --allow-mid-price-subset
```

## Module 4 LP analytics
The Module 4 workflow lives in `src/module4/lp_analytics.py`.

```bash
PYTHONPATH=src python src/module4/lp_analytics.py run-all
```

## Data dictionary

### `swap_events.parquet`

| name | type | unit | description |
| --- | --- | --- | --- |
| `block_number` | int64 | block | Ethereum block number. |
| `block_timestamp` | timestamp UTC | time | Timestamp resolved once per unique swap block. |
| `tx_hash` | string | hex | Transaction hash. |
| `log_index` | int32 | index | Event log index within the transaction receipt/block ordering. |
| `sender` | string | address | Swap event sender. |
| `recipient` | string | address | Swap event recipient. |
| `amount0_raw` | decimal | USDC raw units | Signed token0 amount; positive means deposited into pool. |
| `amount1_raw` | decimal | WETH raw units | Signed token1 amount; positive means deposited into pool. |
| `amount0_usdc` | float64 | USDC | Decimal-adjusted signed USDC amount. |
| `amount1_weth` | float64 | WETH | Decimal-adjusted signed WETH amount. |
| `sqrt_price_x96` | decimal | Q64.96 | Pool sqrt price after the swap. |
| `price_usdc_per_weth` | float64 | USDC/WETH | Human ETH price after the swap. |
| `active_liquidity` | decimal | liquidity | Active pool liquidity after the swap. |
| `tick` | int32 | tick | Active tick after the swap. |
| `trade_direction` | string | category | Taker direction: `buy_weth` or `sell_weth`. |
| `usd_notional` | float64 | USD | Absolute USDC notional. |
| `date` | string | date | UTC calendar date. |

### `mint_burn_events.parquet`

| name | type | unit | description |
| --- | --- | --- | --- |
| `block_number` | int64 | block | Ethereum block number. |
| `block_timestamp` | timestamp UTC | time | Event block timestamp. |
| `tx_hash` | string | hex | Transaction hash. |
| `log_index` | int32 | index | Event log index. |
| `event_type` | string | category | `mint` or `burn`. |
| `owner` | string | address | LP position owner. |
| `tick_lower` | int32 | tick | Lower initialized tick. |
| `tick_upper` | int32 | tick | Upper initialized tick. |
| `liquidity_delta` | decimal | liquidity | Signed liquidity change; burns are negative. |
| `liquidity_amount_raw` | decimal | liquidity | Absolute event liquidity amount. |
| `amount0_raw` | decimal | USDC raw units | Raw USDC amount. |
| `amount1_raw` | decimal | WETH raw units | Raw WETH amount. |
| `amount0_usdc` | float64 | USDC | Decimal-adjusted USDC amount. |
| `amount1_weth` | float64 | WETH | Decimal-adjusted WETH amount. |
| `date` | string | date | UTC calendar date. |

### `collect_events.parquet`

| name | type | unit | description |
| --- | --- | --- | --- |
| `block_number` | int64 | block | Ethereum block number. |
| `block_timestamp` | timestamp UTC | time | Event block timestamp. |
| `tx_hash` | string | hex | Transaction hash. |
| `log_index` | int32 | index | Event log index. |
| `owner` | string | address | LP position owner. |
| `recipient` | string | address | Fee recipient. |
| `tick_lower` | int32 | tick | Lower position tick. |
| `tick_upper` | int32 | tick | Upper position tick. |
| `amount0_raw` | decimal | USDC raw units | Raw collected USDC fees. |
| `amount1_raw` | decimal | WETH raw units | Raw collected WETH fees. |
| `amount0_usdc` | float64 | USDC | Decimal-adjusted collected USDC. |
| `amount1_weth` | float64 | WETH | Decimal-adjusted collected WETH. |
| `date` | string | date | UTC calendar date. |

### `slot0_snapshots.parquet`

| name | type | unit | description |
| --- | --- | --- | --- |
| `date` | string | date | Snapshot date. |
| `snapshot_block` | int64 | block | Block closest to 00:00 UTC. |
| `snapshot_timestamp` | timestamp UTC | time | Timestamp of snapshot block. |
| `sqrt_price_x96` | decimal | Q64.96 | Raw pool sqrt price. |
| `price_usdc_per_weth` | float64 | USDC/WETH | Human ETH price. |
| `current_tick` | int32 | tick | Active pool tick. |
| `observation_index` | int32 | index | Uniswap oracle observation index. |
| `observation_cardinality` | int32 | count | Oracle observation cardinality. |
| `fee_protocol` | int32 | raw | Protocol fee setting. |
| `unlocked` | bool | flag | Pool lock state. |
| `fee_growth_global_0_x128` | decimal | Q128 | Global fee growth for USDC. |
| `fee_growth_global_1_x128` | decimal | Q128 | Global fee growth for WETH. |

### `liquidity_snapshots.parquet`

| name | type | unit | description |
| --- | --- | --- | --- |
| `date` | string | date | Snapshot date. |
| `snapshot_block` | int64 | block | Block closest to 00:00 UTC. |
| `snapshot_timestamp` | timestamp UTC | time | Timestamp of snapshot block. |
| `tick` | int32 | tick | Initialized tick index. |
| `liquidityNet` | decimal | liquidity | Signed liquidity net at this tick. |
| `liquidityGross` | decimal | liquidity | Gross liquidity at this tick. |
| `active_liquidity` | decimal | liquidity | Cumulative active liquidity after applying this tick. |
| `price_lower` | float64 | USDC/WETH | Price at the lower edge of this tick interval. |
| `price_upper` | float64 | USDC/WETH | Price at the upper edge of this tick interval. |
| `fee_growth_outside_0_x128` | decimal | Q128 | Fee growth outside this tick for USDC. |
| `fee_growth_outside_1_x128` | decimal | Q128 | Fee growth outside this tick for WETH. |
