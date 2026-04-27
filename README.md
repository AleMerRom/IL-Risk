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

## Expected data outputs (Parquet)
Place all produced datasets in `data/processed/`:
- `swap_events.parquet` (Module 1)
- `mint_burn_events.parquet` (Module 1; full history since deployment)
- `collect_events.parquet` (Module 1 support file for Module 4 fee analysis)
- `liquidity_snapshots.parquet` (Module 1; daily tick-level liquidity map)
- `slot0_snapshots.parquet` (Module 1; daily slot0 at snapshot blocks)
- `simulated_trades.parquet` (Module 3)
- `perp_prices.parquet` (Module 5)
- `funding_rates.parquet` (Module 5)
- `hedge_results.parquet` (Module 5)

## Repo structure
The current codebase contains only native `il_risk` Module 1 extraction code. Paul’s older `fabdl`
package layout is not used.

```
.
├── data/
│   ├── raw/                 # any raw downloads / intermediate artifacts
│   └── processed/           # final parquet outputs listed above
├── figs/                    # figures used in the report
├── notebooks/               # optional exploratory notebooks
├── reports/                 # final write-up (PDF/LaTeX/Markdown, your choice)
├── scripts/                 # CLI entrypoints
├── src/il_risk/             # Module 1 package code
└── tests/                   # schema checks / validations
```

## Module 1 extraction

Create a local `.env` from `.env.example` and set `RPC_URL`. Set `ARCHIVE_RPC_URL` when the primary
endpoint is not archive-capable.

Run the full Module 1 refresh:

```bash
PYTHONPATH=src python scripts/data_extraction.py extract all
```

Useful individual commands:

```bash
PYTHONPATH=src python scripts/data_extraction.py archive-current
PYTHONPATH=src python scripts/data_extraction.py extract mints-burns
PYTHONPATH=src python scripts/data_extraction.py extract collects
PYTHONPATH=src python scripts/data_extraction.py extract swaps
PYTHONPATH=src python scripts/data_extraction.py extract slot0
PYTHONPATH=src python scripts/data_extraction.py extract liquidity-snapshots
PYTHONPATH=src python scripts/data_extraction.py validate
```

Paul’s normalized draft files are archived under:

```text
data/archive/paul_end_of_day_2025-10-2026-03/
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
