"""CLI for Module 1 on-chain extraction."""

from __future__ import annotations

import logging
import shutil
from datetime import timezone, date, datetime
from pathlib import Path

import typer
from dotenv import load_dotenv

from il_risk.rpc.block_index import BlockIndex
from il_risk.constants import POOL_DEPLOYMENT_BLOCK
from il_risk.rpc.client import RpcClient, RpcConfig

app = typer.Typer(no_args_is_help=True, add_completion=False)
extract_app = typer.Typer(no_args_is_help=True, help="Module 1 on-chain extraction")
app.add_typer(extract_app, name="extract")

DEFAULT_START = "2025-10-01"
DEFAULT_END = "2026-03-31"
ARCHIVE_DIR = Path("data/archive/paul_end_of_day_2025-10-2026-03")
MODULE1_FILES = [
    "swap_events.parquet",
    "mint_burn_events.parquet",
    "collect_events.parquet",
    "slot0_snapshots.parquet",
    "liquidity_snapshots.parquet",
]
MODULE1_PART_DIRS = [
    "swap_events_parts",
    "mint_burn_events_parts",
    "collect_events_parts",
    "slot0_snapshots_parts",
    "liquidity_snapshots_parts",
]
MODULE1_CHECKPOINTS = [
    "swap_events.json",
    "mint_events.json",
    "burn_events.json",
    "collect_events.json",
]
MODULE1_PARALLEL_CHECKPOINT_DIRS = [
    "swap_events_parallel",
    "mint_events_parallel",
    "burn_events_parallel",
    "collect_events_parallel",
]


def _setup() -> tuple[RpcClient, Path]:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    data_dir = Path("data")
    return RpcClient(RpcConfig.from_env(), data_dir=data_dir), data_dir


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _utc_ts(day: date, *, end_of_day: bool = False) -> int:
    t = datetime.max.time() if end_of_day else datetime.min.time()
    return int(datetime.combine(day, t, tzinfo=timezone.utc).timestamp())


def _study_blocks(
    rpc: RpcClient,
    data_dir: Path,
    from_date: str,
    to_date: str,
) -> tuple[int, int]:
    index = BlockIndex(rpc, cache_path=data_dir / "checkpoints" / "block_index.sqlite")
    start = index.first_block_at_or_after(_utc_ts(_parse_date(from_date)))
    end = index.block_at_timestamp(_utc_ts(_parse_date(to_date), end_of_day=True))
    return start, end


def _clear_module1_workdirs(data_dir: Path) -> None:
    for name in MODULE1_PART_DIRS:
        shutil.rmtree(data_dir / "raw" / name, ignore_errors=True)
    for name in MODULE1_PARALLEL_CHECKPOINT_DIRS:
        shutil.rmtree(data_dir / "checkpoints" / name, ignore_errors=True)
    for name in MODULE1_CHECKPOINTS:
        path = data_dir / "checkpoints" / name
        if path.exists():
            path.unlink()


@app.command("archive-current")
def archive_current(
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    archive_dir: Path = typer.Option(ARCHIVE_DIR, "--archive-dir"),
) -> None:
    """Copy current processed Module 1 files into an archive folder."""

    processed = data_dir / "processed"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for filename in MODULE1_FILES:
        src = processed / filename
        if src.exists():
            shutil.copy2(src, archive_dir / filename)
            typer.echo(f"archived {src} -> {archive_dir / filename}")


@extract_app.command("swaps")
def cmd_swaps(
    from_date: str = typer.Option(DEFAULT_START, "--from"),
    to_date: str = typer.Option(DEFAULT_END, "--to"),
    compact_output: bool = typer.Option(True, "--compact/--no-compact"),
    from_block: int | None = typer.Option(None, "--from-block"),
    to_block: int | None = typer.Option(None, "--to-block"),
    workers: int = typer.Option(1, "--workers"),
    chunk_blocks: int | None = typer.Option(None, "--chunk-blocks"),
) -> None:
    """Fetch Swap events for the study window."""

    from il_risk.pipelines.module1.swaps import compact_swaps, extract_swaps

    rpc, data_dir = _setup()
    if from_block is None or to_block is None:
        start_block, end_block = _study_blocks(rpc, data_dir, from_date, to_date)
        from_block = from_block or start_block
        to_block = to_block or end_block
    count = extract_swaps(
        rpc, from_block, to_block, out_dir=data_dir, workers=workers, chunk_blocks=chunk_blocks
    )
    typer.echo(f"fetched {count} swap events over blocks {from_block}..{to_block}")
    if compact_output:
        rows = compact_swaps(data_dir)
        typer.echo(f"compacted swap_events.parquet ({rows} rows)")


@extract_app.command("mints-burns")
def cmd_mints_burns(
    to_date: str = typer.Option(DEFAULT_END, "--to"),
    from_block: int = typer.Option(POOL_DEPLOYMENT_BLOCK, "--from-block"),
    compact_output: bool = typer.Option(True, "--compact/--no-compact"),
    workers: int = typer.Option(1, "--workers"),
    chunk_blocks: int | None = typer.Option(None, "--chunk-blocks"),
) -> None:
    """Fetch complete Mint/Burn history from deployment through the end date."""

    from il_risk.pipelines.module1.liquidity_events import compact_mints_burns, extract_mints_burns

    rpc, data_dir = _setup()
    index = BlockIndex(rpc, cache_path=data_dir / "checkpoints" / "block_index.sqlite")
    to_block = index.block_at_timestamp(_utc_ts(_parse_date(to_date), end_of_day=True))
    count = extract_mints_burns(
        rpc, from_block, to_block, out_dir=data_dir, workers=workers, chunk_blocks=chunk_blocks
    )
    typer.echo(f"fetched {count} mint/burn events over blocks {from_block}..{to_block}")
    if compact_output:
        rows = compact_mints_burns(data_dir)
        typer.echo(f"compacted mint_burn_events.parquet ({rows} rows)")


@extract_app.command("collects")
def cmd_collects(
    to_date: str = typer.Option(DEFAULT_END, "--to"),
    from_block: int = typer.Option(POOL_DEPLOYMENT_BLOCK, "--from-block"),
    compact_output: bool = typer.Option(True, "--compact/--no-compact"),
    workers: int = typer.Option(1, "--workers"),
    chunk_blocks: int | None = typer.Option(None, "--chunk-blocks"),
) -> None:
    """Fetch complete Collect history from deployment through the end date."""

    from il_risk.pipelines.module1.liquidity_events import compact_collects, extract_collects

    rpc, data_dir = _setup()
    index = BlockIndex(rpc, cache_path=data_dir / "checkpoints" / "block_index.sqlite")
    to_block = index.block_at_timestamp(_utc_ts(_parse_date(to_date), end_of_day=True))
    count = extract_collects(
        rpc, from_block, to_block, out_dir=data_dir, workers=workers, chunk_blocks=chunk_blocks
    )
    typer.echo(f"fetched {count} collect events over blocks {from_block}..{to_block}")
    if compact_output:
        rows = compact_collects(data_dir)
        typer.echo(f"compacted collect_events.parquet ({rows} rows)")


@extract_app.command("slot0")
def cmd_slot0(
    from_date: str = typer.Option(DEFAULT_START, "--from"),
    to_date: str = typer.Option(DEFAULT_END, "--to"),
    compact_output: bool = typer.Option(True, "--compact/--no-compact"),
) -> None:
    """Fetch daily slot0 snapshots at blocks closest to 00:00 UTC."""

    from il_risk.pipelines.module1.snapshots import compact_slot0, extract_slot0_daily

    rpc, data_dir = _setup()
    count = extract_slot0_daily(rpc, _parse_date(from_date), _parse_date(to_date), out_dir=data_dir)
    typer.echo(f"wrote {count} slot0 snapshots")
    if compact_output:
        rows = compact_slot0(data_dir)
        typer.echo(f"compacted slot0_snapshots.parquet ({rows} rows)")


@extract_app.command("liquidity-snapshots")
def cmd_liquidity_snapshots(
    from_date: str = typer.Option(DEFAULT_START, "--from"),
    to_date: str = typer.Option(DEFAULT_END, "--to"),
    compact_output: bool = typer.Option(True, "--compact/--no-compact"),
    force_event_replay: bool = typer.Option(False, "--force-event-replay"),
) -> None:
    """Fetch daily tick-level liquidity maps at blocks closest to 00:00 UTC."""

    from il_risk.pipelines.module1.snapshots import (
        compact_liquidity_snapshots,
        extract_liquidity_snapshots_daily,
    )

    rpc, data_dir = _setup()
    count = extract_liquidity_snapshots_daily(
        rpc,
        _parse_date(from_date),
        _parse_date(to_date),
        out_dir=data_dir,
        force_path_b=force_event_replay,
    )
    typer.echo(f"wrote {count} liquidity snapshot rows")
    if compact_output:
        rows = compact_liquidity_snapshots(data_dir)
        typer.echo(f"compacted liquidity_snapshots.parquet ({rows} rows)")


@extract_app.command("swap-mid-prices")
def cmd_swap_mid_prices(
    swap_events_path: Path | None = typer.Option(None, "--swap-events-path"),
    output_path: Path | None = typer.Option(None, "--output-path"),
    limit: int | None = typer.Option(None, "--limit"),
    include_timestamps: bool = typer.Option(False, "--include-timestamps"),
    from_block: int | None = typer.Option(None, "--from-block"),
    to_block: int | None = typer.Option(None, "--to-block"),
    batch_size: int = typer.Option(100, "--batch-size"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    sample_blocks: int | None = typer.Option(None, "--sample-blocks"),
    sample_seed: int = typer.Option(413, "--sample-seed"),
) -> None:
    """Fetch pre-swap slot0 prices for Module 3 effective spread estimates."""

    from il_risk.pipelines.module1.compact import extract_swap_mid_prices

    rpc, data_dir = _setup()
    table = extract_swap_mid_prices(
        rpc,
        data_dir=data_dir,
        swap_events_path=swap_events_path,
        output_path=output_path,
        limit=limit,
        include_timestamps=include_timestamps,
        from_block=from_block,
        to_block=to_block,
        batch_size=batch_size,
        resume=resume,
        sample_blocks=sample_blocks,
        sample_seed=sample_seed,
    )
    path = output_path or data_dir / "processed" / "swap_mid_prices.parquet"
    typer.echo(f"wrote {path} ({len(table)} unique swap blocks)")


@extract_app.command("all")
def cmd_extract_all(
    from_date: str = typer.Option(DEFAULT_START, "--from"),
    to_date: str = typer.Option(DEFAULT_END, "--to"),
    archive_first: bool = typer.Option(True, "--archive-first/--no-archive-first"),
    fresh: bool = typer.Option(True, "--fresh/--resume"),
    workers: int = typer.Option(4, "--workers"),
    chunk_blocks: int | None = typer.Option(None, "--chunk-blocks"),
) -> None:
    """Run the complete Module 1 extraction sequence."""

    if archive_first:
        archive_current(data_dir=Path("data"), archive_dir=ARCHIVE_DIR)
    if fresh:
        _clear_module1_workdirs(Path("data"))
    cmd_mints_burns(
        to_date=to_date,
        from_block=POOL_DEPLOYMENT_BLOCK,
        compact_output=True,
        workers=workers,
        chunk_blocks=chunk_blocks,
    )
    cmd_collects(
        to_date=to_date,
        from_block=POOL_DEPLOYMENT_BLOCK,
        compact_output=True,
        workers=workers,
        chunk_blocks=chunk_blocks,
    )
    cmd_swaps(
        from_date=from_date,
        to_date=to_date,
        compact_output=True,
        from_block=None,
        to_block=None,
        workers=workers,
        chunk_blocks=chunk_blocks,
    )
    cmd_slot0(from_date=from_date, to_date=to_date, compact_output=True)
    cmd_liquidity_snapshots(
        from_date=from_date,
        to_date=to_date,
        compact_output=True,
        force_event_replay=False,
    )


@app.command("validate")
def cmd_validate(data_dir: Path = typer.Option(Path("data"), "--data-dir")) -> None:
    """Validate Module 1 processed Parquets."""

    from il_risk.pipelines.module1.validate import validate_module1

    for message in validate_module1(data_dir):
        typer.echo(message)


@app.command("validation-tables")
def cmd_validation_tables(data_dir: Path = typer.Option(Path("data"), "--data-dir")) -> None:
    """Write report-ready validation tables that do not require RPC."""

    from il_risk.pipelines.module1.validate import validate_slot0_against_swaps

    out_dir = data_dir / "processed"
    table = validate_slot0_against_swaps(data_dir)
    path = out_dir / "slot0_swap_price_check.parquet"
    table.to_parquet(path, index=False)
    typer.echo(f"wrote {path} ({len(table)} rows)")


@app.command("onchain-tick-check")
def cmd_onchain_tick_check(
    data_dir: Path = typer.Option(Path("data"), "--data-dir"),
    snapshot_block: int | None = typer.Option(None, "--snapshot-block"),
    sample_size: int = typer.Option(10, "--sample-size"),
) -> None:
    """Compare sampled liquidity ticks against direct archive pool.ticks() calls."""

    from il_risk.pipelines.module1.validate import validate_liquidity_ticks_against_rpc

    rpc, _ = _setup()
    table = validate_liquidity_ticks_against_rpc(
        data_dir, rpc, snapshot_block=snapshot_block, sample_size=sample_size
    )
    path = data_dir / "processed" / "liquidity_tick_spot_check.parquet"
    table.to_parquet(path, index=False)
    typer.echo(f"wrote {path} ({len(table)} rows)")


if __name__ == "__main__":
    app()
