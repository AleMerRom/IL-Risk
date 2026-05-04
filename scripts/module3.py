"""Module 3 report artifact runner.

Examples:
    PYTHONPATH=src python scripts/module3.py check-inputs
    PYTHONPATH=src python scripts/module3.py simulate
    PYTHONPATH=src python scripts/module3.py effective-spreads
    PYTHONPATH=src python scripts/module3.py run-all
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import typer

from il_risk.pipelines.module3.slippage_analysis import (
    compute_effective_spreads,
    plot_effective_spread_comparison,
    plot_price_impact_curves,
    run_simulation_grid,
    summarize_effective_spreads,
    summarize_price_impact,
)

app = typer.Typer(no_args_is_help=True, add_completion=False)

DEFAULT_PROCESSED_DIR = Path("data/processed_parquets")
DEFAULT_FIGS_DIR = Path("figs")

_CACHE_ROOT = Path(tempfile.gettempdir()) / "il-risk-cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPLCONFIGDIR = _CACHE_ROOT / "matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))


def _paths(processed_dir: Path, figs_dir: Path) -> dict[str, Path]:
    return {
        "liquidity": processed_dir / "liquidity_snapshots.parquet",
        "slot0": processed_dir / "slot0_snapshots.parquet",
        "swaps": processed_dir / "swap_events.parquet",
        "swap_mid_prices": processed_dir / "swap_mid_prices.parquet",
        "simulated_trades": processed_dir / "simulated_trades.parquet",
        "price_impact_summary": processed_dir / "price_impact_summary.parquet",
        "effective_spreads": processed_dir / "effective_spreads.parquet",
        "effective_spread_summary": processed_dir / "effective_spread_summary.parquet",
        "price_impact_plot": figs_dir / "module3_price_impact_curves.png",
        "effective_spread_plot": figs_dir / "module3_effective_spread_by_size.png",
    }


def _require(path: Path, label: str) -> None:
    if not path.exists():
        typer.echo(f"missing {label}: {path}", err=True)
        raise typer.Exit(1)


def _row_count(path: Path) -> int:
    return len(pd.read_parquet(path))


def _run_simulation_artifacts(paths: dict[str, Path]) -> None:
    _require(paths["liquidity"], "liquidity snapshots")
    _require(paths["slot0"], "slot0 snapshots")

    simulated = run_simulation_grid(
        liquidity_path=paths["liquidity"],
        slot0_path=paths["slot0"],
        output_path=paths["simulated_trades"],
    )
    typer.echo(f"wrote simulated trades: {paths['simulated_trades']} ({len(simulated)} rows)")

    summary = summarize_price_impact(
        simulated_trades=simulated,
        output_path=paths["price_impact_summary"],
    )
    typer.echo(f"wrote price impact summary: {paths['price_impact_summary']} ({len(summary)} rows)")

    plot_path = plot_price_impact_curves(
        summary=summary,
        output_path=paths["price_impact_plot"],
    )
    typer.echo(f"wrote price impact figure: {plot_path}")


def _run_effective_spread_artifacts(paths: dict[str, Path]) -> None:
    _require(paths["swaps"], "swap events")
    _require(paths["swap_mid_prices"], "pre-swap mid prices")
    _require(paths["price_impact_summary"], "price impact summary")

    spreads = compute_effective_spreads(
        swaps_path=paths["swaps"],
        pre_swap_mid_prices_path=paths["swap_mid_prices"],
        output_path=paths["effective_spreads"],
    )
    typer.echo(f"wrote effective spreads: {paths['effective_spreads']} ({len(spreads)} rows)")

    summary = summarize_effective_spreads(
        effective_spreads=spreads,
        output_path=paths["effective_spread_summary"],
    )
    typer.echo(
        f"wrote effective spread summary: {paths['effective_spread_summary']} "
        f"({len(summary)} rows)"
    )

    plot_path = plot_effective_spread_comparison(
        simulated_summary_path=paths["price_impact_summary"],
        empirical_summary=summary,
        output_path=paths["effective_spread_plot"],
    )
    typer.echo(f"wrote effective spread comparison figure: {plot_path}")


@app.command("check-inputs")
def check_inputs(
    processed_dir: Path = typer.Option(DEFAULT_PROCESSED_DIR, "--processed-dir"),
    require_swap_mid_prices: bool = typer.Option(False, "--require-swap-mid-prices"),
) -> None:
    """Check that the parquet inputs needed by Module 3 are present."""

    paths = _paths(processed_dir, DEFAULT_FIGS_DIR)
    required = [
        ("liquidity snapshots", paths["liquidity"]),
        ("slot0 snapshots", paths["slot0"]),
        ("swap events", paths["swaps"]),
    ]
    if require_swap_mid_prices:
        required.append(("pre-swap mid prices", paths["swap_mid_prices"]))

    for label, path in required:
        _require(path, label)
        typer.echo(f"found {label}: {path} ({_row_count(path)} rows)")


@app.command("simulate")
def simulate(
    processed_dir: Path = typer.Option(DEFAULT_PROCESSED_DIR, "--processed-dir"),
    figs_dir: Path = typer.Option(DEFAULT_FIGS_DIR, "--figs-dir"),
) -> None:
    """Generate Task 3.2 and 3.3 simulated trade tables and figure."""

    _run_simulation_artifacts(_paths(processed_dir, figs_dir))


@app.command("effective-spreads")
def effective_spreads(
    processed_dir: Path = typer.Option(DEFAULT_PROCESSED_DIR, "--processed-dir"),
    figs_dir: Path = typer.Option(DEFAULT_FIGS_DIR, "--figs-dir"),
) -> None:
    """Generate Task 3.4 effective-spread tables and comparison figure."""

    _run_effective_spread_artifacts(_paths(processed_dir, figs_dir))


@app.command("run-all")
def run_all(
    processed_dir: Path = typer.Option(DEFAULT_PROCESSED_DIR, "--processed-dir"),
    figs_dir: Path = typer.Option(DEFAULT_FIGS_DIR, "--figs-dir"),
    skip_effective_spreads: bool = typer.Option(False, "--skip-effective-spreads"),
) -> None:
    """Generate all Module 3 tables and figures from existing parquet inputs."""

    paths = _paths(processed_dir, figs_dir)
    _run_simulation_artifacts(paths)
    if skip_effective_spreads:
        typer.echo("skipped Task 3.4 effective-spread artifacts")
        return
    _run_effective_spread_artifacts(paths)


if __name__ == "__main__":
    app()
