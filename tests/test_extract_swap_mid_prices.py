from __future__ import annotations

import json
from decimal import Decimal

import pandas as pd
from eth_abi import encode as abi_encode

from il_risk.pipelines.module1.compact import extract_swap_mid_prices
from il_risk.uniswap_v3.math import get_sqrt_ratio_at_tick


class FakeRpc:
    def __init__(self) -> None:
        self.called_blocks: list[int] = []
        self.batch_sizes: list[int] = []

    def call(self, to: str, data: bytes, block: int | str = "latest") -> bytes:
        assert isinstance(block, int)
        self.called_blocks.append(block)
        return self._slot0_response(block)

    def call_many(self, calls, *, batch_size: int = 100):
        call_list = list(calls)
        self.batch_sizes.append(len(call_list))
        out = []
        for _to, _data, block in call_list:
            assert isinstance(block, int)
            self.called_blocks.append(block)
            out.append(self._slot0_response(block))
        return out

    def _slot0_response(self, block: int) -> bytes:
        sqrt_price_x96 = get_sqrt_ratio_at_tick(200_000 + block % 10)
        return abi_encode(
            ["uint160", "int24", "uint16", "uint16", "uint16", "uint8", "bool"],
            [sqrt_price_x96, 200_000 + block % 10, 1, 2, 3, 0, True],
        )

    def get_block(self, number: int, *, full_transactions: bool = False) -> dict:
        return {"timestamp": hex(1_700_000_000 + number)}


def test_extract_swap_mid_prices_fetches_previous_block_per_unique_swap_block(tmp_path) -> None:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        return

    processed = tmp_path / "processed"
    processed.mkdir()
    swaps_path = processed / "swap_events.parquet"
    output_path = processed / "swap_mid_prices.parquet"
    pd.DataFrame(
        [
            {"block_number": 100},
            {"block_number": 100},
            {"block_number": 102},
        ]
    ).to_parquet(swaps_path, index=False)

    rpc = FakeRpc()
    result = extract_swap_mid_prices(
        rpc,  # type: ignore[arg-type]
        data_dir=tmp_path,
        swap_events_path=swaps_path,
        output_path=output_path,
        batch_size=10,
    )

    assert rpc.called_blocks == [99, 101]
    assert rpc.batch_sizes == [2]
    assert output_path.exists()
    assert result["block_number"].tolist() == [100, 102]
    assert result["pre_swap_block"].tolist() == [99, 101]
    assert result["mid_price_usdc_per_weth"].gt(0).all()
    assert result["sqrt_price_x96"].map(lambda x: isinstance(x, Decimal)).all()

    resumed_rpc = FakeRpc()
    resumed = extract_swap_mid_prices(
        resumed_rpc,  # type: ignore[arg-type]
        data_dir=tmp_path,
        swap_events_path=swaps_path,
        output_path=output_path,
        batch_size=10,
    )
    assert resumed_rpc.called_blocks == []
    assert resumed["block_number"].tolist() == [100, 102]


def test_extract_swap_mid_prices_can_sample_reproducible_strata(tmp_path) -> None:
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        return

    processed = tmp_path / "processed"
    processed.mkdir()
    swaps_path = processed / "swap_events.parquet"
    output_path = processed / "swap_mid_prices.parquet"
    rows = []
    for block in range(100, 130):
        rows.append(
            {
                "block_number": block,
                "date": "2026-01-01" if block < 115 else "2026-01-02",
                "trade_direction": "buy_weth" if block % 2 else "sell_weth",
                "usd_notional": 1_000 if block % 3 else 100_000,
            }
        )
    pd.DataFrame(rows).to_parquet(swaps_path, index=False)

    rpc = FakeRpc()
    result = extract_swap_mid_prices(
        rpc,  # type: ignore[arg-type]
        data_dir=tmp_path,
        swap_events_path=swaps_path,
        output_path=output_path,
        batch_size=5,
        sample_blocks=12,
        sample_seed=7,
    )

    assert len(result) == 12
    assert result["block_number"].is_unique
    assert rpc.called_blocks == [block - 1 for block in result["block_number"]]
    metadata = json.loads(output_path.with_suffix(".metadata.json").read_text())
    assert metadata["usd_notional_bucket_edges"][-1] == "Infinity"
