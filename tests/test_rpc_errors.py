import pytest

from shared.rpc import (
    RateLimitError,
    RpcClient,
    RpcConfig,
    RpcTransientError,
    TooManyResultsError,
    _validate_log_response,
)


def _rpc(tmp_path):
    return RpcClient(
        RpcConfig(url="https://example.com", rate_limit_rps=None),
        data_dir=tmp_path,
    )


def test_rpc_config_rejects_placeholder_urls(monkeypatch):
    monkeypatch.setenv("RPC_URL", "your_alchemy_url")

    with pytest.raises(RuntimeError, match="full http"):
        RpcConfig.from_env()


@pytest.mark.parametrize(
    "message",
    [
        "cannot parse json-rpc response: upstream connect error",
        "Custom error: All RPCs are unreachable and no Alchemy fallback is configured.",
    ],
)
def test_provider_upstream_failures_are_transient(tmp_path, message):
    with pytest.raises(RpcTransientError):
        _rpc(tmp_path)._parse_rpc_result(
            {"error": {"message": message}},
            "https://example.com",
        )


def test_too_many_requests_is_rate_limit(tmp_path):
    with pytest.raises(RateLimitError):
        _rpc(tmp_path)._parse_rpc_result(
            {"error": {"message": "Too many requests"}},
            "https://example.com",
        )


def test_too_many_results_is_chunk_split_signal(tmp_path):
    with pytest.raises(TooManyResultsError):
        _rpc(tmp_path)._parse_rpc_result(
            {"error": {"message": "query returned more than 10000 results"}},
            "https://example.com",
        )


def test_implausible_log_index_is_transient_provider_corruption():
    with pytest.raises(RpcTransientError, match="implausible logIndex"):
        _validate_log_response(
            [{"logIndex": "0xfffffffc"}],
            "https://example.com",
        )
