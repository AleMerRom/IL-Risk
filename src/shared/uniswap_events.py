"""Uniswap V3 pool event signatures and topic0 hashes.

Event topic0 = keccak256(signature). Computed at import time with ``eth_utils``
so the values are auditable against etherscan if needed.
"""

from eth_utils import keccak


def _topic0(sig: str) -> str:
    return "0x" + keccak(text=sig).hex()


SWAP_SIG = "Swap(address,address,int256,int256,uint160,uint128,int24)"
MINT_SIG = "Mint(address,address,int24,int24,uint128,uint256,uint256)"
BURN_SIG = "Burn(address,int24,int24,uint128,uint256,uint256)"
COLLECT_SIG = "Collect(address,address,int24,int24,uint128,uint128)"

SWAP_TOPIC0 = _topic0(SWAP_SIG)
MINT_TOPIC0 = _topic0(MINT_SIG)
BURN_TOPIC0 = _topic0(BURN_SIG)
COLLECT_TOPIC0 = _topic0(COLLECT_SIG)
