"""Port of Uniswap v3-core TickMath.sol.

``get_sqrt_ratio_at_tick(tick)`` returns sqrt(1.0001^tick) * 2^96 as a uint160.
``get_tick_at_sqrt_ratio(sqrtPriceX96)`` is the inverse.

Line-by-line audit-able against v3-core/contracts/libraries/TickMath.sol.
All arithmetic is pure Python int — no floats, no numpy.
"""

MIN_TICK: int = -887272
MAX_TICK: int = 887272

MIN_SQRT_RATIO: int = 4295128739
MAX_SQRT_RATIO: int = 1461446703485210103287273052203988822378723970342

_UINT256_MAX = (1 << 256) - 1


def get_sqrt_ratio_at_tick(tick: int) -> int:
    abs_tick = -tick if tick < 0 else tick
    if abs_tick > MAX_TICK:
        raise ValueError(f"tick {tick} out of range")

    ratio = 0xFFFCB933BD6FAD37AA2D162D1A594001 if (abs_tick & 0x1) != 0 else 0x100000000000000000000000000000000
    if abs_tick & 0x2:
        ratio = (ratio * 0xFFF97272373D413259A46990580E213A) >> 128
    if abs_tick & 0x4:
        ratio = (ratio * 0xFFF2E50F5F656932EF12357CF3C7FDCC) >> 128
    if abs_tick & 0x8:
        ratio = (ratio * 0xFFE5CACA7E10E4E61C3624EAA0941CD0) >> 128
    if abs_tick & 0x10:
        ratio = (ratio * 0xFFCB9843D60F6159C9DB58835C926644) >> 128
    if abs_tick & 0x20:
        ratio = (ratio * 0xFF973B41FA98C081472E6896DFB254C0) >> 128
    if abs_tick & 0x40:
        ratio = (ratio * 0xFF2EA16466C96A3843EC78B326B52861) >> 128
    if abs_tick & 0x80:
        ratio = (ratio * 0xFE5DEE046A99A2A811C461F1969C3053) >> 128
    if abs_tick & 0x100:
        ratio = (ratio * 0xFCBE86C7900A88AEDCFFC83B479AA3A4) >> 128
    if abs_tick & 0x200:
        ratio = (ratio * 0xF987A7253AC413176F2B074CF7815E54) >> 128
    if abs_tick & 0x400:
        ratio = (ratio * 0xF3392B0822B70005940C7A398E4B70F3) >> 128
    if abs_tick & 0x800:
        ratio = (ratio * 0xE7159475A2C29B7443B29C7FA6E889D9) >> 128
    if abs_tick & 0x1000:
        ratio = (ratio * 0xD097F3BDFD2022B8845AD8F792AA5825) >> 128
    if abs_tick & 0x2000:
        ratio = (ratio * 0xA9F746462D870FDF8A65DC1F90E061E5) >> 128
    if abs_tick & 0x4000:
        ratio = (ratio * 0x70D869A156D2A1B890BB3DF62BAF32F7) >> 128
    if abs_tick & 0x8000:
        ratio = (ratio * 0x31BE135F97D08FD981231505542FCFA6) >> 128
    if abs_tick & 0x10000:
        ratio = (ratio * 0x9AA508B5B7A84E1C677DE54F3E99BC9) >> 128
    if abs_tick & 0x20000:
        ratio = (ratio * 0x5D6AF8DEDB81196699C329225EE604) >> 128
    if abs_tick & 0x40000:
        ratio = (ratio * 0x2216E584F5FA1EA926041BEDFE98) >> 128
    if abs_tick & 0x80000:
        ratio = (ratio * 0x48A170391F7DC42444E8FA2) >> 128

    if tick > 0:
        ratio = _UINT256_MAX // ratio

    # From Solidity: shift right 32 to convert Q128.128 -> Q64.96, rounding up.
    return (ratio >> 32) + (0 if ratio % (1 << 32) == 0 else 1)


def get_tick_at_sqrt_ratio(sqrt_price_x96: int) -> int:
    if sqrt_price_x96 < MIN_SQRT_RATIO or sqrt_price_x96 >= MAX_SQRT_RATIO:
        raise ValueError(f"sqrtPriceX96 {sqrt_price_x96} out of range")

    ratio = sqrt_price_x96 << 32

    r = ratio
    msb = 0
    for shift, mask in [
        (7, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF),
        (6, 0xFFFFFFFFFFFFFFFF),
        (5, 0xFFFFFFFF),
        (4, 0xFFFF),
        (3, 0xFF),
        (2, 0xF),
        (1, 0x3),
        (0, 0x1),
    ]:
        f = (1 if r > mask else 0) << shift
        msb |= f
        r >>= f

    r = ratio >> (msb - 127) if msb >= 128 else ratio << (127 - msb)

    log_2 = (msb - 128) << 64

    for i in range(14):
        r = (r * r) >> 127
        f = r >> 128
        log_2 |= f << (63 - i)
        r >>= f

    log_sqrt10001 = log_2 * 255738958999603826347141  # 128.128-bit

    tick_low = (log_sqrt10001 - 3402992956809132418596140100660247210) >> 128
    tick_high = (log_sqrt10001 + 291339464771989622907027621153398088495) >> 128

    # Solidity uses int256; ensure signed interpretation.
    def _to_int256(x: int) -> int:
        x &= (1 << 256) - 1
        return x - (1 << 256) if x >= (1 << 255) else x

    tick_low = _to_int256(tick_low)
    tick_high = _to_int256(tick_high)

    if tick_low == tick_high:
        return tick_low
    return tick_high if get_sqrt_ratio_at_tick(tick_high) <= sqrt_price_x96 else tick_low
