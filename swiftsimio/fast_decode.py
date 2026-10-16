"""
Read SWIFT snapshot datasets quickly by decoding their HDF5 chunks directly.

HDF5 undoes SWIFT's lossy compression slowly. Its N-bit filter, and the
conversion of SWIFT's reduced-precision floats back to standard IEEE floats,
work one element and one bit at a time, at a few tens of MB/s. Here each
chunk is read from the file as raw bytes and its filters are undone with
numba instead. The result is exactly the bytes that h5py would return.

The following filters are supported, on chunked numeric datasets of any
chunk shape. Together they cover everything SWIFT writes (see SWIFT's
``src/io_compression.c``, ``single_io.c``, ``distributed_io.c`` and
``common_io_cells.c``):

+ an optional lossy filter, which must come first:

  + N-bit, used for SWIFT's ``FMantissa*``, ``DMantissa*``, ``BFloat16`` and
    ``HalfFloat`` floats (any custom float layout, converted to a standard
    float exactly as HDF5 converts it) and ``Nbit*`` integers (sign-extended
    when signed);
  + scale-offset, used for SWIFT's ``DScale*`` floats (float32 and float64)
    and for integers;

+ then shuffle, deflate and Fletcher-32, each optional and in any order.

Big-endian data is decoded unless it has a lossy filter. A filter that was
skipped when a chunk was written is skipped here too, except for a skipped
lossy filter, in which case that chunk is read with h5py. Every chunk's
Fletcher-32 checksum is verified, as HDF5 does.

Virtual datasets are mapped onto their source files when each source supplies
whole rows to one contiguous block of the virtual dataset, as in SWIFT's
virtual snapshots. Rows that no source maps read as the fill value. Missing
source files and other mappings are read with h5py, which can also find
sources through ``HDF5_VDS_PREFIX``.

Anything not listed here is read with h5py, as before. To always read with
h5py, set the environment variable ``SWIFTSIMIO_FAST_DECODE=0`` or set
``swiftsimio.fast_decode.ENABLED = False``.

The numba kernels are compiled the first time they are used and cached on
disk. When starting many processes at once, especially from a read-only
installation, set ``NUMBA_CACHE_DIR`` to a writable directory and run one
read first so that the cache already exists. If the cache cannot be written
the kernels are compiled without it.
"""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
import functools
import itertools
import math
import os
import warnings
import zlib

import h5py
import numpy as np

from .optional_packages import jit as _jit, NUMBA_AVAILABLE
from ._handle_provider import open_hdf5

try:  # libdeflate decompresses about twice as fast as zlib
    import deflate as _libdeflate
except (ImportError, ModuleNotFoundError):
    _libdeflate = None

__all__ = [
    "ENABLED",
    "MINIMUM_ELEMENTS",
    "COLUMNS_CACHE_BYTES",
    "ChecksumError",
    "DecodePlan",
    "decoder_for",
    "read_ranges",
]

#: Whether to decode chunks here rather than with HDF5 (needs numba to be fast).
ENABLED = NUMBA_AVAILABLE and os.environ.get("SWIFTSIMIO_FAST_DECODE", "1") != "0"

#: Datasets with fewer elements than this are read with h5py, which is fast
#: enough for them. The size of the dataset matters rather than the size of
#: the read: HDF5 decodes every chunk that a read touches in full, so even a
#: read of a few rows of a large dataset costs whole chunks.
MINIMUM_ELEMENTS = 1 << 16

#: Multi-column fields larger than this many bytes (all columns together) are
#: read one column at a time instead of being kept for the other columns.
COLUMNS_CACHE_BYTES = 2 << 30

# HDF5's identifiers for the filters handled here.
FILTER_DEFLATE = 1
FILTER_SHUFFLE = 2
FILTER_FLETCHER32 = 3
FILTER_NBIT = 5
FILTER_SCALEOFFSET = 6

#: Size of the header of a scale-offset chunk; the packed values follow it.
SCALEOFFSET_HEADER_BYTES = 21

#: Chunks are read from the file in runs. Chunks no more than ``RUN_GAP``
#: bytes apart (SWIFT writes a dataset's chunks one after another) are read
#: together, in one request of at most ``RUN_BYTES`` bytes, instead of one
#: request per chunk.
RUN_GAP = 1 << 20
RUN_BYTES = 64 << 20

#: Number of validated chunk indexes kept between reads.
INDEX_CACHE_SIZE = 256


class ChecksumError(OSError):
    """A chunk's Fletcher-32 checksum does not match its data."""


# ---------------------------------------------------------------------------
# Numba kernels
# ---------------------------------------------------------------------------


def jit(**options: object) -> Callable:
    """
    Compile a kernel with numba, without disk caching if the cache is unusable.

    Parameters
    ----------
    **options : object
        Options passed to numba's ``jit`` decorator.

    Returns
    -------
    Callable
        The decorator. It retries without caching if numba cannot create the
        cache.
    """

    def decorate(function: Callable) -> Callable:
        """
        Compile one kernel.

        Parameters
        ----------
        function : Callable
            The Python function to compile.

        Returns
        -------
        Callable
            The compiled kernel.
        """
        try:
            return _jit(**options)(function)
        except (OSError, RuntimeError):
            return _jit(**dict(options, cache=False))(function)

    return decorate


@jit(nopython=True, cache=True)
def fletcher32(data: np.ndarray, length: int) -> np.uint32:
    """
    Compute HDF5's Fletcher-32 checksum (``H5_checksum_fletcher32``).

    The data is read as 16-bit big-endian words, and both sums are folded
    back into 16 bits after every 360 words.

    Parameters
    ----------
    data : np.ndarray[np.uint8]
        The bytes to check.
    length : int
        Number of bytes, from the start of ``data``, to include.

    Returns
    -------
    np.uint32
        The checksum.
    """
    sum1 = np.int64(0)
    sum2 = np.int64(0)
    position = 0
    words_left = length // 2
    while words_left > 0:
        block_words = 360 if words_left > 360 else words_left
        words_left -= block_words
        # HDF5 adds sum1 to sum2 after every word. Over a block that is
        # block_words * sum1, plus each word weighted by the number of sums
        # it is part of. Summing it this way gives the same integers as
        # HDF5's word-by-word loop, but each step no longer waits on the last.
        block_sum = np.int64(0)
        weighted_sum = np.int64(0)
        for j in range(block_words):
            word = (np.int64(data[position + 2 * j]) << 8) | np.int64(
                data[position + 2 * j + 1]
            )
            block_sum += word
            weighted_sum += (block_words - j) * word
        sum2 += block_words * sum1 + weighted_sum
        sum1 += block_sum
        position += 2 * block_words
        sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
        sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    if length % 2:
        sum1 += np.int64(data[position]) << 8
        sum2 += sum1
        sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
        sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
    sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    return np.uint32((sum2 << 16) | sum1)


@jit(nopython=True, cache=True)
def unshuffle(shuffled: np.ndarray, element_size: int, out: np.ndarray) -> None:
    """
    Undo HDF5's byte shuffle.

    The shuffle stores byte ``b`` of element ``e`` at ``shuffled[b * m + e]``,
    where ``m`` is the number of whole elements. Any leftover bytes at the end
    are stored unchanged.

    Parameters
    ----------
    shuffled : np.ndarray[np.uint8]
        The shuffled bytes.
    element_size : int
        Bytes per element.
    out : np.ndarray[np.uint8]
        Array of the same size as ``shuffled`` to hold the unshuffled bytes.
    """
    m = shuffled.size // element_size
    # The common element sizes are written out so that numba unrolls them.
    if element_size == 4:
        for e in range(m):
            o = 4 * e
            out[o] = shuffled[e]
            out[o + 1] = shuffled[m + e]
            out[o + 2] = shuffled[2 * m + e]
            out[o + 3] = shuffled[3 * m + e]
    elif element_size == 8:
        for e in range(m):
            o = 8 * e
            out[o] = shuffled[e]
            out[o + 1] = shuffled[m + e]
            out[o + 2] = shuffled[2 * m + e]
            out[o + 3] = shuffled[3 * m + e]
            out[o + 4] = shuffled[4 * m + e]
            out[o + 5] = shuffled[5 * m + e]
            out[o + 6] = shuffled[6 * m + e]
            out[o + 7] = shuffled[7 * m + e]
    else:
        for e in range(m):
            for b in range(element_size):
                out[e * element_size + b] = shuffled[b * m + e]
    for j in range(m * element_size, shuffled.size):
        out[j] = shuffled[j]


@jit(nopython=True, cache=True, inline="always")
def fmantissa9_at(stream: np.ndarray, index: int) -> np.uint32:
    """
    Unpack one 18-bit ``FMantissa9`` value into a float32 bit pattern.

    Parameters
    ----------
    stream : np.ndarray[np.uint8]
        The packed values, as a big-endian bit stream.
    index : int
        Index of the value to unpack.

    Returns
    -------
    np.uint32
        The float32 bit pattern.
    """
    bit = index * 18
    b = bit >> 3
    word = (
        (np.uint32(stream[b]) << 24)
        | (np.uint32(stream[b + 1]) << 16)
        | (np.uint32(stream[b + 2]) << 8)
    )
    if b + 3 < stream.size:
        word |= np.uint32(stream[b + 3])
    return ((word >> np.uint32(14 - (bit & 7))) & np.uint32(0x3FFFF)) << np.uint32(14)


@jit(nopython=True, cache=True)
def unpack_fmantissa9(
    stream: np.ndarray, out: np.ndarray, first: int, count: int
) -> None:
    """
    Unpack 18-bit ``FMantissa9`` values into float32 bit patterns.

    ``FMantissa9`` floats have a 1-bit sign, an 8-bit exponent and a 9-bit
    mantissa, so each becomes a float32 by shifting it left by 14 bits. Four
    values fill exactly 9 bytes, so most are unpacked four at a time with
    fixed shifts.

    Parameters
    ----------
    stream : np.ndarray[np.uint8]
        The packed values, as a big-endian bit stream.
    out : np.ndarray[np.uint32]
        Array to hold the float32 bit patterns.
    first : int
        Index of the first value to unpack.
    count : int
        Number of values to unpack.
    """
    i = 0
    # values before the first whole group of four
    while i < count and (first + i) & 3:
        out[i] = fmantissa9_at(stream, first + i)
        i += 1
    while i + 4 <= count:
        b = ((first + i) >> 2) * 9
        b0 = np.uint32(stream[b])
        b1 = np.uint32(stream[b + 1])
        b2 = np.uint32(stream[b + 2])
        b3 = np.uint32(stream[b + 3])
        b4 = np.uint32(stream[b + 4])
        b5 = np.uint32(stream[b + 5])
        b6 = np.uint32(stream[b + 6])
        b7 = np.uint32(stream[b + 7])
        b8 = np.uint32(stream[b + 8])
        out[i] = ((b0 << 10) | (b1 << 2) | (b2 >> 6)) << np.uint32(14)
        out[i + 1] = (
            ((b2 & np.uint32(0x3F)) << 12) | (b3 << 4) | (b4 >> 4)
        ) << np.uint32(14)
        out[i + 2] = (
            ((b4 & np.uint32(0x0F)) << 14) | (b5 << 6) | (b6 >> 2)
        ) << np.uint32(14)
        out[i + 3] = (((b6 & np.uint32(0x03)) << 16) | (b7 << 8) | b8) << np.uint32(14)
        i += 4
    # values after the last whole group of four
    while i < count:
        out[i] = fmantissa9_at(stream, first + i)
        i += 1


@jit(nopython=True, cache=True, inline="always")
def load_big_endian_word(stream: np.ndarray, start: np.uint64) -> np.uint64:
    """
    Read eight bytes as one big-endian 64-bit word.

    The index is unsigned so that numba does not check for negative indices,
    which would make these reads several times slower.

    Parameters
    ----------
    stream : np.ndarray[np.uint8]
        The bytes.
    start : np.uint64
        Index of the first of the eight bytes.

    Returns
    -------
    np.uint64
        The word.
    """
    return (
        (np.uint64(stream[start]) << np.uint64(56))
        | (np.uint64(stream[start + np.uint64(1)]) << np.uint64(48))
        | (np.uint64(stream[start + np.uint64(2)]) << np.uint64(40))
        | (np.uint64(stream[start + np.uint64(3)]) << np.uint64(32))
        | (np.uint64(stream[start + np.uint64(4)]) << np.uint64(24))
        | (np.uint64(stream[start + np.uint64(5)]) << np.uint64(16))
        | (np.uint64(stream[start + np.uint64(6)]) << np.uint64(8))
        | np.uint64(stream[start + np.uint64(7)])
    )


@jit(nopython=True, cache=True)
def word_loads(
    size: int, start_byte: int, precision: int, first: int, count: int
) -> int:
    """
    Count the values that can each be read with a single 8-byte load.

    These are the values of at most 57 bits whose eight bytes lie inside the
    stream, which is all but the last few.

    Parameters
    ----------
    size : int
        Number of bytes in the stream.
    start_byte : int
        Byte at which the packed values start.
    precision : int
        Bits per value.
    first : int
        Index of the first value.
    count : int
        Number of values.

    Returns
    -------
    int
        How many of the values, from the first, can be read with
        ``load_big_endian_word``.
    """
    if precision > 57 or size - 8 - start_byte < 0:
        return 0
    last_index = ((size - 8 - start_byte) * 8 + 7) // precision
    return max(0, min(count, last_index - first + 1))


@jit(nopython=True, cache=True)
def bits_at(
    stream: np.ndarray, start_byte: int, precision: int, index: int
) -> np.uint64:
    """
    Read one packed value of at most 64 bits, a byte at a time.

    This is used for the values that ``word_loads`` leaves out: those near the
    end of the stream, and those spread over nine bytes (more than 57 bits).

    Parameters
    ----------
    stream : np.ndarray[np.uint8]
        The packed values, as a big-endian bit stream.
    start_byte : int
        Byte at which the packed values start.
    precision : int
        Bits per value.
    index : int
        Index of the value to read.

    Returns
    -------
    np.uint64
        The value.
    """
    bit = index * precision
    b = start_byte + (bit >> 3)
    end = (bit & 7) + precision  # bits counted from the start of byte b
    top = np.uint64(0)
    if end > 64:  # the value spans nine bytes: keep the first one apart
        top = np.uint64(stream[b])
        b += 1
        end -= 8
    nbytes = (end + 7) >> 3
    word = np.uint64(0)
    for k in range(nbytes):
        word = word << np.uint64(8)
        if b + k < stream.size:
            word |= np.uint64(stream[b + k])
    right = np.uint64(nbytes * 8 - end)
    value = word >> right
    if top:
        value |= top << np.uint64(64 - right)
    return value & (np.uint64(0xFFFFFFFFFFFFFFFF) >> np.uint64(64 - precision))


@jit(nopython=True, cache=True)
def unpack_bits(
    stream: np.ndarray,
    start_byte: int,
    precision: int,
    out: np.ndarray,
    first: int,
    count: int,
) -> None:
    """
    Unpack values of at most 64 bits from a big-endian bit stream.

    Parameters
    ----------
    stream : np.ndarray[np.uint8]
        The packed values, as a big-endian bit stream.
    start_byte : int
        Byte at which the packed values start.
    precision : int
        Bits per value.
    out : np.ndarray[np.uint64]
        Array to hold the values.
    first : int
        Index of the first value to unpack.
    count : int
        Number of values to unpack.
    """
    fast = word_loads(stream.size, start_byte, precision, first, count)
    step = np.uint64(precision)
    right = np.uint64(64 - precision)
    base = np.uint64(start_byte)
    bit = np.uint64(first) * step
    for i in range(fast):
        word = load_big_endian_word(stream, base + (bit >> np.uint64(3)))
        out[i] = (word << (bit & np.uint64(7))) >> right
        bit += step
    for i in range(fast, count):
        out[i] = bits_at(stream, start_byte, precision, first + i)


@jit(nopython=True, cache=True)
def unpack_shifted_floats(
    stream: np.ndarray,
    precision: int,
    shift: int,
    exponent_mask: np.uint64,
    mantissa_mask: np.uint64,
    out: np.ndarray,
    first: int,
    count: int,
) -> None:
    """
    Unpack IEEE floats with shortened mantissas into full IEEE bit patterns.

    This is ``unpack_bits`` followed by a left shift of ``shift`` bits and
    ``canonical_nans``, done in a single pass.

    Parameters
    ----------
    stream : np.ndarray[np.uint8]
        The packed values, as a big-endian bit stream.
    precision : int
        Bits per value.
    shift : int
        Number of mantissa bits dropped when the values were written.
    exponent_mask : np.uint64
        Mask of the IEEE exponent bits.
    mantissa_mask : np.uint64
        Mask of the IEEE mantissa bits.
    out : np.ndarray[np.uint32] or np.ndarray[np.uint64]
        Array to hold the IEEE bit patterns.
    first : int
        Index of the first value to unpack.
    count : int
        Number of values to unpack.
    """
    fast = word_loads(stream.size, 0, precision, first, count)
    step = np.uint64(precision)
    right = np.uint64(64 - precision)
    left = np.uint64(shift)
    bit = np.uint64(first) * step
    for i in range(count):
        if i < fast:
            value = (
                load_big_endian_word(stream, bit >> np.uint64(3))
                << (bit & np.uint64(7))
            ) >> right
        else:
            value = bits_at(stream, 0, precision, first + i)
        bit += step
        word = value << left
        if (word & exponent_mask) == exponent_mask and (word & mantissa_mask) != 0:
            word |= mantissa_mask
        out[i] = word


@jit(nopython=True, cache=True)
def canonical_nans(
    words: np.ndarray, exponent_mask: np.uint64, mantissa_mask: np.uint64
) -> None:
    """
    Set every mantissa bit of NaNs, as HDF5's float conversion leaves them.

    Parameters
    ----------
    words : np.ndarray[np.uint32] or np.ndarray[np.uint64]
        IEEE bit patterns, changed in place.
    exponent_mask : np.uint64
        Mask of the exponent bits.
    mantissa_mask : np.uint64
        Mask of the mantissa bits.
    """
    for i in range(words.size):
        word = words[i]
        if (word & exponent_mask) == exponent_mask and (word & mantissa_mask) != 0:
            words[i] = word | mantissa_mask


@jit(nopython=True, cache=True)
def widen_float(
    bits: np.ndarray,
    sign_position: int,
    exponent_position: int,
    exponent_size: int,
    mantissa_position: int,
    mantissa_size: int,
    bias: int,
    target_exponent_size: int,
    target_mantissa_size: int,
    target_bias: int,
    out: np.ndarray,
) -> None:
    """
    Convert custom floats to a wider IEEE format, as HDF5 does.

    Zeros keep their sign, subnormal values are normalised, infinities stay
    infinite and NaNs get every mantissa bit set.

    Parameters
    ----------
    bits : np.ndarray[np.uint64]
        Bit patterns of the custom floats.
    sign_position : int
        Bit position of the custom float's sign.
    exponent_position : int
        Bit position of the custom float's exponent.
    exponent_size : int
        Number of exponent bits in the custom float.
    mantissa_position : int
        Bit position of the custom float's mantissa.
    mantissa_size : int
        Number of mantissa bits in the custom float.
    bias : int
        Exponent bias of the custom float.
    target_exponent_size : int
        Number of exponent bits in the IEEE format.
    target_mantissa_size : int
        Number of mantissa bits in the IEEE format.
    target_bias : int
        Exponent bias of the IEEE format.
    out : np.ndarray[np.uint64]
        Array to hold the IEEE bit patterns.
    """
    one = np.uint64(1)
    exponent_max = (one << np.uint64(exponent_size)) - one
    mantissa_mask = (one << np.uint64(mantissa_size)) - one
    target_exponent_max = (one << np.uint64(target_exponent_size)) - one
    target_mantissa_mask = (one << np.uint64(target_mantissa_size)) - one
    shift = np.uint64(target_mantissa_size - mantissa_size)
    for i in range(bits.size):
        value = bits[i]
        sign = (value >> np.uint64(sign_position)) & one
        exponent = (value >> np.uint64(exponent_position)) & exponent_max
        mantissa = (value >> np.uint64(mantissa_position)) & mantissa_mask
        if exponent == exponent_max:  # infinity or NaN
            new_exponent = target_exponent_max
            new_mantissa = np.uint64(0) if mantissa == 0 else target_mantissa_mask
        elif exponent == 0:
            if mantissa == 0:  # zero
                new_exponent = np.uint64(0)
                new_mantissa = np.uint64(0)
            else:  # subnormal: normalise it
                unbiased = 1 - bias
                while (mantissa >> np.uint64(mantissa_size)) == 0:
                    mantissa = mantissa << one
                    unbiased -= 1
                new_exponent = np.uint64(unbiased + target_bias)
                new_mantissa = (mantissa & mantissa_mask) << shift
        else:
            new_exponent = np.uint64(np.int64(exponent) - bias + target_bias)
            new_mantissa = mantissa << shift
        out[i] = (
            (sign << np.uint64(target_exponent_size + target_mantissa_size))
            | (new_exponent << np.uint64(target_mantissa_size))
            | new_mantissa
        )


@jit(nopython=True, cache=True)
def sign_extend(bits: np.ndarray, precision: int, out: np.ndarray) -> None:
    """
    Sign-extend two's complement values of ``precision`` bits to 64 bits.

    Parameters
    ----------
    bits : np.ndarray[np.uint64]
        The values.
    precision : int
        Bits per value.
    out : np.ndarray[np.uint64]
        Array to hold the sign-extended values (may be ``bits`` itself).
    """
    sign = np.uint64(1) << np.uint64(precision - 1)
    fill = (
        ~((np.uint64(1) << np.uint64(precision)) - np.uint64(1))
        if precision < 64
        else np.uint64(0)
    )
    for i in range(bits.size):
        value = bits[i]
        out[i] = value | fill if value & sign else value


@jit(nopython=True, cache=True)
def scaleoffset_header(stream: np.ndarray) -> tuple:
    """
    Read the header of a scale-offset chunk.

    Parameters
    ----------
    stream : np.ndarray[np.uint8]
        The chunk's bytes, after the other filters are undone.

    Returns
    -------
    tuple
        The number of bits per packed value and the bit pattern of the
        chunk's minimum.
    """
    bits_per_value = np.int64(0)
    for i in range(4):
        bits_per_value |= np.int64(stream[i]) << (8 * i)
    minimum = np.uint64(0)
    for i in range(8):
        minimum |= np.uint64(stream[5 + i]) << np.uint64(8 * i)
    return bits_per_value, minimum


@jit(nopython=True, cache=True)
def scaleoffset_float(
    stream: np.ndarray,
    bits_per_value: int,
    minimum: float,
    scale: float,
    fill_available: bool,
    fill: float,
    out: np.ndarray,
    first: int,
    count: int,
) -> None:
    """
    Decode D-scaled floats, as HDF5's ``H5Z_scaleoffset_modify_3`` does.

    Each packed value is read as a signed integer of the element's size,
    divided by ``10**D`` and added to the chunk's minimum, in the element's
    own precision. The all-ones value stands for the fill value.

    Parameters
    ----------
    stream : np.ndarray[np.uint8]
        The chunk's bytes, after the other filters are undone.
    bits_per_value : int
        Number of bits per packed value.
    minimum : float
        The chunk's minimum.
    scale : float
        The scale factor ``10**D``.
    fill_available : bool
        Whether the dataset has a fill value.
    fill : float
        The fill value.
    out : np.ndarray[np.float32] or np.ndarray[np.float64]
        Array to hold the values.
    first : int
        Index of the first value to decode.
    count : int
        Number of values to decode.
    """
    if bits_per_value == 0:
        for i in range(count):
            out[i] = minimum
        return
    header = SCALEOFFSET_HEADER_BYTES
    fill_marker = np.uint64(0xFFFFFFFFFFFFFFFF) >> np.uint64(64 - bits_per_value)
    fast = word_loads(stream.size, header, bits_per_value, first, count)
    step = np.uint64(bits_per_value)
    right = np.uint64(64 - bits_per_value)
    bit = np.uint64(first) * step
    for i in range(count):
        if i < fast:
            start = np.uint64(header) + (bit >> np.uint64(3))
            value = (
                load_big_endian_word(stream, start) << (bit & np.uint64(7))
            ) >> right
        else:
            value = bits_at(stream, header, bits_per_value, first + i)
        bit += step
        if fill_available and value == fill_marker:
            out[i] = fill
        elif out.itemsize == 4:
            out[i] = np.float32(np.int32(np.uint32(value))) / np.float32(
                scale
            ) + np.float32(minimum)
        else:
            out[i] = np.float64(np.int64(value)) / scale + minimum


# ---------------------------------------------------------------------------
# Deciding which datasets can be decoded
# ---------------------------------------------------------------------------


@dataclass
class DecodePlan:
    """
    How to decode the chunks of one dataset.

    Attributes
    ----------
    dtype : np.dtype
        The dataset's type, as stored in the file.
    stages : list
        The lossless filters as ``(name, parameter, position)``, in the order
        they were applied. ``name`` is ``"shuffle"``, ``"deflate"`` or
        ``"fletcher32"``; ``parameter`` is the element size for the shuffle;
        ``position`` is the filter's place in the dataset's pipeline, which
        is the bit set in a chunk's filter mask when the filter was skipped.
    kind : str
        How the values are stored once the lossless filters are undone:
        ``"plain"`` (as they are), ``"nbit_shift"`` (IEEE floats with a
        shortened mantissa), ``"nbit_float"`` (other custom floats),
        ``"nbit_int"``, ``"scaleoffset_float"`` or ``"scaleoffset_int"``.
    precision : int
        Bits per value, for the N-bit kinds.
    offset : int
        Bit offset of the value within an element, for the N-bit kinds.
    signed : bool
        Whether N-bit integers are signed.
    shift : int
        Number of mantissa bits dropped, for ``"nbit_shift"``.
    layout : tuple
        Sign position, exponent position, exponent size, mantissa position,
        mantissa size and exponent bias of the custom float, for
        ``"nbit_float"``.
    target : tuple
        Exponent size, mantissa size and exponent bias of the IEEE format to
        convert to, for ``"nbit_float"``.
    scale : float
        The scale factor ``10**D``, for ``"scaleoffset_float"``.
    fill_available : bool
        Whether the dataset has a fill value, for the scale-offset kinds.
    fill : object
        The fill value, for the scale-offset kinds.
    """

    dtype: np.dtype
    stages: list = field(default_factory=list)
    kind: str = "plain"
    precision: int = 0
    offset: int = 0
    signed: bool = False
    shift: int = 0
    layout: tuple = ()
    target: tuple = ()
    scale: float = 1.0
    fill_available: bool = False
    fill: object = None


def _filters(dataset: h5py.Dataset) -> list:
    """
    List a dataset's filters.

    Parameters
    ----------
    dataset : h5py.Dataset
        The dataset.

    Returns
    -------
    list
        Each filter's ``(identifier, flags, parameters)``, in the order they
        were applied when writing.
    """
    plist = dataset.id.get_create_plist()
    return [plist.get_filter(i)[:3] for i in range(plist.get_nfilters())]


def decoder_for(dataset: h5py.Dataset) -> DecodePlan | None:
    """
    Work out how to decode a dataset's chunks, if they can be decoded here.

    Every assumption that the kernels make about the data is checked here.

    Parameters
    ----------
    dataset : h5py.Dataset
        A dataset in a regular (not virtual) file.

    Returns
    -------
    DecodePlan or None
        How to decode the chunks, or ``None`` if the dataset should be read
        with h5py.
    """
    if dataset.chunks is None or dataset.dtype.kind not in "fiu":
        return None
    filters = _filters(dataset)
    identifiers = [identifier for identifier, _, _ in filters]
    lossy = bool(identifiers) and identifiers[0] in (FILTER_NBIT, FILTER_SCALEOFFSET)
    dtype = dataset.dtype
    if dtype.byteorder == ">" and lossy:
        return None  # the big-endian layouts of the lossy filters are not handled
    plan = DecodePlan(dtype=dtype)
    for position, (identifier, _, parameters) in enumerate(filters):
        if lossy and position == 0:
            continue
        if identifier == FILTER_SHUFFLE:
            plan.stages.append(("shuffle", parameters[0], position))
        elif identifier == FILTER_DEFLATE:
            plan.stages.append(("deflate", None, position))
        elif identifier == FILTER_FLETCHER32:
            plan.stages.append(("fletcher32", None, position))
        else:
            return None
    if not lossy:
        return plan
    parameters = filters[0][2]
    if identifiers[0] == FILTER_NBIT:
        return _plan_nbit(dataset, plan, parameters)
    return _plan_scaleoffset(plan, parameters)


def _plan_nbit(
    dataset: h5py.Dataset, plan: DecodePlan, parameters: tuple
) -> DecodePlan | None:
    """
    Complete the plan for a dataset written with the N-bit filter.

    Parameters
    ----------
    dataset : h5py.Dataset
        The dataset.
    plan : DecodePlan
        The plan so far, with the lossless filters filled in.
    parameters : tuple
        The N-bit filter's parameters, as HDF5 stores them.

    Returns
    -------
    DecodePlan or None
        The plan, or ``None`` if the dataset should be read with h5py.
    """
    dtype = dataset.dtype
    file_type = dataset.id.get_type()
    if len(parameters) != 8:
        return None
    if parameters[1] == 1:
        # The type uses every bit of its size, so HDF5 stored the values
        # without packing them.
        ieee = dtype.kind in "iu" or file_type.get_fields() == (
            (31, 23, 8, 0, 23) if dtype.itemsize == 4 else (63, 52, 11, 0, 52)
        )
        if not ieee or file_type.get_offset() != 0:
            return None
        return plan
    if parameters[1] != 0 or parameters[3] != 1 or parameters[5] != 0:
        return None  # only packed, single-type, little-endian data is handled
    precision, offset = file_type.get_precision(), file_type.get_offset()
    if parameters[6] != precision or parameters[7] != offset or precision + offset > 64:
        return None
    plan.precision, plan.offset = precision, offset
    if dtype.kind in "iu":
        plan.kind = "nbit_int"
        plan.signed = file_type.get_sign() == h5py.h5t.SGN_2
        return plan
    if dtype.itemsize not in (4, 8):
        return None
    (
        sign_position,
        exponent_position,
        exponent_size,
        mantissa_position,
        mantissa_size,
    ) = file_type.get_fields()
    target_exponent_size, target_mantissa_size, target_bias = (
        (8, 23, 127) if dtype.itemsize == 4 else (11, 52, 1023)
    )
    bias = file_type.get_ebias()
    highest_bit = max(
        sign_position,
        exponent_position + exponent_size - 1,
        mantissa_position + mantissa_size - 1,
    )
    if offset != 0 or highest_bit >= precision:
        return None
    if (
        file_type.get_norm() != h5py.h5t.NORM_IMPLIED
        or exponent_size > target_exponent_size
        or mantissa_size > target_mantissa_size
    ):
        return None
    if (
        (exponent_size, bias) == (target_exponent_size, target_bias)
        and mantissa_position == 0
        and exponent_position == mantissa_size
        and sign_position == exponent_position + exponent_size
    ):
        # an IEEE float with some mantissa bits dropped
        plan.kind = "nbit_shift"
        plan.shift = target_mantissa_size - mantissa_size
    else:
        plan.kind = "nbit_float"
        plan.layout = (
            sign_position,
            exponent_position,
            exponent_size,
            mantissa_position,
            mantissa_size,
            bias,
        )
        plan.target = (target_exponent_size, target_mantissa_size, target_bias)
    return plan


def _plan_scaleoffset(plan: DecodePlan, parameters: tuple) -> DecodePlan | None:
    """
    Complete the plan for a dataset written with the scale-offset filter.

    Parameters
    ----------
    plan : DecodePlan
        The plan so far, with the lossless filters filled in.
    parameters : tuple
        The scale-offset filter's parameters, as HDF5 stores them.

    Returns
    -------
    DecodePlan or None
        The plan, or ``None`` if the dataset should be read with h5py.
    """
    dtype = plan.dtype
    scale_type, scale_factor = parameters[0], parameters[1]
    type_class, size = parameters[3], parameters[4]
    byte_order, fill_available = parameters[6], parameters[7]
    if byte_order != 0 or size != dtype.itemsize:
        return None
    fill = np.zeros(1, dtype=dtype)
    if fill_available:
        # the fill value follows, packed into 4-byte parameters
        fill_words = parameters[8 : 8 + (size + 3) // 4]
        fill = np.frombuffer(
            np.array(fill_words, dtype=np.uint32).tobytes()[:size], dtype=dtype
        )
    if scale_type == 0 and type_class == 1 and dtype.kind == "f":
        plan.kind = "scaleoffset_float"
        plan.scale = 10.0**scale_factor
    elif scale_type == 2 and type_class == 0 and dtype.kind in "iu":
        plan.kind = "scaleoffset_int"
    else:
        return None
    plan.fill_available = bool(fill_available)
    plan.fill = fill[0]
    return plan


def _sources(handle: h5py.Dataset) -> list | None:
    """
    Map a dataset's rows onto the files that store them.

    Parameters
    ----------
    handle : h5py.Dataset
        The dataset, which may be virtual.

    Returns
    -------
    list or None
        ``(first row, end row, file name, dataset name)`` for each block of
        rows, or ``None`` if a virtual dataset's sources do not each supply
        whole rows to one contiguous block.
    """
    path = handle.file.filename
    if not handle.is_virtual:
        return [(0, handle.shape[0], path, handle.name)]
    directory = os.path.dirname(os.path.abspath(path))
    row_size = int(np.prod(handle.shape[1:]))
    blocks = []
    for source in handle.virtual_sources():
        bounds = source.vspace.get_select_bounds()
        if bounds is None:  # a source with no rows of this dataset
            continue
        low, high = bounds
        rows = high[0] + 1 - low[0]
        # the source must fill whole rows, and nothing but this block of rows
        whole_rows = all(
            low[k] == 0 and high[k] == handle.shape[k] - 1
            for k in range(1, handle.ndim)
        )
        if not whole_rows or source.vspace.get_select_npoints() != rows * row_size:
            return None
        selection = source.src_space
        if selection.get_select_type() != h5py.h5s.SEL_ALL:
            # part of the source: it must be whole rows from its first row
            if selection.get_select_type() != h5py.h5s.SEL_HYPERSLABS:
                return None
            if selection.get_select_npoints() != source.vspace.get_select_npoints():
                return None
            source_low, source_high = selection.get_select_bounds()
            expected_high = (rows - 1,) + tuple(n - 1 for n in handle.shape[1:])
            if source_low != (0,) * handle.ndim or source_high != expected_high:
                return None
        # (a selection of the whole source is checked against the source's
        # shape when the file is opened, in _read_unchecked)
        if source.file_name == ".":
            name = path
        elif os.path.isabs(source.file_name):
            name = source.file_name
        else:
            name = os.path.join(directory, source.file_name)
        blocks.append((low[0], low[0] + rows, name, source.dset_name))
    return blocks


# ---------------------------------------------------------------------------
# Decoding chunks
# ---------------------------------------------------------------------------


def _inflate(data: np.ndarray, size_bound: int) -> np.ndarray:
    """
    Undo the deflate filter.

    Parameters
    ----------
    data : np.ndarray[np.uint8]
        Compressed bytes.
    size_bound : int
        Upper bound on the decompressed size (needed by libdeflate).

    Returns
    -------
    np.ndarray[np.uint8]
        The decompressed bytes.
    """
    if _libdeflate is not None:
        return np.frombuffer(
            _libdeflate.zlib_decompress(data, size_bound), dtype=np.uint8
        )
    return np.frombuffer(zlib.decompress(data), dtype=np.uint8)


def _undo_lossless_filters(
    plan: DecodePlan,
    raw: bytes,
    values: int,
    scratch: np.ndarray,
    filter_mask: int,
) -> np.ndarray:
    """
    Undo the shuffle, deflate and Fletcher-32 filters of one chunk.

    Parameters
    ----------
    plan : DecodePlan
        How to decode the dataset's chunks.
    raw : bytes
        The chunk as stored in the file.
    values : int
        Number of values in a full chunk.
    scratch : np.ndarray[np.uint8]
        Reusable space for unshuffling; a new array is made if it is too
        small.
    filter_mask : int
        The chunk's filter mask: a set bit marks a filter skipped when the
        chunk was written.

    Returns
    -------
    np.ndarray[np.uint8]
        The chunk's bytes with the lossless filters undone.

    Raises
    ------
    ChecksumError
        If the chunk's Fletcher-32 checksum does not match its data.
    """
    data = np.frombuffer(raw, dtype=np.uint8)
    for name, parameter, position in reversed(plan.stages):
        if filter_mask >> position & 1:  # skipped when this chunk was written
            continue
        if name == "fletcher32":
            stored = int.from_bytes(data[-4:].tobytes(), "little")
            computed = int(fletcher32(data, data.size - 4))
            swapped = int.from_bytes(computed.to_bytes(4, "little"), "big")
            # HDF5 accepts the checksum in either byte order
            if stored != computed and stored != swapped:
                raise ChecksumError(
                    "Fletcher-32 checksum mismatch: the file may be corrupt"
                )
            data = data[:-4]
        elif name == "deflate":
            data = _inflate(data, values * plan.dtype.itemsize + 64)
        else:
            if scratch.size >= data.size:
                unshuffled = scratch[: data.size]
            else:
                unshuffled = np.empty(data.size, np.uint8)
            unshuffle(data, parameter, unshuffled)
            data = unshuffled
    return data


def _decode_values(
    plan: DecodePlan,
    data: np.ndarray,
    values: int,
    first: int,
    count: int,
    out: np.ndarray,
) -> None:
    """
    Decode values of one chunk whose lossless filters have been undone.

    Parameters
    ----------
    plan : DecodePlan
        How to decode the dataset's chunks.
    data : np.ndarray[np.uint8]
        The chunk's bytes, with the lossless filters undone.
    values : int
        Number of values in a full chunk.
    first : int
        Index of the first value to decode.
    count : int
        Number of values to decode.
    out : np.ndarray
        One-dimensional array to hold the values, in the dataset's type.

    Raises
    ------
    ValueError
        If the data is too short to hold the values.
    """
    dtype, kind = plan.dtype, plan.kind
    # Check the length first: the numba kernels do not check their indices.
    if kind == "plain":
        required = values * dtype.itemsize
    elif kind.startswith("nbit"):
        required = ((first + count) * plan.precision + 7) // 8
    else:
        if data.size < SCALEOFFSET_HEADER_BYTES:
            raise ValueError("truncated scale-offset header")
        bits_per_value, minimum_bits = scaleoffset_header(data)
        if not 0 <= bits_per_value <= dtype.itemsize * 8:
            raise ValueError("invalid scale-offset precision")
        packed_bytes = ((first + count) * bits_per_value + 7) // 8
        required = SCALEOFFSET_HEADER_BYTES + packed_bytes
    if data.size < required:
        raise ValueError("truncated decoded chunk")

    if kind == "plain":
        out[:] = data[: values * dtype.itemsize].view(dtype)[first : first + count]
    elif kind == "nbit_shift":
        _decode_shifted_floats(plan, data, first, count, out)
    elif kind == "nbit_float":
        bits = np.empty(count, dtype=np.uint64)
        unpack_bits(data, 0, plan.precision, bits, first, count)
        wide = np.empty(count, dtype=np.uint64)
        widen_float(bits, *plan.layout, *plan.target, wide)
        word = np.uint32 if dtype.itemsize == 4 else np.uint64
        out.view(word)[:] = wide.astype(word)
    elif kind == "nbit_int":
        # 8-byte integers are unpacked, and sign-extended, in place
        if dtype.itemsize == 8:
            bits = out.view(np.uint64)
        else:
            bits = np.empty(count, np.uint64)
        unpack_bits(data, 0, plan.precision, bits, first, count)
        if plan.signed:
            sign_extend(bits, plan.precision, bits)
        if dtype.itemsize != 8:
            out[:] = bits.astype(np.dtype(f"u{dtype.itemsize}")).view(dtype)
    else:
        _decode_scaleoffset(plan, data, bits_per_value, minimum_bits, first, count, out)


def _decode_shifted_floats(
    plan: DecodePlan, data: np.ndarray, first: int, count: int, out: np.ndarray
) -> None:
    """
    Decode IEEE floats stored with some of their mantissa bits dropped.

    Parameters
    ----------
    plan : DecodePlan
        How to decode the dataset's chunks.
    data : np.ndarray[np.uint8]
        The chunk's bytes, with the lossless filters undone.
    first : int
        Index of the first value to decode.
    count : int
        Number of values to decode.
    out : np.ndarray[np.float32] or np.ndarray[np.float64]
        Array to hold the values.
    """
    if plan.dtype.itemsize == 4:
        words = out.view(np.uint32)
        exponent_mask, mantissa_mask = 0x7F800000, 0x007FFFFF
        if plan.precision == 18 and plan.shift == 14:  # FMantissa9
            unpack_fmantissa9(data, words, first, count)
            canonical_nans(words, np.uint32(exponent_mask), np.uint32(mantissa_mask))
            return
    else:
        words = out.view(np.uint64)
        exponent_mask, mantissa_mask = 0x7FF0000000000000, 0x000FFFFFFFFFFFFF
    unpack_shifted_floats(
        data,
        plan.precision,
        plan.shift,
        np.uint64(exponent_mask),
        np.uint64(mantissa_mask),
        words,
        first,
        count,
    )


def _decode_scaleoffset(
    plan: DecodePlan,
    data: np.ndarray,
    bits_per_value: int,
    minimum_bits: int,
    first: int,
    count: int,
    out: np.ndarray,
) -> None:
    """
    Decode values stored with the scale-offset filter.

    Parameters
    ----------
    plan : DecodePlan
        How to decode the dataset's chunks.
    data : np.ndarray[np.uint8]
        The chunk's bytes, with the lossless filters undone.
    bits_per_value : int
        Number of bits per packed value, from the chunk's header.
    minimum_bits : int
        Bit pattern of the chunk's minimum, from the chunk's header.
    first : int
        Index of the first value to decode.
    count : int
        Number of values to decode.
    out : np.ndarray
        Array to hold the values.
    """
    dtype = plan.dtype
    size = dtype.itemsize
    word = np.dtype(f"u{size}")
    element_mask = (1 << (8 * size)) - 1
    minimum = np.array([int(minimum_bits) & element_mask], dtype=word).view(dtype)[0]
    if plan.kind == "scaleoffset_float":
        scaleoffset_float(
            data,
            bits_per_value,
            minimum,
            plan.scale,
            plan.fill_available,
            plan.fill,
            out,
            first,
            count,
        )
    elif bits_per_value == 0:
        out[:] = minimum
    else:
        # (packed value + minimum), wrapped to the element's size as HDF5's
        # C cast does
        packed = np.empty(count, dtype=np.uint64)
        unpack_bits(
            data, SCALEOFFSET_HEADER_BYTES, bits_per_value, packed, first, count
        )
        wrapped = (packed + np.uint64(minimum_bits)) & np.uint64(element_mask)
        result = wrapped.astype(word).view(dtype)
        if plan.fill_available:
            fill_marker = np.uint64((1 << bits_per_value) - 1)
            result = np.where(packed == fill_marker, plan.fill, result)
        out[:] = result


def _decode_chunk(
    plan: DecodePlan,
    raw: bytes,
    values: int,
    first: int,
    count: int,
    out: np.ndarray,
    scratch: np.ndarray,
    filter_mask: int = 0,
) -> None:
    """
    Decode some of the values of one chunk.

    Parameters
    ----------
    plan : DecodePlan
        How to decode the dataset's chunks.
    raw : bytes
        The chunk as stored in the file.
    values : int
        Number of values in a full chunk.
    first : int
        Index of the first value to decode.
    count : int
        Number of values to decode.
    out : np.ndarray
        One-dimensional array to hold the values, in the dataset's type.
    scratch : np.ndarray[np.uint8]
        Reusable space for unshuffling.
    filter_mask : int, optional
        The chunk's filter mask. Default is ``0`` (no filter skipped).
    """
    data = _undo_lossless_filters(plan, raw, values, scratch, filter_mask)
    _decode_values(plan, data, values, first, count, out)


def _convert(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Convert values to another numeric type, as HDF5 does.

    Floats are rounded and integers are clamped to the new type's range.

    Parameters
    ----------
    values : np.ndarray
        The values.
    dtype : np.dtype
        The type to convert to.

    Returns
    -------
    np.ndarray
        The converted values (``values`` itself if no conversion is needed).

    Raises
    ------
    TypeError
        If there is no such conversion here.
    """
    dtype = np.dtype(dtype)
    if values.dtype == dtype:
        return values
    if dtype.kind == "f":
        return values.astype(dtype)
    if values.dtype.kind in "iu" and dtype.kind in "iu":
        info = np.iinfo(dtype)
        if values.dtype.kind == "u":
            return np.minimum(values, np.uint64(info.max)).astype(dtype)
        source = np.iinfo(values.dtype)
        return np.clip(
            values, max(info.min, source.min), min(info.max, source.max)
        ).astype(dtype)
    raise TypeError(f"no conversion from {values.dtype} to {dtype}")


# ---------------------------------------------------------------------------
# Reading from the file
# ---------------------------------------------------------------------------


def _path_identity(path: str) -> tuple | None:
    """
    Identify a file by its path, as ``_file_identity`` identifies an open file.

    Parameters
    ----------
    path : str
        The file's path.

    Returns
    -------
    tuple or None
        The file's device, inode, modification time and size, or ``None`` if
        the file cannot be found.
    """
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size


def _file_identity(file: h5py.File) -> tuple | None:
    """
    Identify an open file in a way that survives closing and reopening it.

    swiftsimio opens a snapshot again for every field, so caches keyed on the
    open file would never be reused. ``fstat`` on HDF5's own file descriptor
    gives the file's device, inode, modification time and size in one call,
    without looking up the path (on Lustre, that would cost a request to the
    metadata server for every directory in the path).

    Parameters
    ----------
    file : h5py.File
        The open file.

    Returns
    -------
    tuple or None
        The file's device, inode, modification time and size, or ``None`` for
        files without a descriptor (such as files in memory) or open for
        writing.
    """
    if file.mode != "r" or file.driver != "sec2":
        return None
    stat = os.fstat(file.id.get_vfd_handle())
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size


def _pread(fd: int, size: int, offset: int, exact: bool = True) -> bytes:
    """
    Read bytes from a file at a given offset.

    ``os.pread`` may return fewer bytes than asked for, so it is called until
    all of them have been read.

    Parameters
    ----------
    fd : int
        The file's descriptor.
    size : int
        Number of bytes to read.
    offset : int
        Offset of the first byte in the file.
    exact : bool, optional
        If ``True`` (the default), it is an error to reach the end of the
        file first. If ``False``, fewer bytes are returned instead.

    Returns
    -------
    bytes
        The bytes.

    Raises
    ------
    OSError
        If the end of the file is reached before ``size`` bytes with
        ``exact=True``.
    """
    parts, done = [], 0
    while done < size:
        part = os.pread(fd, size - done, offset + done)
        if not part:
            if exact:
                raise OSError("unexpected end of file")
            break
        parts.append(part)
        done += len(part)
    return parts[0] if len(parts) == 1 else b"".join(parts)


def _read_regions(fd: int, regions: list) -> dict:
    """
    Read regions of a file, merging nearby regions into single requests.

    Parameters
    ----------
    fd : int
        The file's descriptor.
    regions : list
        ``(offset, size)`` of each region.

    Returns
    -------
    dict
        Each region's bytes (a ``memoryview``, shorter if the region runs past
        the end of the file), by its offset.
    """
    out = {}
    regions = sorted(set(regions))
    i = 0
    while i < len(regions):
        start = regions[i][0]
        end = start + regions[i][1]
        j = i + 1
        while (
            j < len(regions)
            and regions[j][0] - end <= RUN_GAP
            and regions[j][0] + regions[j][1] - start <= RUN_BYTES
        ):
            end = max(end, regions[j][0] + regions[j][1])
            j += 1
        data = memoryview(_pread(fd, end - start, start, exact=False))
        for offset, size in regions[i:j]:
            out[offset] = data[offset - start : offset - start + size]
        i = j
    return out


# ---------------------------------------------------------------------------
# Reading chunk indexes
#
# HDF5 finds a chunk by searching the dataset's chunk index (a B-tree), one
# search per chunk. On a busy file system each search costs several small
# reads. Instead, the whole index is read here in a few large reads. The
# layouts below are from the HDF5 file format specification.
# ---------------------------------------------------------------------------

# Object header message types used here.
MESSAGE_LAYOUT = 0x08
MESSAGE_CONTINUATION = 0x10


def _layout_message(body: bytes) -> tuple | None:
    """
    Read the chunk index's address from a data layout message.

    Parameters
    ----------
    body : bytes
        The body of the message.

    Returns
    -------
    tuple or None
        The address of the chunk index (a version 1 B-tree) and the number of
        dimensions of the layout (the dataset's plus one), or ``None`` if this
        is not a version 3 chunked layout (for example, a version 4 layout,
        whose chunk indexes are not read here).
    """
    if len(body) < 11 or body[0] != 3 or body[1] != 2:
        return None
    return int.from_bytes(body[3:11], "little"), body[2]


def _continuation(body: bytes) -> tuple:
    """
    Read where the messages of an object header continue.

    Parameters
    ----------
    body : bytes
        The body of a continuation message.

    Returns
    -------
    tuple
        The address and size of the next block of messages.
    """
    return int.from_bytes(body[0:8], "little"), int.from_bytes(body[8:16], "little")


def _layout_btree(fd: int, base: int, header: int) -> tuple | None:
    """
    Find the chunk index of a chunked dataset from its object header.

    Version 1 and version 2 object headers are read, following continuation
    messages to the header's other blocks.

    Parameters
    ----------
    fd : int
        The file's descriptor.
    base : int
        The file's base address (the size of its user block).
    header : int
        Address of the dataset's object header.

    Returns
    -------
    tuple or None
        The address of the chunk index and the number of dimensions of the
        layout, or ``None`` for any other header or layout.
    """
    prefix = _pread(fd, 64, base + header, exact=False)
    if prefix[:4] == b"OHDR":
        return _layout_btree_v2(fd, base, header, prefix)
    if prefix[0] == 1:
        return _layout_btree_v1(fd, base, header, prefix)
    return None


def _layout_btree_v1(fd: int, base: int, header: int, prefix: bytes) -> tuple | None:
    """
    Find the chunk index in a version 1 object header.

    Parameters
    ----------
    fd : int
        The file's descriptor.
    base : int
        The file's base address (the size of its user block).
    header : int
        Address of the dataset's object header.
    prefix : bytes
        The first bytes of the object header.

    Returns
    -------
    tuple or None
        As for ``_layout_btree``.
    """
    messages = int.from_bytes(prefix[2:4], "little")
    blocks = [(header + 16, int.from_bytes(prefix[8:12], "little"))]
    seen = 0
    while blocks and seen < messages:
        address, size = blocks.pop(0)
        data = _pread(fd, size, base + address)
        position = 0
        while position + 8 <= size and seen < messages:
            kind = int.from_bytes(data[position : position + 2], "little")
            length = int.from_bytes(data[position + 2 : position + 4], "little")
            body = data[position + 8 : position + 8 + length]
            seen += 1
            if kind == MESSAGE_CONTINUATION:
                blocks.append(_continuation(body))
            elif kind == MESSAGE_LAYOUT:
                return _layout_message(body)
            position += 8 + length
    return None


def _layout_btree_v2(fd: int, base: int, header: int, prefix: bytes) -> tuple | None:
    """
    Find the chunk index in a version 2 (``OHDR``) object header.

    Parameters
    ----------
    fd : int
        The file's descriptor.
    base : int
        The file's base address (the size of its user block).
    header : int
        Address of the dataset's object header.
    prefix : bytes
        The first bytes of the object header.

    Returns
    -------
    tuple or None
        As for ``_layout_btree``.
    """
    if prefix[4] != 2:
        return None
    flags = prefix[5]
    # The size of the first block follows the optional times and attribute
    # limits, in 1, 2, 4 or 8 bytes.
    position = 6 + (16 if flags & 0x20 else 0) + (4 if flags & 0x10 else 0)
    width = 1 << (flags & 0x03)
    size = int.from_bytes(prefix[position : position + width], "little")
    blocks = [(header + position + width, size)]
    # each message has a creation order number if the flags say so
    message_header = 6 if flags & 0x04 else 4
    while blocks:
        address, size = blocks.pop(0)
        data = _pread(fd, size, base + address)
        position = 0
        if data[:4] == b"OCHK":  # a continuation block
            position = 4
            size -= 4  # the block's checksum
        while position + message_header <= size:
            kind = data[position]
            length = int.from_bytes(data[position + 1 : position + 3], "little")
            body_start = position + message_header
            body = data[body_start : body_start + length]
            if kind == MESSAGE_CONTINUATION:
                blocks.append(_continuation(body))
            elif kind == MESSAGE_LAYOUT:
                return _layout_message(body)
            position = body_start + length
    return None


class _ChunkIndex:
    """
    Locations of the chunks of one dataset, held as sorted arrays.

    Building a dict of every chunk costs microseconds of Python per chunk,
    which adds up to seconds for a snapshot with hundreds of thousands of
    chunks, though a masked read needs only a few of them. Here each chunk is
    numbered by its position in the grid of chunks (in row-major order, so
    numbers sort the same way as coordinates), and the chunks of a read are
    found with one ``np.searchsorted``.

    Parameters
    ----------
    chunks : tuple
        The dataset's chunk shape.
    coords : np.ndarray
        Each chunk's coordinates, with shape ``(number of chunks, ndim)``.
        Each coordinate must be a multiple of the chunk shape.
    offsets : np.ndarray
        Each chunk's offset in the file.
    sizes : np.ndarray
        Each chunk's size in the file.
    masks : np.ndarray
        Each chunk's filter mask.
    """

    def __init__(
        self,
        chunks: tuple,
        coords: np.ndarray,
        offsets: np.ndarray,
        sizes: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        self.chunks = np.asarray(chunks, dtype=np.int64)
        if (coords % self.chunks).any():
            raise ValueError("chunk coordinates are not on the chunk grid")
        grid = coords // self.chunks
        extent = (grid.max(axis=0) + 1).tolist() if len(grid) else [1] * len(chunks)
        if math.prod(extent) >= 2**62:
            raise ValueError("chunk grid too large to number")
        strides = [math.prod(extent[d + 1 :]) for d in range(len(extent))]
        self.extent = np.array(extent, dtype=np.int64)
        self.strides = np.array(strides, dtype=np.int64)
        self.numbers = grid @ self.strides
        self.offsets, self.sizes, self.masks = offsets, sizes, masks
        # the leaves of a B-tree list the chunks in order, so this is rare
        if (np.diff(self.numbers) <= 0).any():
            order = np.argsort(self.numbers, kind="stable")
            self.numbers = self.numbers[order]
            if (np.diff(self.numbers) == 0).any():
                raise ValueError("a chunk is listed twice")
            self.offsets = offsets[order]
            self.sizes = sizes[order]
            self.masks = masks[order]

    def __len__(self) -> int:
        """
        Count the chunks in the index.

        Returns
        -------
        int
            The number of chunks.
        """
        return len(self.numbers)

    def coords(self, i: int) -> tuple:
        """
        Return the coordinates of a chunk.

        Parameters
        ----------
        i : int
            Position of the chunk, in coordinate order.

        Returns
        -------
        tuple
            The chunk's coordinates.
        """
        grid, number = [], int(self.numbers[i])
        for stride in self.strides.tolist():
            grid.append(number // stride)
            number %= stride
        return tuple(int(g) * int(c) for g, c in zip(grid, self.chunks))

    def get_many(self, coords: list) -> list:
        """
        Look up chunks by their coordinates.

        Parameters
        ----------
        coords : list
            Coordinates of the chunks.

        Returns
        -------
        list
            ``(offset, size, filter mask)`` of each chunk, or ``None`` for
            chunks not in the index.
        """
        if not len(self.numbers):
            return [None] * len(coords)
        grid = np.asarray(coords, dtype=np.int64).reshape(len(coords), -1)
        aligned = (grid % self.chunks == 0).all(axis=1)
        grid //= self.chunks
        inside = aligned & ((grid >= 0) & (grid < self.extent)).all(axis=1)
        numbers = np.where(inside, grid @ self.strides, -1)
        at = np.minimum(np.searchsorted(self.numbers, numbers), len(self.numbers) - 1)
        found = inside & (self.numbers[at] == numbers)
        offsets = self.offsets[at].tolist()
        sizes = self.sizes[at].tolist()
        masks = self.masks[at].tolist()
        return [
            (offsets[i], sizes[i], masks[i]) if hit else None
            for i, hit in enumerate(found.tolist())
        ]

    def items(self) -> list:
        """
        List every chunk in the index.

        Returns
        -------
        list
            ``(coordinates, (offset, size, filter mask))`` of each chunk, in
            coordinate order.
        """
        return [
            (self.coords(i), info)
            for i, info in enumerate(
                zip(self.offsets.tolist(), self.sizes.tolist(), self.masks.tolist())
            )
        ]


@functools.lru_cache
def _btree_entry(ndim: int) -> np.dtype:
    """
    Describe one entry of a version 1 B-tree node of chunks as a NumPy type.

    Each entry is a key (the chunk's stored size, its filter mask and its
    coordinates, followed by an element offset that is always 0) and then the
    address of the child node, or of the chunk in a leaf.

    Parameters
    ----------
    ndim : int
        The dataset's number of dimensions.

    Returns
    -------
    np.dtype
        The structured type of one entry.
    """
    key = 16 + 8 * ndim
    return np.dtype(
        {
            "names": ["size", "mask", "coords", "child"],
            "formats": ["<u4", "<u4", ("<u8", (ndim,)), "<u8"],
            "offsets": [0, 4, 8, key],
            "itemsize": key + 8,
        }
    )


def _chunk_index(dataset: h5py.Dataset, fd: int) -> _ChunkIndex | None:
    """
    Read the location of every written chunk from a dataset's B-tree.

    The tree is read one level at a time. The nodes of each level are merged
    into a few large reads, and the entries of each node are read as one
    NumPy array.

    Parameters
    ----------
    dataset : h5py.Dataset
        A chunked dataset in a regular file.
    fd : int
        The descriptor of the dataset's file.

    Returns
    -------
    _ChunkIndex or None
        Every chunk's file offset, size and filter mask, or ``None`` if the
        dataset's chunks are not indexed this way, or anything looks
        unexpected. The chunks are then looked up one at a time instead.
    """
    file, ndim = dataset.file, dataset.ndim
    plist = file.id.get_create_plist()
    if plist.get_sizes() != (8, 8):  # 8-byte addresses and lengths
        return None
    base = plist.get_userblock()
    found = _layout_btree(fd, base, h5py.h5o.get_info(dataset.id).addr)
    if found is None:
        return None
    root, layout_dimensions = found
    if layout_dimensions != ndim + 1:
        return None
    key = 8 + 8 * layout_dimensions  # chunk size, filter mask, coordinates
    entry = _btree_entry(ndim)
    # the largest node expected: 128 entries (HDF5's default is 64)
    largest = 24 + 129 * key + 128 * 8
    leaves, level, nodes = [], None, [root]
    while nodes:
        regions = _read_regions(fd, [(base + node, largest) for node in nodes])
        children = []
        for node in nodes:
            data = regions[base + node]
            if len(data) < 24 or bytes(data[0:4]) != b"TREE" or data[4] != 1:
                return None
            if level is not None and data[5] != level:
                return None
            level, entries = data[5], int.from_bytes(data[6:8], "little")
            if 24 + (entries + 1) * key + entries * 8 > len(data):
                return None  # a node larger than expected
            table = np.frombuffer(data, dtype=entry, count=entries, offset=24)
            if level > 0:
                children.extend(table["child"].tolist())
            else:
                leaves.append(table)
        nodes = children
        if children:
            level -= 1
    table = np.concatenate(leaves) if leaves else np.empty(0, dtype=entry)
    if (table["coords"] >= 2**63).any():
        return None
    return _ChunkIndex(
        dataset.chunks,
        table["coords"].astype(np.int64).reshape(len(table), ndim),
        table["child"].astype(np.int64) + base,
        table["size"].astype(np.int64),
        table["mask"].astype(np.int64),
    )


#: Validated chunk indexes of read-only files, most recently used last.
_index_cache = OrderedDict()


@dataclass
class _ChunkRun:
    """
    Adjacent chunks read from the file in one request.

    Attributes
    ----------
    offset : int
        Offset of the run's first byte in the file.
    length : int
        Number of bytes in the run.
    chunks_left : int
        Number of the run's chunks not yet handed out.
    data : bytes or None
        The run's bytes once read, dropped when every chunk is handed out.
    """

    offset: int
    length: int = 0
    chunks_left: int = 0
    data: bytes | None = None


class _ChunkReader:
    """
    Read the raw chunks of one dataset, in runs of adjacent chunks.

    Each ``read_direct_chunk`` call is a separate request to the file system.
    On a busy Lustre file system each costs milliseconds, which dominates the
    read time for datasets of many small chunks. Here the chunks' offsets in
    the file are looked up once, and each run of adjacent chunks is read with
    one ``os.pread``. The bytes are the same as ``read_direct_chunk`` gives,
    which is checked on the first chunk. Chunks that cannot be located this
    way are read one at a time with ``read_direct_chunk``.

    Parameters
    ----------
    source : h5py.Dataset
        A dataset in a regular (not virtual) file.
    coords : list
        Coordinates of the chunks that will be read, in any order.
    """

    def __init__(self, source: h5py.Dataset, coords: list) -> None:
        self.source = source
        self.fd = None
        # chunk coordinates -> (run number, start in run, size, filter mask)
        self.where = {}
        # chunks that were never written, which read as the fill value
        self.unallocated = set()
        self.runs = []
        if not coords:
            return
        self.cache_key = None
        self.index = None
        self.validated = False
        if not hasattr(os, "pread") or source.file.driver != "sec2":
            return
        # borrow HDF5's own descriptor rather than opening the file again
        self.fd = source.file.id.get_vfd_handle()
        locations = self._locate(coords)
        self.unallocated = {c for c, info in locations.items() if info is None}
        locations = {c: info for c, info in locations.items() if info is not None}
        self._plan_runs(locations)
        if locations and not self.validated:
            self._validate(next(iter(locations)))

    def _plan_runs(self, locations: dict) -> None:
        """
        Group the chunks into runs to read in one request each.

        Parameters
        ----------
        locations : dict
            ``(offset, size, filter mask)`` of each chunk, by its coordinates.
        """
        for coord in sorted(locations, key=lambda c: locations[c][0]):
            offset, size, filter_mask = locations[coord]
            run = self.runs[-1] if self.runs else None
            if run is None or not (
                0 <= offset - (run.offset + run.length) <= RUN_GAP
                and offset + size - run.offset <= RUN_BYTES
            ):
                run = _ChunkRun(offset)
                self.runs.append(run)
            run.length = offset + size - run.offset
            run.chunks_left += 1
            run_number = len(self.runs) - 1
            self.where[coord] = (run_number, offset - run.offset, size, filter_mask)

    def _validate(self, coord: tuple) -> None:
        """
        Check one chunk against ``read_direct_chunk``, then cache the index.

        Parameters
        ----------
        coord : tuple
            Coordinates of the chunk to check.

        Raises
        ------
        OSError
            If the chunk's bytes or filter mask differ from HDF5's.
        """
        try:
            mask, raw = self.read(coord)
            if (mask, bytes(raw)) != tuple(self.source.id.read_direct_chunk(coord)):
                raise OSError("chunk offsets do not match the chunks")
            # read() counted this chunk as handed out, but it will be read again
            self.runs[self.where[coord][0]].chunks_left += 1
            if self.cache_key is not None and self.index is not None:
                _index_cache[self.cache_key] = self.index
                _index_cache.move_to_end(self.cache_key)
                while len(_index_cache) > INDEX_CACHE_SIZE:
                    _index_cache.popitem(last=False)
        except OSError:
            self.close()
            raise

    def _ask_hdf5(self, coord: tuple) -> tuple | None:
        """
        Look up one chunk with HDF5.

        Parameters
        ----------
        coord : tuple
            Coordinates of the chunk.

        Returns
        -------
        tuple or None
            The chunk's ``(offset, size, filter mask)``, or ``None`` if it was
            never written.
        """
        info = self.source.id.get_chunk_info_by_coord(coord)
        if info.byte_offset is None:
            return None
        return info.byte_offset, info.size, info.filter_mask

    def _locate(self, coords: list) -> dict:
        """
        Find where the chunks are in the file.

        The dataset's chunk index is read directly where possible, in a few
        large reads. Its first and last entries are checked against HDF5, and
        a chunk missing from it is confirmed with HDF5. Otherwise HDF5 is
        asked for each chunk.

        Parameters
        ----------
        coords : list
            Coordinates of the chunks.

        Returns
        -------
        dict
            ``(offset, size, filter mask)`` of each chunk by its coordinates,
            or ``None`` for chunks that were never written.
        """
        source = self.source
        identity = _file_identity(source.file)
        if identity is not None:
            self.cache_key = identity + (source.name,)
            if self.cache_key in _index_cache:
                self.validated = True
                self.index = _index_cache[self.cache_key]
                _index_cache.move_to_end(self.cache_key)
                return self._lookup(coords)
        try:
            index = _chunk_index(source, self.fd)
            if index:
                ends = [index.coords(0), index.coords(len(index) - 1)]
                if any(
                    self._ask_hdf5(c) != info
                    for c, info in zip(ends, index.get_many(ends))
                ):
                    index = None  # not what HDF5 finds: ask HDF5 for each chunk
        except (OSError, ValueError, IndexError, AttributeError, RuntimeError):
            index = None  # for example, an index layout not read here
        if not index:
            return {c: self._ask_hdf5(c) for c in coords}
        self.index = index
        return self._lookup(coords)

    def _lookup(self, coords: list) -> dict:
        """
        Find chunks in the index, confirming any missing ones with HDF5.

        Parameters
        ----------
        coords : list
            Coordinates of the chunks.

        Returns
        -------
        dict
            As for ``_locate``.
        """
        coords = list(coords)
        return {
            c: self._ask_hdf5(c) if info is None else info
            for c, info in zip(coords, self.index.get_many(coords))
        }

    def read(self, coord: tuple) -> tuple:
        """
        Read one chunk as stored in the file, like ``read_direct_chunk``.

        Parameters
        ----------
        coord : tuple
            Coordinates of the chunk.

        Returns
        -------
        tuple
            The chunk's filter mask and its bytes. The bytes are ``None`` for
            a chunk that was never written.
        """
        if coord in self.unallocated:
            return 0, None
        if coord not in self.where:
            return self.source.id.read_direct_chunk(coord)
        run_number, start, size, filter_mask = self.where[coord]
        run = self.runs[run_number]
        if run.data is None:
            run.data = _pread(self.fd, run.length, run.offset)
        raw = memoryview(run.data)[start : start + size]  # a view, not a copy
        run.chunks_left -= 1
        if run.chunks_left == 0:
            run.data = None  # every chunk of the run has been handed out
        return filter_mask, raw

    def close(self) -> None:
        """Stop using the borrowed descriptor, leaving HDF5's file open."""
        self.fd = None


# ---------------------------------------------------------------------------
# Reading rows
# ---------------------------------------------------------------------------


def _all_columns(columns: object) -> bool:
    """
    Check whether a column selection selects every column (``[:]``).

    Parameters
    ----------
    columns : object
        The column selection.

    Returns
    -------
    bool
        Whether every column is selected.
    """
    return isinstance(columns, slice) and columns == slice(None)


def _copy_pieces(
    source: h5py.Dataset, pieces: list, output: np.ndarray, columns: object
) -> None:
    """
    Copy pieces of a regular dataset's rows, with columns selected, to output.

    Each chunk is read and decoded once, however many pieces it holds.

    Parameters
    ----------
    source : h5py.Dataset
        A dataset in a regular (not virtual) file.
    pieces : list
        ``(first row, end row, output row)`` of each piece.
    output : np.ndarray
        The output array, with rows along its first axis.
    columns : object
        Column selection, used only if the dataset has more than one
        dimension.
    """
    plan = decoder_for(source)
    if plan is None:
        for start, stop, at in pieces:
            rows = source[start:stop]
            output[at : at + stop - start] = _convert(
                np.ascontiguousarray(
                    rows[(slice(None), columns)] if source.ndim > 1 else rows
                ),
                output.dtype,
            )
        return
    chunk_rows, chunk_tail = source.chunks[0], tuple(source.chunks[1:])
    tail = tuple(source.shape[1:])
    whole_rows = chunk_tail == tail  # SWIFT's layout: each chunk holds whole rows
    row_values = int(np.prod(chunk_tail)) if source.ndim > 1 else 1
    values = chunk_rows * row_values
    native = source.dtype.newbyteorder("=")
    scratch = np.empty(values * source.dtype.itemsize + 64, dtype=np.uint8)
    decoded = np.empty(values, dtype=native)
    # The chunks across the other dimensions, leaving out those that hold
    # none of the selected columns.
    wanted = np.zeros(tail[0] if tail else 1, dtype=bool)
    if tail:
        wanted[columns] = True
    else:
        wanted[:] = True
    grid = [
        corner
        for corner in itertools.product(
            *(range(0, n, c) for n, c in zip(tail, chunk_tail))
        )
        if whole_rows or wanted[corner[0] : corner[0] + chunk_tail[0]].any()
    ]
    lossy = plan.kind != "plain"
    # decode straight into the output when it is exactly the decoded rows
    direct = (
        whole_rows
        and output.dtype == native
        and (source.ndim == 1 or _all_columns(columns))
    )
    # the pieces in each block of chunk rows
    by_chunk = {}
    for start, stop, at in pieces:
        for k in range(start // chunk_rows, (stop - 1) // chunk_rows + 1):
            low, high = max(start, k * chunk_rows), min(stop, (k + 1) * chunk_rows)
            by_chunk.setdefault(k, []).append((low, high, at + low - start))
    reader = _ChunkReader(
        source, [(k * chunk_rows,) + corner for k in by_chunk for corner in grid]
    )
    span_buffer = None
    try:
        for k in sorted(by_chunk):
            parts = by_chunk[k]
            span_start = min(low for low, _, _ in parts)
            span_stop = max(high for _, high, _ in parts)
            rows = span_stop - span_start
            straight_to_output = direct and len(parts) == 1
            if straight_to_output:
                block = output[parts[0][2] : parts[0][2] + rows]
            else:
                if span_buffer is None:
                    span_buffer = np.empty((chunk_rows,) + tail, dtype=native)
                block = span_buffer[:rows]
            for corner in grid:
                inside = tuple(
                    slice(o, min(o + c, n)) for o, c, n in zip(corner, chunk_tail, tail)
                )
                filter_mask, raw = reader.read((k * chunk_rows,) + corner)
                if raw is None:  # never written: HDF5 reads its fill value
                    block[(slice(None),) + inside] = np.array(source.fillvalue).astype(
                        native
                    )
                    continue
                if lossy and filter_mask & 1:
                    # written without its lossy filter: let HDF5 convert it
                    block[(slice(None),) + inside] = source[
                        (slice(span_start, span_stop),) + inside
                    ]
                    continue
                flat = block.reshape(-1) if whole_rows else decoded[: rows * row_values]
                _decode_chunk(
                    plan,
                    raw,
                    values,
                    (span_start - k * chunk_rows) * row_values,
                    rows * row_values,
                    flat,
                    scratch,
                    filter_mask,
                )
                if not whole_rows:
                    filled = tuple(slice(0, q.stop - q.start) for q in inside)
                    block[(slice(None),) + inside] = flat.reshape((rows,) + chunk_tail)[
                        (slice(None),) + filled
                    ]
            if straight_to_output:
                continue
            for low, high, at in parts:
                piece = block[low - span_start : high - span_start]
                if source.ndim > 1:
                    piece = piece[(slice(None), columns)]
                output[at : at + high - low] = _convert(np.asarray(piece), output.dtype)
    finally:
        reader.close()


#: The last multi-column field read for one of its columns, with every column
#: decoded. The field's other columns (swiftsimio's named columns) are then
#: copied from here instead of decoding every chunk again. It is dropped once
#: every column has been read, or when another field takes its place.
_columns_cache = {"key": None, "data": None, "served": None, "sources": ()}

#: The source files of the last virtual dataset read, with their identities.
_last_sources = ()


def read_ranges(
    handle: h5py.Dataset,
    ranges: np.ndarray,
    output_shape: tuple,
    output_type: type,
    columns: object = np.s_[:],
) -> np.ndarray | None:
    """
    Read ranges of rows of a dataset by decoding its chunks directly.

    Parameters
    ----------
    handle : h5py.Dataset
        The dataset, which may be virtual.
    ranges : np.ndarray
        ``[start, stop)`` of each range of rows.
    output_shape : tuple
        Shape of the result.
    output_type : type
        :class:`numpy.dtype` of the result.
    columns : object, optional
        Column selection for datasets with more than one dimension. Default
        is every column.

    Returns
    -------
    np.ndarray or None
        The rows, identical to what h5py gives, or ``None`` if this dataset
        should be read with h5py instead (the caller then does so).
    """
    if not ENABLED or handle.ndim == 0 or handle.dtype.kind not in "fiu":
        return None
    ranges = np.asarray(ranges, dtype=np.int64).reshape(-1, 2)
    dtype = np.dtype(output_type).newbyteorder("=")
    if handle.size < MINIMUM_ELEMENTS:
        return None
    if handle.ndim == 2 and not _all_columns(columns):
        return _read_columns(handle, ranges, dtype, columns)
    return _read(handle, ranges, np.empty(output_shape, dtype=dtype), columns)


def _read_columns(
    handle: h5py.Dataset, ranges: np.ndarray, dtype: np.dtype, columns: object
) -> np.ndarray | None:
    """
    Read some columns of the rows, decoding every column once for all of them.

    Parameters
    ----------
    handle : h5py.Dataset
        A two-dimensional dataset, which may be virtual.
    ranges : np.ndarray
        ``[start, stop)`` of each range of rows.
    dtype : np.dtype
        Type of the result.
    columns : object
        Column selection.

    Returns
    -------
    np.ndarray or None
        The selected columns of the rows, or ``None`` if h5py should read
        them instead.
    """
    global _last_sources
    rows = int((ranges[:, 1] - ranges[:, 0]).sum())
    width = handle.shape[1]
    identity = _file_identity(handle.file)
    if rows * width * dtype.itemsize > COLUMNS_CACHE_BYTES:
        selected_shape = np.empty((0, width))[:, columns].shape[1:]
        output = np.empty((rows,) + selected_shape, dtype)
        return _read(handle, ranges, output, columns)
    # The file is identified as a whole (for a virtual file, by its own
    # identity: SWIFT's sub-files do not change while the virtual file stays
    # the same). Files in memory are identified by their open handle.
    key = (
        identity if identity is not None else ("open", handle.file.id.fileno),
        handle.name,
        dtype.str,
        ranges.tobytes(),
    )
    cache = _columns_cache
    if cache["key"] == key and not all(
        _path_identity(path) == source_identity
        for path, source_identity in cache["sources"]
    ):
        cache["key"] = None  # a source file of the virtual dataset has changed
    if cache["key"] != key:
        # free the old field before reading the new one
        cache.update(key=None, data=None, served=None, sources=())
        _last_sources = ()
        data = _read(handle, ranges, np.empty((rows, width), dtype), np.s_[:])
        if data is None:
            return None
        cache.update(
            key=key,
            data=data,
            served=np.zeros(width, dtype=bool),
            sources=_last_sources if handle.is_virtual else (),
        )
    result = np.ascontiguousarray(cache["data"][:, columns])
    cache["served"][columns] = True
    if cache["served"].all():
        cache.update(key=None, data=None, served=None, sources=())
    return result


def _with_gaps(blocks: list, rows: int) -> list | None:
    """
    Add the blocks of rows that no source of a virtual dataset maps.

    Parameters
    ----------
    blocks : list
        ``(first row, end row, file name, dataset name)`` of each block.
    rows : int
        Number of rows in the virtual dataset.

    Returns
    -------
    list or None
        Every block, sorted by first row, with the unmapped blocks given the
        file name ``None``; or ``None`` if two sources map the same rows.
    """
    blocks = sorted(blocks)
    covered, gaps = 0, []
    for start, stop, _, _ in blocks:
        if start < covered:
            return None
        if start > covered:
            gaps.append((covered, start, None, None))
        covered = stop
    if covered < rows:
        gaps.append((covered, rows, None, None))
    return sorted(blocks + gaps, key=lambda block: block[0])


def _split_ranges(ranges: np.ndarray, blocks: list) -> tuple:
    """
    Split ranges of rows between the blocks of rows that store them.

    Parameters
    ----------
    ranges : np.ndarray
        ``[start, stop)`` of each range of rows.
    blocks : list
        ``(first row, end row, file name, dataset name)`` of each block,
        sorted by first row.

    Returns
    -------
    tuple
        A dict of each block's ``(first row, end row, output row)`` pieces,
        with rows counted from the block's first row; and a list of
        ``(output row, number of rows)`` for each piece that no source maps.
    """
    pieces = {block: [] for block in blocks}
    unmapped = []
    position = 0
    for low, high in ranges:
        for block in blocks:
            start, stop, file_name, _ = block
            a, b = max(low, start), min(high, stop)
            if a >= b:
                continue
            if file_name is None:
                unmapped.append((position, b - a))
            else:
                pieces[block].append((a - start, b - start, position))
            position += b - a
    return pieces, unmapped


def _read_unchecked(
    handle: h5py.Dataset, ranges: np.ndarray, output: np.ndarray, columns: object
) -> np.ndarray | None:
    """
    Read ranges of rows into output, letting any error propagate.

    Parameters
    ----------
    handle : h5py.Dataset
        The dataset, which may be virtual.
    ranges : np.ndarray
        ``[start, stop)`` of each range of rows.
    output : np.ndarray
        Array to hold the rows.
    columns : object
        Column selection for datasets with more than one dimension.

    Returns
    -------
    np.ndarray or None
        ``output``, or ``None`` if h5py should read the rows instead.
    """
    global _last_sources
    if not handle.is_virtual:
        # Read through this handle: its file may not be on disk under its
        # name (for example, a file in memory), so it is never reopened.
        pieces, position = [], 0
        for low, high in ranges:
            if high > low:
                pieces.append((low, high, position))
            position += max(high - low, 0)
        _copy_pieces(handle, pieces, output, columns)
        return output
    blocks = _sources(handle)
    if blocks is None:
        return None
    # Rows that no source maps read as the fill value. Sources that map the
    # same rows are left to h5py.
    blocks = _with_gaps(blocks, handle.shape[0])
    if blocks is None:
        return None
    if len(ranges):  # only the files that the ranges touch
        blocks = [
            block
            for block in blocks
            if any(low < block[1] and high > block[0] for low, high in ranges)
        ]
    files = {}
    try:
        for start, stop, file_name, dataset_name in blocks:
            if file_name is None:
                continue
            if file_name not in files:
                try:
                    files[file_name] = open_hdf5(file_name)
                except FileNotFoundError:
                    # HDF5 may still find it through HDF5_VDS_PREFIX
                    return None
            source = files[file_name][dataset_name]
            if source.shape[0] < stop - start or source.shape[1:] != handle.shape[1:]:
                return None  # the source does not hold the rows this block maps
        _last_sources = tuple(
            (file_name, _file_identity(file)) for file_name, file in files.items()
        )
        pieces, unmapped = _split_ranges(ranges, blocks)
        for position, rows in unmapped:
            fill = np.array(handle.fillvalue).astype(output.dtype)
            output[position : position + rows] = fill
        for (_, _, file_name, dataset_name), block_pieces in pieces.items():
            if block_pieces:
                source = files[file_name][dataset_name]
                _copy_pieces(source, block_pieces, output, columns)
    finally:
        for file in files.values():
            file.close()
    return output


def _read(
    handle: h5py.Dataset, ranges: np.ndarray, output: np.ndarray, columns: object
) -> np.ndarray | None:
    """
    Read ranges of rows into output, leaving unexpected failures to h5py.

    Checksum errors are always raised, since h5py would raise them too.

    Parameters
    ----------
    handle : h5py.Dataset
        The dataset, which may be virtual.
    ranges : np.ndarray
        ``[start, stop)`` of each range of rows.
    output : np.ndarray
        Array to hold the rows.
    columns : object
        Column selection for datasets with more than one dimension.

    Returns
    -------
    np.ndarray or None
        ``output``, or ``None`` if h5py should read the rows instead.
    """
    try:
        return _read_unchecked(handle, ranges, output, columns)
    except ChecksumError:
        raise
    except Exception as error:
        warnings.warn(
            f"Fast decode of {handle.name} failed: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
