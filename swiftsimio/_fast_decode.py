"""
Fast reads of SWIFT snapshot datasets, decoding HDF5 chunks without HDF5.

HDF5 undoes SWIFT's lossy compression slowly: its N-bit filter and the
conversion of SWIFT's reduced-precision floats back to IEEE work element by
element and bit by bit, at a few tens of MB/s. Here every chunk is read raw
with ``read_direct_chunk`` and its filters are undone in numba instead, giving
exactly the bytes h5py would return.

Supported (chunked numeric datasets, any chunk shape), covering everything
SWIFT writes (``src/io_compression.c``, ``single_io.c``, ``distributed_io.c``,
``common_io_cells.c``):

+ an optional first filter:

  + N-bit: SWIFT's ``FMantissa*``, ``DMantissa*``, ``BFloat16``, ``HalfFloat``
    (any custom float layout, widened exactly as HDF5 converts it) and
    ``Nbit*`` (integers, sign-extended when signed);
  + scale-offset: SWIFT's ``DScale*`` (float32 and float64) and integers;

+ then shuffle, deflate and Fletcher-32 in any order, each optional.

Big-endian data is decoded unless it has a lossy filter. Filters skipped when
a chunk was written are skipped here too, except a skipped lossy filter (that
chunk is read with h5py). Everything else is read with h5py as before. Every
chunk's Fletcher-32 checksum is verified, as HDF5 does. Virtual datasets are
mapped onto their source files when each source supplies whole rows to one
contiguous block, as in SWIFT's virtual snapshots; unmapped rows read as
the fill value. Missing source files and unsupported mappings go to h5py,
which can resolve sources through HDF5_VDS_PREFIX.

Set ``NUMBA_CACHE_DIR`` to a writable directory and warm the kernel cache
once before starting many concurrent processes, especially for read-only
installations. Cache or compilation failures fall back to h5py.

Set the environment variable ``SWIFTSIMIO_FAST_DECODE=0``, or
``swiftsimio._fast_decode.ENABLED = False``, to always read with h5py.
"""

from collections import OrderedDict
from collections.abc import Callable
import itertools
import os
import zlib
import warnings

import h5py
import numpy as np

from .optional_packages import jit as _jit, NUMBA_AVAILABLE
from ._handle_provider import open_hdf5

try:  # libdeflate inflates about twice as fast as zlib
    import deflate as _libdeflate
except (ImportError, ModuleNotFoundError):
    _libdeflate = None

__all__ = ["ENABLED", "MINIMUM_ELEMENTS", "ChecksumError", "read_ranges"]

#: Decode chunks here rather than with HDF5 (needs numba to be fast).
ENABLED = NUMBA_AVAILABLE and os.environ.get("SWIFTSIMIO_FAST_DECODE", "1") != "0"

#: Datasets of fewer elements than this go to h5py, which is fast enough
#: there. (It is the dataset that counts, not the read: HDF5 decodes every
#: chunk a read touches whole, so even a few rows cost it whole chunks.)
MINIMUM_ELEMENTS = 1 << 16

NBIT, SHUFFLE, DEFLATE, FLETCHER32, SCALEOFFSET = 5, 2, 1, 3, 6


class ChecksumError(OSError):
    """A chunk's Fletcher-32 checksum does not match its data."""


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


def jit(**options: object) -> Callable:
    """
    Build a kernel without disk caching if numba cannot create its cache.

    Parameters
    ----------
    **options : object
        Options passed to numba's jit decorator.

    Returns
    -------
    Callable
        The decorator, retrying without caching if the cache is unavailable.
    """

    def decorate(function: Callable) -> Callable:
        try:
            return _jit(**options)(function)
        except (OSError, RuntimeError):
            return _jit(**dict(options, cache=False))(function)

    return decorate


@jit(nopython=True, cache=True)
def fletcher32(data: np.ndarray, n: int) -> np.uint32:
    """
    HDF5's ``H5_checksum_fletcher32`` of ``data[:n]``.

    Parameters
    ----------
    data : np.ndarray
        Bytes (uint8).

    n : int
        Number of bytes to checksum.

    Returns
    -------
    np.uint32
        The checksum: 16-bit big-endian words, sums folded every 360 words.
    """
    sum1 = np.int64(0)
    sum2 = np.int64(0)
    i = 0
    words = n // 2
    while words > 0:
        tlen = 360 if words > 360 else words
        words -= tlen
        # sum2 gains sum1 after every word, i.e. tlen * sum1 plus each word
        # weighted by how many sums it is in: the same integers as HDF5's
        # word-by-word loop, without its serial dependency
        block = np.int64(0)
        weighted = np.int64(0)
        for j in range(tlen):
            w = (np.int64(data[i + 2 * j]) << 8) | np.int64(data[i + 2 * j + 1])
            block += w
            weighted += (tlen - j) * w
        sum2 += tlen * sum1 + weighted
        sum1 += block
        i += 2 * tlen
        sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
        sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    if n % 2:
        sum1 += np.int64(data[i]) << 8
        sum2 += sum1
        sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
        sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    sum1 = (sum1 & 0xFFFF) + (sum1 >> 16)
    sum2 = (sum2 & 0xFFFF) + (sum2 >> 16)
    return np.uint32((sum2 << 16) | sum1)


@jit(nopython=True, cache=True)
def unshuffle(buf: np.ndarray, element_size: int, out: np.ndarray) -> None:
    """
    Undo HDF5's byte shuffle.

    Byte ``b`` of element ``e`` is at ``buf[b * m + e]`` for the ``m`` whole
    elements; leftover tail bytes are stored as they are.

    Parameters
    ----------
    buf : np.ndarray
        Shuffled bytes.

    element_size : int
        Bytes per element.

    out : np.ndarray
        Unshuffled bytes (same size as ``buf``).
    """
    m = buf.size // element_size
    if element_size == 4:
        for e in range(m):
            o = 4 * e
            out[o] = buf[e]
            out[o + 1] = buf[m + e]
            out[o + 2] = buf[2 * m + e]
            out[o + 3] = buf[3 * m + e]
    elif element_size == 8:
        for e in range(m):
            o = 8 * e
            out[o] = buf[e]
            out[o + 1] = buf[m + e]
            out[o + 2] = buf[2 * m + e]
            out[o + 3] = buf[3 * m + e]
            out[o + 4] = buf[4 * m + e]
            out[o + 5] = buf[5 * m + e]
            out[o + 6] = buf[6 * m + e]
            out[o + 7] = buf[7 * m + e]
    else:
        for e in range(m):
            for b in range(element_size):
                out[e * element_size + b] = buf[b * m + e]
    for j in range(m * element_size, buf.size):
        out[j] = buf[j]


@jit(nopython=True, cache=True)
def unpack_nbit18(stream: np.ndarray, out: np.ndarray, first: int, count: int) -> None:
    """
    Unpack 18-bit ``FMantissa9`` values into IEEE float32 bit patterns.

    Four values fill exactly 9 bytes, so they are unpacked four at a time with
    fixed shifts. The 1/8/9-bit float becomes float32 by a shift of 14.

    Parameters
    ----------
    stream : np.ndarray
        Big-endian bit stream (uint8).

    out : np.ndarray
        Output words (uint32).

    first : int
        Index of the first value to unpack.

    count : int
        Number of values.
    """
    i = 0
    while i < count and (first + i) & 3:
        bit = (first + i) * 18
        b = bit >> 3
        w = (
            (np.uint32(stream[b]) << 24)
            | (np.uint32(stream[b + 1]) << 16)
            | (np.uint32(stream[b + 2]) << 8)
        )
        if b + 3 < stream.size:
            w |= np.uint32(stream[b + 3])
        out[i] = ((w >> np.uint32(14 - (bit & 7))) & np.uint32(0x3FFFF)) << np.uint32(
            14
        )
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
    while i < count:
        bit = (first + i) * 18
        b = bit >> 3
        w = (
            (np.uint32(stream[b]) << 24)
            | (np.uint32(stream[b + 1]) << 16)
            | (np.uint32(stream[b + 2]) << 8)
        )
        if b + 3 < stream.size:
            w |= np.uint32(stream[b + 3])
        out[i] = ((w >> np.uint32(14 - (bit & 7))) & np.uint32(0x3FFFF)) << np.uint32(
            14
        )
        i += 1


@jit(nopython=True, cache=True, inline="always")
def load_be64(stream: np.ndarray, b: np.uint64) -> np.uint64:
    """
    Eight bytes from ``stream[b]`` as a big-endian word.

    Indexed with unsigned integers: numba then skips its negative-index
    wraparound, which otherwise makes these loads several times slower.

    Parameters
    ----------
    stream : np.ndarray
        Bytes (uint8).

    b : np.uint64
        First byte.

    Returns
    -------
    np.uint64
        The word.
    """
    return (
        (np.uint64(stream[b]) << np.uint64(56))
        | (np.uint64(stream[b + np.uint64(1)]) << np.uint64(48))
        | (np.uint64(stream[b + np.uint64(2)]) << np.uint64(40))
        | (np.uint64(stream[b + np.uint64(3)]) << np.uint64(32))
        | (np.uint64(stream[b + np.uint64(4)]) << np.uint64(24))
        | (np.uint64(stream[b + np.uint64(5)]) << np.uint64(16))
        | (np.uint64(stream[b + np.uint64(6)]) << np.uint64(8))
        | np.uint64(stream[b + np.uint64(7)])
    )


@jit(nopython=True, cache=True)
def word_loads(
    size: int, start_byte: int, precision: int, first: int, count: int
) -> int:
    """
    How many of the values can be read with one 8-byte load each.

    That is every value of at most 57 bits whose 8 bytes lie inside the
    stream: all but the last few.

    Parameters
    ----------
    size : int
        Bytes in the stream.

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
        The number of leading values to read with ``load_be64``.
    """
    if precision > 57 or size - 8 - start_byte < 0:
        return 0
    last_index = ((size - 8 - start_byte) * 8 + 7) // precision  # last that fits
    return max(0, min(count, last_index - first + 1))


@jit(nopython=True, cache=True)
def bits_at(
    stream: np.ndarray, start_byte: int, precision: int, index: int
) -> np.uint64:
    """
    Value ``index`` of ``precision``-bit values (at most 64), byte by byte.

    For the values ``word_loads`` leaves: near the end of the stream, or
    spanning 9 bytes (precision above 57).

    Parameters
    ----------
    stream : np.ndarray
        Big-endian bit stream (uint8).

    start_byte : int
        Byte at which the packed values start.

    precision : int
        Bits per value.

    index : int
        Which value.

    Returns
    -------
    np.uint64
        The value.
    """
    bit = index * precision
    b = start_byte + (bit >> 3)
    end = (bit & 7) + precision  # bits from the start of byte b
    top = np.uint64(0)
    if end > 64:  # keep the first byte apart
        top = np.uint64(stream[b])
        b += 1
        end -= 8
    nbytes = (end + 7) >> 3
    w = np.uint64(0)
    for k in range(nbytes):
        w = w << np.uint64(8)
        if b + k < stream.size:
            w |= np.uint64(stream[b + k])
    right = np.uint64(nbytes * 8 - end)
    value = w >> right
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
    Unpack ``precision``-bit values (at most 64) from a big-endian bit stream.

    Parameters
    ----------
    stream : np.ndarray
        Bit stream (uint8).

    start_byte : int
        Byte at which the packed values start.

    precision : int
        Bits per value.

    out : np.ndarray
        Output values (uint64).

    first : int
        Index of the first value to unpack.

    count : int
        Number of values.
    """
    fast = word_loads(stream.size, start_byte, precision, first, count)
    p = np.uint64(precision)
    right = np.uint64(64 - precision)
    base = np.uint64(start_byte)
    bit = np.uint64(first) * p
    for i in range(fast):
        w = load_be64(stream, base + (bit >> np.uint64(3)))
        out[i] = (w << (bit & np.uint64(7))) >> right
        bit += p
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
    Unpack truncated-mantissa IEEE floats straight into their bit patterns.

    The same as ``unpack_bits``, shifting left by ``shift`` and
    ``canonical_nans`` in one pass.

    Parameters
    ----------
    stream : np.ndarray
        Bit stream (uint8).

    precision : int
        Bits per value.

    shift : int
        Mantissa bits dropped when the values were written.

    exponent_mask, mantissa_mask : np.uint64
        Masks of the IEEE exponent and mantissa bits.

    out : np.ndarray
        IEEE bit patterns (uint32 or uint64).

    first : int
        Index of the first value to unpack.

    count : int
        Number of values.
    """
    fast = word_loads(stream.size, 0, precision, first, count)
    p = np.uint64(precision)
    right = np.uint64(64 - precision)
    left = np.uint64(shift)
    bit = np.uint64(first) * p
    for i in range(count):
        if i < fast:
            v = (
                load_be64(stream, bit >> np.uint64(3)) << (bit & np.uint64(7))
            ) >> right
        else:
            v = bits_at(stream, 0, precision, first + i)
        bit += p
        w = v << left
        if (w & exponent_mask) == exponent_mask and (w & mantissa_mask) != 0:
            w |= mantissa_mask
        out[i] = w


@jit(nopython=True, cache=True)
def canonical_nans(
    words: np.ndarray, exponent_mask: np.uint64, mantissa_mask: np.uint64
) -> None:
    """
    Set every mantissa bit of NaNs, as HDF5's float conversion leaves them.

    Parameters
    ----------
    words : np.ndarray
        IEEE bit patterns (uint32 or uint64), changed in place.

    exponent_mask : np.uint64
        Mask of the exponent bits.

    mantissa_mask : np.uint64
        Mask of the mantissa bits.
    """
    for i in range(words.size):
        w = words[i]
        if (w & exponent_mask) == exponent_mask and (w & mantissa_mask) != 0:
            words[i] = w | mantissa_mask


@jit(nopython=True, cache=True)
def widen_float(
    bits: np.ndarray,
    spos: int,
    epos: int,
    esize: int,
    mpos: int,
    msize: int,
    bias: int,
    target_esize: int,
    target_msize: int,
    target_bias: int,
    out: np.ndarray,
) -> None:
    """
    Convert custom floats to a wider IEEE format, as HDF5 does.

    Zeros keep their sign, subnormals are normalised, infinities stay infinite
    and NaNs get every mantissa bit set.

    Parameters
    ----------
    bits : np.ndarray
        Custom float bit patterns (uint64).

    spos, epos, esize, mpos, msize, bias : int
        Sign position, exponent position and size, mantissa position and size
        and exponent bias of the custom float.

    target_esize, target_msize, target_bias : int
        Exponent size, mantissa size and bias of the IEEE target.

    out : np.ndarray
        IEEE bit patterns (uint64).
    """
    emax = (np.uint64(1) << np.uint64(esize)) - np.uint64(1)
    mmask = (np.uint64(1) << np.uint64(msize)) - np.uint64(1)
    target_emax = (np.uint64(1) << np.uint64(target_esize)) - np.uint64(1)
    target_mmask = (np.uint64(1) << np.uint64(target_msize)) - np.uint64(1)
    shift = np.uint64(target_msize - msize)
    for i in range(bits.size):
        v = bits[i]
        sign = (v >> np.uint64(spos)) & np.uint64(1)
        e = (v >> np.uint64(epos)) & emax
        m = (v >> np.uint64(mpos)) & mmask
        if e == emax:
            te = target_emax
            tm = np.uint64(0) if m == 0 else target_mmask
        elif e == 0:
            if m == 0:
                te = np.uint64(0)
                tm = np.uint64(0)
            else:
                exponent = 1 - bias
                while (m >> np.uint64(msize)) == 0:
                    m = m << np.uint64(1)
                    exponent -= 1
                te = np.uint64(exponent + target_bias)
                tm = (m & mmask) << shift
        else:
            te = np.uint64(np.int64(e) - bias + target_bias)
            tm = m << shift
        out[i] = (
            (sign << np.uint64(target_esize + target_msize))
            | (te << np.uint64(target_msize))
            | tm
        )


@jit(nopython=True, cache=True)
def sign_extend(bits: np.ndarray, precision: int, out: np.ndarray) -> None:
    """
    Sign-extend ``precision``-bit two's complement values to 64 bits.

    Parameters
    ----------
    bits : np.ndarray
        Values (uint64).

    precision : int
        Bits per value.

    out : np.ndarray
        Sign-extended values (uint64).
    """
    sign = np.uint64(1) << np.uint64(precision - 1)
    fill = (
        ~((np.uint64(1) << np.uint64(precision)) - np.uint64(1))
        if precision < 64
        else np.uint64(0)
    )
    for i in range(bits.size):
        v = bits[i]
        out[i] = v | fill if v & sign else v


@jit(nopython=True, cache=True)
def scaleoffset_header(stream: np.ndarray) -> tuple:
    """
    Read a scale-offset chunk's header.

    Parameters
    ----------
    stream : np.ndarray
        The chunk's bytes after the other filters are undone.

    Returns
    -------
    tuple
        ``minbits`` and the minimum's bit pattern; the packed values start at
        byte 21.
    """
    minbits = np.int64(0)
    for i in range(4):
        minbits |= np.int64(stream[i]) << (8 * i)
    minval = np.uint64(0)
    for i in range(8):
        minval |= np.uint64(stream[5 + i]) << np.uint64(8 * i)
    return minbits, minval


@jit(nopython=True, cache=True)
def scaleoffset_float(
    stream: np.ndarray,
    minbits: int,
    minimum: float,
    scale: float,
    fill_available: bool,
    fill: float,
    out: np.ndarray,
    first: int,
    count: int,
) -> None:
    """
    Decode D-scaled floats as HDF5's ``H5Z_scaleoffset_modify_3``.

    The packed value is read as a signed integer of the element's size,
    divided by 10^D and added to the minimum, in the element's own precision;
    the all-ones value stands for the fill value.

    Parameters
    ----------
    stream : np.ndarray
        The chunk's bytes after the other filters are undone.

    minbits : int
        Bits per packed value.

    minimum : float
        The chunk's minimum.

    scale : float
        10^D.

    fill_available : bool
        Whether the dataset has a fill value.

    fill : float
        The fill value.

    out : np.ndarray
        Output values.

    first : int
        Index of the first value to decode.

    count : int
        Number of values.
    """
    if minbits == 0:
        for i in range(count):
            out[i] = minimum
        return
    sentinel = np.uint64(0xFFFFFFFFFFFFFFFF) >> np.uint64(64 - minbits)
    fast = word_loads(stream.size, 21, minbits, first, count)
    p = np.uint64(minbits)
    right = np.uint64(64 - minbits)
    bit = np.uint64(first) * p
    for i in range(count):
        if i < fast:
            b = np.uint64(21) + (bit >> np.uint64(3))
            v = (load_be64(stream, b) << (bit & np.uint64(7))) >> right
        else:
            v = bits_at(stream, 21, minbits, first + i)
        bit += p
        if fill_available and v == sentinel:
            out[i] = fill
        elif out.itemsize == 4:
            out[i] = np.float32(np.int32(np.uint32(v))) / np.float32(
                scale
            ) + np.float32(minimum)
        else:
            out[i] = np.float64(np.int64(v)) / scale + minimum


# ---------------------------------------------------------------------------
# Which datasets can be decoded
# ---------------------------------------------------------------------------


def _pipeline(dataset: h5py.Dataset) -> list:
    plist = dataset.id.get_create_plist()
    return [plist.get_filter(i)[:3] for i in range(plist.get_nfilters())]


def decoder_for(dataset: h5py.Dataset) -> dict | None:
    """
    Describe how to decode a (non-virtual) dataset's chunks.

    Every assumption the kernels make is checked here.

    Parameters
    ----------
    dataset : h5py.Dataset
        A dataset in a regular (non-virtual) file.

    Returns
    -------
    dict or None
        The decoding plan, or ``None`` to read the dataset with h5py.
    """
    if dataset.chunks is None or dataset.dtype.kind not in "fiu":
        return None
    filters = _pipeline(dataset)
    ids = [f[0] for f in filters]
    lossy = bool(ids) and ids[0] in (NBIT, SCALEOFFSET)
    if dataset.dtype.byteorder == ">" and lossy:
        return None  # the lossy filters' big-endian layouts are not decoded here
    dtype = dataset.dtype
    # each stage keeps its index in the pipeline: a chunk's filter mask has
    # that bit set if the stage was skipped when the chunk was written
    spec = dict(dtype=dtype, stages=[])
    for index, (code, flags, values) in enumerate(filters):
        if lossy and index == 0:
            continue
        if code == SHUFFLE:
            spec["stages"].append(("shuffle", values[0], index))
        elif code == DEFLATE:
            spec["stages"].append(("deflate", None, index))
        elif code == FLETCHER32:
            spec["stages"].append(("fletcher32", None, index))
        else:
            return None
    if not lossy:
        spec["kind"] = "plain"
        return spec
    cd = filters[0][2]
    if ids[0] == NBIT and len(cd) == 8 and cd[1] == 1:
        # the type uses every bit, so HDF5 stored the values unpacked
        ftype = dataset.id.get_type()
        ieee = dtype.kind in "iu" or ftype.get_fields() == (
            (31, 23, 8, 0, 23) if dtype.itemsize == 4 else (63, 52, 11, 0, 52)
        )
        if not ieee or ftype.get_offset() != 0:
            return None
        spec["kind"] = "plain"
        return spec
    if ids[0] == NBIT:
        if len(cd) != 8 or cd[1] != 0 or cd[3] != 1 or cd[5] != 0:
            return None  # only atomic, compressed, little-endian data
        ftype = dataset.id.get_type()
        precision, offset = ftype.get_precision(), ftype.get_offset()
        if cd[6] != precision or cd[7] != offset or precision + offset > 64:
            return None
        spec.update(precision=precision, offset=offset)
        if dtype.kind in "iu":
            spec.update(kind="nbit_int", signed=ftype.get_sign() == h5py.h5t.SGN_2)
            return spec
        if dtype.itemsize not in (4, 8):
            return None
        spos, epos, esize, mpos, msize = ftype.get_fields()
        target_esize, target_msize, target_bias = (
            (8, 23, 127) if dtype.itemsize == 4 else (11, 52, 1023)
        )
        bias = ftype.get_ebias()
        if offset != 0 or max(spos, epos + esize - 1, mpos + msize - 1) >= precision:
            return None
        if (
            ftype.get_norm() != h5py.h5t.NORM_IMPLIED
            or esize > target_esize
            or msize > target_msize
        ):
            return None
        if (
            (esize, bias) == (target_esize, target_bias)
            and mpos == 0
            and epos == msize
            and spos == epos + esize
        ):
            spec.update(kind="nbit_shift", shift=target_msize - msize)
        else:
            spec.update(
                kind="nbit_float",
                layout=(spos, epos, esize, mpos, msize, bias),
                target=(target_esize, target_msize, target_bias),
            )
        return spec
    scale_type, factor, klass, size, order, fill_available = (
        cd[0],
        cd[1],
        cd[3],
        cd[4],
        cd[6],
        cd[7],
    )
    if order != 0 or size != dtype.itemsize:
        return None
    fill = np.zeros(1, dtype=dtype)
    if fill_available:
        fill = np.frombuffer(
            np.array(cd[8 : 8 + (size + 3) // 4], dtype=np.uint32).tobytes()[:size],
            dtype=dtype,
        )
    if scale_type == 0 and klass == 1 and dtype.kind == "f":
        spec.update(kind="scaleoffset_float", scale=10.0**factor)
    elif scale_type == 2 and klass == 0 and dtype.kind in "iu":
        spec.update(kind="scaleoffset_int")
    else:
        return None
    spec.update(fill_available=bool(fill_available), fill=fill[0])
    return spec


def _sources(handle: h5py.Dataset) -> list | None:
    """
    Map a dataset's rows onto the files that store them.

    Parameters
    ----------
    handle : h5py.Dataset
        The dataset (possibly virtual).

    Returns
    -------
    list or None
        ``(first row, end row, file, dataset)`` blocks, or ``None`` if a
        virtual dataset's mapping is not whole rows in contiguous blocks.
    """
    path = handle.file.filename
    if not handle.is_virtual:
        return [(0, handle.shape[0], path, handle.name)]
    base = os.path.dirname(os.path.abspath(path))
    blocks = []
    for source in handle.virtual_sources():
        bounds = source.vspace.get_select_bounds()
        if bounds is None:  # a source with no rows of this dataset
            continue
        lo, hi = bounds
        rows = hi[0] + 1 - lo[0]
        # whole rows of every other dimension, and nothing but this block
        full_rows = all(
            lo[k] == 0 and hi[k] == handle.shape[k] - 1 for k in range(1, handle.ndim)
        )
        if not full_rows or source.vspace.get_select_npoints() != rows * int(
            np.prod(handle.shape[1:])
        ):
            return None
        src = source.src_space
        if src.get_select_type() != h5py.h5s.SEL_ALL:
            # a block of the source: whole rows from its first row
            if src.get_select_type() != h5py.h5s.SEL_HYPERSLABS:
                return None
            if src.get_select_npoints() != source.vspace.get_select_npoints():
                return None
            src_lo, src_hi = src.get_select_bounds()
            if src_lo != (0,) * handle.ndim or src_hi != (rows - 1,) + tuple(
                n - 1 for n in handle.shape[1:]
            ):
                return None
        # (a whole-source selection is checked against the source's shape
        # when the file is opened, in read_ranges)
        if source.file_name == ".":
            name = path
        elif os.path.isabs(source.file_name):
            name = source.file_name
        else:
            name = os.path.join(base, source.file_name)
        blocks.append((lo[0], lo[0] + rows, name, source.dset_name))
    return blocks


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------


def _inflate(data: np.ndarray, bound: int) -> np.ndarray:
    if _libdeflate is not None:
        return np.frombuffer(_libdeflate.zlib_decompress(data, bound), dtype=np.uint8)
    return np.frombuffer(zlib.decompress(data), dtype=np.uint8)


def _decode_chunk(
    spec: dict,
    raw: bytes,
    values: int,
    first: int,
    count: int,
    out: np.ndarray,
    scratch: np.ndarray,
    filter_mask: int = 0,
) -> None:
    """Decode values first..first+count of one raw chunk into out (flat)."""
    dtype = spec["dtype"]
    data = np.frombuffer(raw, dtype=np.uint8)
    for stage, parameter, index in reversed(spec["stages"]):
        if filter_mask >> index & 1:  # skipped when this chunk was written
            continue
        if stage == "fletcher32":
            stored = int.from_bytes(data[-4:].tobytes(), "little")
            computed = int(fletcher32(data, data.size - 4))
            swapped = int.from_bytes(computed.to_bytes(4, "little"), "big")
            if stored != computed and stored != swapped:  # HDF5 accepts either order
                raise ChecksumError(
                    "Fletcher-32 checksum mismatch: the file may be corrupt"
                )
            data = data[:-4]
        elif stage == "deflate":
            data = _inflate(data, values * dtype.itemsize + 64)
        else:
            unshuffled = (
                scratch[: data.size]
                if scratch.size >= data.size
                else np.empty(data.size, np.uint8)
            )
            unshuffle(data, parameter, unshuffled)
            data = unshuffled
    kind = spec["kind"]
    if kind == "plain":
        required = values * dtype.itemsize
    elif kind.startswith("nbit"):
        required = ((first + count) * spec["precision"] + 7) // 8
    else:
        if data.size < 21:
            raise ValueError("truncated scale-offset header")
        minbits, minval = scaleoffset_header(data)
        if not 0 <= minbits <= dtype.itemsize * 8:
            raise ValueError("invalid scale-offset precision")
        required = 21 + ((first + count) * minbits + 7) // 8
    if data.size < required:
        raise ValueError("truncated decoded chunk")
    if kind == "plain":
        out[:] = data[: values * dtype.itemsize].view(dtype)[first : first + count]
    elif kind == "nbit_shift":
        if spec["precision"] == 18 and spec["shift"] == 14 and dtype.itemsize == 4:
            unpack_nbit18(data, out.view(np.uint32), first, count)
            canonical_nans(
                out.view(np.uint32), np.uint32(0x7F800000), np.uint32(0x007FFFFF)
            )
        elif dtype.itemsize == 4:
            unpack_shifted_floats(
                data,
                spec["precision"],
                spec["shift"],
                np.uint64(0x7F800000),
                np.uint64(0x007FFFFF),
                out.view(np.uint32),
                first,
                count,
            )
        else:
            unpack_shifted_floats(
                data,
                spec["precision"],
                spec["shift"],
                np.uint64(0x7FF0000000000000),
                np.uint64(0x000FFFFFFFFFFFFF),
                out.view(np.uint64),
                first,
                count,
            )
    elif kind == "nbit_float":
        bits = np.empty(count, dtype=np.uint64)
        unpack_bits(data, 0, spec["precision"], bits, first, count)
        wide = np.empty(count, dtype=np.uint64)
        widen_float(bits, *spec["layout"], *spec["target"], wide)
        word = np.uint32 if dtype.itemsize == 4 else np.uint64
        out.view(word)[:] = wide.astype(word)
    elif kind == "nbit_int":
        # 8-byte integers are unpacked (and sign-extended) in place
        bits = (
            out.view(np.uint64) if dtype.itemsize == 8 else np.empty(count, np.uint64)
        )
        unpack_bits(data, 0, spec["precision"], bits, first, count)
        if spec["signed"]:
            sign_extend(bits, spec["precision"], bits)
        if dtype.itemsize != 8:
            out[:] = bits.astype(np.dtype(f"u{dtype.itemsize}")).view(dtype)
    else:
        size = dtype.itemsize
        word = np.dtype(f"u{size}")
        minimum = np.array([int(minval) & ((1 << (8 * size)) - 1)], dtype=word).view(
            dtype
        )[0]
        if kind == "scaleoffset_float":
            scaleoffset_float(
                data,
                minbits,
                minimum,
                spec["scale"],
                spec["fill_available"],
                spec["fill"],
                out,
                first,
                count,
            )
        elif minbits == 0:
            out[:] = minimum
        else:
            # (packed + minimum) wrapped to the element's width, as HDF5's C cast
            packed = np.empty(count, dtype=np.uint64)
            unpack_bits(data, 21, minbits, packed, first, count)
            wrapped = (packed + np.uint64(minval)) & np.uint64(
                (1 << (8 * size)) - 1 if size < 8 else 0xFFFFFFFFFFFFFFFF
            )
            result = wrapped.astype(word).view(dtype)
            if spec["fill_available"]:
                result = np.where(
                    packed == np.uint64((1 << minbits) - 1), spec["fill"], result
                )
            out[:] = result


def _convert(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Values in another numeric type as HDF5 converts them (floats rounded, integers clamped)."""
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


#: Chunks are read from the file in runs: adjacent chunks (SWIFT writes a
#: dataset's chunks one after another) with gaps of at most _RUN_GAP bytes go
#: in one read of at most _RUN_BYTES, instead of one request per chunk.
_RUN_GAP = 1 << 20
_RUN_BYTES = 64 << 20


def _path_identity(path: str) -> tuple | None:
    """Device, inode, modification time and size of a file, as _file_identity."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size


def _file_identity(file: h5py.File) -> tuple | None:
    """
    Identify an open file across reopenings, without looking up its path.

    swiftsimio opens a snapshot again for every field, so caches keyed on the
    open handle never hit. ``fstat`` on HDF5's own descriptor gives the file's
    device, inode, modification time and size in one call (a path lookup
    would cost a metadata request per directory on Lustre). ``None`` for
    files without a descriptor (e.g. in memory) or open for writing.
    """
    if file.mode != "r" or file.driver != "sec2":
        return None
    stat = os.fstat(file.id.get_vfd_handle())
    return stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size


#: Validated indexes of read-only datasets, bounded across reads.
_INDEX_CACHE_SIZE = 256
_index_cache = OrderedDict()


def _pread(fd: int, size: int, offset: int, exact: bool = True) -> bytes:
    """
    Read ``size`` bytes at ``offset`` (``pread`` may return fewer at a time).

    With ``exact=False`` fewer bytes are returned at the end of the file.
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
    Read ``(offset, size)`` regions, merging nearby ones into single requests.

    Returns each region's bytes (a ``memoryview``, shorter at the end of the
    file) by its offset.
    """
    out = {}
    regions = sorted(set(regions))
    i = 0
    while i < len(regions):
        start, end, j = regions[i][0], regions[i][0] + regions[i][1], i + 1
        while (
            j < len(regions)
            and regions[j][0] - end <= _RUN_GAP
            and regions[j][0] + regions[j][1] - start <= _RUN_BYTES
        ):
            end = max(end, regions[j][0] + regions[j][1])
            j += 1
        data = memoryview(_pread(fd, end - start, start, exact=False))
        for offset, size in regions[i:j]:
            out[offset] = data[offset - start : offset - start + size]
        i = j
    return out


def _layout_message(body: bytes) -> tuple | None:
    """Return the B-tree address and dimensionality of a version 3 chunked layout."""
    if len(body) < 11 or body[0] != 3 or body[1] != 2:
        return None  # not version 3 chunked (e.g. version 4 chunk indexes)
    return int.from_bytes(body[3:11], "little"), body[2]


def _layout_btree(fd: int, base: int, header: int) -> tuple | None:
    """
    Find a chunked dataset's v1 B-tree from its object header.

    Reads version 1 and version 2 (``OHDR``) object headers, following
    continuation blocks, as laid out in the HDF5 file format specification.

    Parameters
    ----------
    fd : int
        The file, opened for reading.

    base : int
        The file's base address (its user block size).

    header : int
        Address of the dataset's object header.

    Returns
    -------
    tuple or None
        The B-tree's address and the layout's dimensionality (the dataset's
        rank plus one), or ``None`` for any other header or layout.
    """
    prefix = _pread(fd, 64, base + header, exact=False)
    if prefix[:4] == b"OHDR":
        if prefix[4] != 2:
            return None
        flags = prefix[5]
        position = 6 + (16 if flags & 0x20 else 0) + (4 if flags & 0x10 else 0)
        width = 1 << (flags & 0x03)
        size = int.from_bytes(prefix[position : position + width], "little")
        blocks = [(header + position + width, size)]
        creation_order = bool(flags & 0x04)
        while blocks:
            address, size = blocks.pop(0)
            data = _pread(fd, size, base + address)
            position = 0
            if data[:4] == b"OCHK":
                position = 4
                size -= 4  # the block's checksum
            step = 6 if creation_order else 4
            while position + step <= size:
                kind = data[position]
                length = int.from_bytes(data[position + 1 : position + 3], "little")
                body = data[position + step : position + step + length]
                if kind == 0x10:
                    blocks.append(
                        (
                            int.from_bytes(body[0:8], "little"),
                            int.from_bytes(body[8:16], "little"),
                        )
                    )
                elif kind == 0x08:
                    return _layout_message(body)
                position += step + length
        return None
    if prefix[0] != 1:
        return None
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
            if kind == 0x10:  # continuation: more messages elsewhere
                blocks.append(
                    (
                        int.from_bytes(body[0:8], "little"),
                        int.from_bytes(body[8:16], "little"),
                    )
                )
            elif kind == 0x08:  # layout
                return _layout_message(body)
            position += 8 + length
    return None


def _chunk_index(dataset: h5py.Dataset, fd: int) -> dict | None:
    """
    Read every allocated chunk's location from a dataset's v1 B-tree.

    HDF5 finds a chunk by searching this tree, one search per chunk; on a
    busy file system each search costs small reads. Here the tree is read a
    level at a time, each level's nodes merged into a few large reads.

    Parameters
    ----------
    dataset : h5py.Dataset
        A chunked dataset in a regular file.

    fd : int
        The dataset's file, opened for reading.

    Returns
    -------
    dict or None
        ``{chunk coordinates: (file offset, size, filter mask)}``, or
        ``None`` if the dataset is not indexed this way (or anything looks
        unexpected): look the chunks up one by one instead.
    """
    if dataset.file.id.get_create_plist().get_sizes() != (8, 8):
        return None
    base = dataset.file.userblock_size
    found = _layout_btree(fd, base, h5py.h5o.get_info(dataset.id).addr)
    if found is None:
        return None
    root, dimensionality = found
    if dimensionality != dataset.ndim + 1:
        return None
    key = 8 + 8 * dimensionality  # chunk size, filter mask, offsets
    largest = 24 + 129 * key + 128 * 8  # a node of 128 entries (HDF5's 2K = 64)
    index, level, nodes = {}, None, [root]
    while nodes:
        regions = _read_regions(fd, [(base + n, largest) for n in nodes])
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
            for i in range(entries):
                k = 24 + i * (key + 8)
                child = int.from_bytes(data[k + key : k + key + 8], "little")
                if level > 0:
                    children.append(child)
                    continue
                size = int.from_bytes(data[k : k + 4], "little")
                mask = int.from_bytes(data[k + 4 : k + 8], "little")
                coords = tuple(
                    int.from_bytes(data[k + 8 + 8 * d : k + 16 + 8 * d], "little")
                    for d in range(dataset.ndim)
                )
                index[coords] = (base + child, size, mask)
        nodes = children
        if children:
            level -= 1
    return index


class _ChunkReader:
    """
    Raw chunks of one dataset, read in runs of adjacent chunks.

    Each ``read_direct_chunk`` is a separate request to the file system: on a
    busy Lustre that costs milliseconds each, which dominates for datasets of
    many small chunks. Here the chunks' file offsets are looked up once and
    each run of adjacent chunks is read with one ``pread``. The bytes are the
    same as ``read_direct_chunk`` gives (checked on the first chunk); without
    offsets (e.g. unallocated chunks) it reads chunk by chunk as before.

    Parameters
    ----------
    source : h5py.Dataset
        A dataset in a regular (non-virtual) file.

    coords : list
        The chunk coordinates that will be read, in any order.
    """

    def __init__(self, source: h5py.Dataset, coords: list) -> None:
        self.source = source
        self.fd = None
        self.where = {}  # chunk -> (run, start in run, size, filter mask)
        self.unallocated = set()  # chunks never written: the fill value
        self.runs = []  # [file offset, length, chunks left, bytes or None]
        if not coords:
            return
        self.cache_key = None
        self.index = None
        self.validated = False
        if not hasattr(os, "pread") or source.file.driver != "sec2":
            return
        self.fd = source.file.id.get_vfd_handle()
        infos = self._locate(coords)
        self.unallocated = {c for c, i in infos.items() if i is None}
        infos = {c: i for c, i in infos.items() if i is not None}
        for c in sorted(infos, key=lambda c: infos[c][0]):
            offset, size, filter_mask = infos[c]
            run = self.runs[-1] if self.runs else None
            if (
                run is None
                or offset - (run[0] + run[1]) > _RUN_GAP
                or offset - (run[0] + run[1]) < 0
                or offset + size - run[0] > _RUN_BYTES
            ):
                run = [offset, 0, 0, None]
                self.runs.append(run)
            run[1] = offset + size - run[0]
            run[2] += 1
            self.where[c] = (len(self.runs) - 1, offset - run[0], size, filter_mask)
        if not infos or self.validated:
            return
        try:
            first = next(iter(infos))
            mask, raw = self.read(first)
            if (mask, bytes(raw)) != tuple(source.id.read_direct_chunk(first)):
                raise OSError("chunk offsets do not match the chunks")
            # (read() counted the first chunk as served; serve it again later)
            self.runs[self.where[first][0]][2] += 1
            if self.cache_key is not None and self.index is not None:
                _index_cache[self.cache_key] = self.index
                _index_cache.move_to_end(self.cache_key)
                while len(_index_cache) > _INDEX_CACHE_SIZE:
                    _index_cache.popitem(last=False)
        except OSError:
            self.close()
            raise

    def _locate(self, coords: list) -> dict:
        """
        Find the chunks in the file: ``{coords: (offset, size, mask) or None}``.

        The dataset's chunk index is read directly where possible (a few large
        reads); its first and last entries are checked against HDF5, and a
        chunk missing from it is confirmed unallocated with HDF5. Otherwise
        HDF5 is asked chunk by chunk.
        """
        source = self.source

        def ask(c: tuple) -> tuple | None:
            info = source.id.get_chunk_info_by_coord(c)
            if info.byte_offset is None:
                return None
            return info.byte_offset, info.size, info.filter_mask

        identity = _file_identity(source.file)
        if identity is not None:
            self.cache_key = identity + (source.name,)
            if self.cache_key in _index_cache:
                self.validated = True
                self.index = _index_cache[self.cache_key]
                _index_cache.move_to_end(self.cache_key)
                return {c: self.index[c] if c in self.index else ask(c) for c in coords}
        try:
            index = _chunk_index(source, self.fd)
            if index:
                ordered = sorted(index)
                if any(ask(c) != index[c] for c in {ordered[0], ordered[-1]}):
                    index = None  # not what HDF5 finds: ask it chunk by chunk
        except (OSError, ValueError, IndexError, AttributeError, RuntimeError):
            index = None  # e.g. an index layout not parsed here
        if not index:
            return {c: ask(c) for c in coords}
        self.index = index
        return {c: index[c] if c in index else ask(c) for c in coords}

    def read(self, coord: tuple) -> tuple:
        """
        Return ``(filter mask, raw bytes)`` of one chunk, like ``read_direct_chunk``.

        The bytes are ``None`` for a chunk that was never written.
        """
        if coord in self.unallocated:
            return 0, None
        if coord not in self.where:
            return self.source.id.read_direct_chunk(coord)
        index, start, size, filter_mask = self.where[coord]
        run = self.runs[index]
        if run[3] is None:
            parts, done = [], 0
            while done < run[1]:  # pread may return less than asked
                part = os.pread(self.fd, run[1] - done, run[0] + done)
                if not part:
                    raise OSError("unexpected end of file reading chunks")
                parts.append(part)
                done += len(part)
            run[3] = parts[0] if len(parts) == 1 else b"".join(parts)
        raw = memoryview(run[3])[start : start + size]  # no copy
        run[2] -= 1
        if run[2] == 0:
            run[3] = None  # every chunk of the run served
        return filter_mask, raw

    def close(self) -> None:
        """Release the borrowed descriptor without closing HDF5's file."""
        self.fd = None


def _all_columns(columns: object) -> bool:
    """Whether a column selection is every column (``[:]``)."""
    return isinstance(columns, slice) and columns == slice(None)


def _copy_pieces(
    source: h5py.Dataset, pieces: list, output: np.ndarray, columns: object
) -> None:
    """
    Copy row pieces of a regular dataset, with columns selected, into output.

    Each chunk is read and decoded once, however many pieces it holds.

    Parameters
    ----------
    source : h5py.Dataset
        A dataset in a regular (non-virtual) file.

    pieces : list
        ``(first row, end row, output row)`` triples.

    output : np.ndarray
        The output array (rows first).

    columns : object
        Column selection, used only if the dataset is multidimensional.
    """
    spec = decoder_for(source)
    if spec is None:
        for lo, hi, at in pieces:
            rows = source[lo:hi]
            output[at : at + hi - lo] = _convert(
                np.ascontiguousarray(
                    rows[(slice(None), columns)] if source.ndim > 1 else rows
                ),
                output.dtype,
            )
        return
    chunk_rows, chunk_tail = source.chunks[0], tuple(source.chunks[1:])
    tail = tuple(source.shape[1:])
    whole_rows = chunk_tail == tail  # SWIFT's layout: one chunk per block of rows
    row_values = int(np.prod(chunk_tail)) if source.ndim > 1 else 1
    values = chunk_rows * row_values
    native = source.dtype.newbyteorder("=")
    scratch = np.empty(values * source.dtype.itemsize + 64, dtype=np.uint8)
    decoded = np.empty(values, dtype=native)
    # chunks across the other dimensions, without those no selected column needs
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
    lossy = spec["kind"] != "plain"  # the lossy filter is the pipeline's first
    # decode straight into the output when it is exactly the decoded rows
    direct = (
        whole_rows
        and output.dtype == native
        and (source.ndim == 1 or _all_columns(columns))
    )
    by_chunk = {}
    for lo, hi, at in pieces:
        for k in range(lo // chunk_rows, (hi - 1) // chunk_rows + 1):
            c0, c1 = max(lo, k * chunk_rows), min(hi, (k + 1) * chunk_rows)
            by_chunk.setdefault(k, []).append((c0, c1, at + c0 - lo))
    reader = _ChunkReader(
        source, [(k * chunk_rows,) + corner for k in by_chunk for corner in grid]
    )
    span_buffer = None
    try:
        for k in sorted(by_chunk):
            parts = by_chunk[k]
            s0 = min(c0 for c0, _, _ in parts)
            s1 = max(c1 for _, c1, _ in parts)
            rows = s1 - s0
            if direct and len(parts) == 1:
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
                    block[(slice(None),) + inside] = source[(slice(s0, s1),) + inside]
                    continue
                flat = block.reshape(-1) if whole_rows else decoded[: rows * row_values]
                _decode_chunk(
                    spec,
                    raw,
                    values,
                    (s0 - k * chunk_rows) * row_values,
                    rows * row_values,
                    flat,
                    scratch,
                    filter_mask,
                )
                if not whole_rows:
                    block[(slice(None),) + inside] = flat.reshape((rows,) + chunk_tail)[
                        (slice(None),)
                        + tuple(slice(0, q.stop - q.start) for q in inside)
                    ]
            if direct and len(parts) == 1:
                continue
            for c0, c1, at in parts:
                piece = block[c0 - s0 : c1 - s0]
                if source.ndim > 1:
                    piece = piece[(slice(None), columns)]
                output[at : at + c1 - c0] = _convert(np.asarray(piece), output.dtype)

    finally:
        reader.close()


#: The last multi-column field read for one of its columns, all columns
#: decoded: the other columns (swiftsimio's named columns) are then copied
#: from it instead of decoding every chunk again. Dropped once every column
#: has been read, or when another field takes its place.
_columns_cache = {"key": None, "data": None, "served": None, "sources": ()}

#: The source files (and their identities) the last virtual read used.
_last_sources = ()

#: Fields larger than this (bytes, all columns) are read a column at a time.
COLUMNS_CACHE_BYTES = 2 << 30


def read_ranges(
    handle: h5py.Dataset,
    ranges: np.ndarray,
    output_shape: tuple,
    output_type: type,
    columns: object = np.s_[:],
) -> np.ndarray | None:
    """
    Read row ranges of a dataset by decoding its chunks directly.

    Parameters
    ----------
    handle : h5py.Dataset
        The dataset (possibly virtual).

    ranges : np.ndarray
        ``[start, stop)`` row ranges.

    output_shape : tuple
        Shape of the result.

    output_type : type
        :class:`numpy.dtype` of the result.

    columns : object, optional
        Column selection for multi-dimensional datasets.

    Returns
    -------
    np.ndarray or None
        The rows, identical to h5py's result, or ``None`` if this dataset is
        better read with h5py (the caller then does so).
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
    """Read some columns of the rows, decoding every column once for all of them."""
    rows = int((ranges[:, 1] - ranges[:, 0]).sum())
    width = handle.shape[1]
    identity = _file_identity(handle.file)
    if rows * width * dtype.itemsize > COLUMNS_CACHE_BYTES:
        return _read(
            handle,
            ranges,
            np.empty((rows,) + np.empty((0, width))[:, columns].shape[1:], dtype),
            columns,
        )
    # the file (a virtual file's own identity: SWIFT's sub-files do not change
    # under an unchanged virtual file), or this open handle for files in memory
    key = (
        identity if identity is not None else ("open", handle.file.id.fileno),
        handle.name,
        dtype.str,
        ranges.tobytes(),
    )
    cache = _columns_cache
    if cache["key"] == key and not all(
        _path_identity(path) == identity for path, identity in cache["sources"]
    ):
        cache["key"] = None  # a virtual dataset's source file changed since
    if cache["key"] != key:
        cache.update(key=None, data=None, served=None, sources=())  # free it first
        global _last_sources
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


def _read_unchecked(
    handle: h5py.Dataset, ranges: np.ndarray, output: np.ndarray, columns: object
) -> np.ndarray | None:
    """Read the ranges into output; ``None`` if h5py should read them instead."""
    if not handle.is_virtual:
        # read through this handle: its file may not be on disk under its
        # name (e.g. an in-memory file), so it is never reopened
        pieces, position = [], 0
        for lo, hi in ranges:
            if hi > lo:
                pieces.append((lo, hi, position))
            position += max(hi - lo, 0)
        _copy_pieces(handle, pieces, output, columns)
        return output
    blocks = _sources(handle)
    if blocks is None:
        return None
    # rows no source maps read as the fill value too (file None); sources
    # mapping the same rows are left to h5py
    blocks.sort()
    covered, gaps = 0, []
    for start, stop, _, _ in blocks:
        if start < covered:
            return None
        if start > covered:
            gaps.append((covered, start, None, None))
        covered = stop
    if covered < handle.shape[0]:
        gaps.append((covered, handle.shape[0], None, None))
    blocks = sorted(blocks + gaps, key=lambda block: block[0])
    if len(ranges):  # only the files the ranges touch
        blocks = [
            b for b in blocks if any(lo < b[1] and hi > b[0] for lo, hi in ranges)
        ]
    handles = {}
    missing = {None}  # rows mapped by no source
    try:
        for start, stop, fname, dname in blocks:
            if fname in missing:
                continue
            if fname not in handles:
                try:
                    handles[fname] = open_hdf5(fname)
                except FileNotFoundError:
                    # HDF5 may still find it (HDF5_VDS_PREFIX): leave it to h5py
                    return None
            source = handles[fname][dname]
            if source.shape[0] < stop - start or source.shape[1:] != handle.shape[1:]:
                return None  # the source is not the rows this block maps
        global _last_sources
        _last_sources = tuple(
            (fname, _file_identity(h.file)) for fname, h in handles.items()
        )
        # each block's pieces, in output order
        pieces = {block: [] for block in blocks}
        position = 0
        for lo, hi in ranges:
            for block in blocks:
                start, stop, fname, _ = block
                a, b = max(lo, start), min(hi, stop)
                if a >= b:
                    continue
                if fname in missing:
                    output[position : position + b - a] = np.array(
                        handle.fillvalue
                    ).astype(output.dtype)
                else:
                    pieces[block].append((a - start, b - start, position))
                position += b - a
        for block, block_pieces in pieces.items():
            if block_pieces:
                start, stop, fname, dname = block
                _copy_pieces(handles[fname][dname], block_pieces, output, columns)
    finally:
        for h in handles.values():
            h.close()
    return output


def _read(
    handle: h5py.Dataset, ranges: np.ndarray, output: np.ndarray, columns: object
) -> np.ndarray | None:
    """Read ranges, leaving unexpected failures to h5py except checksum errors."""
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
