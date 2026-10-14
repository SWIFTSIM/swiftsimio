"""
Fast reads of SWIFT snapshot datasets, decoding HDF5 chunks without HDF5.

HDF5 undoes SWIFT's lossy compression slowly: its N-bit filter and the
conversion of SWIFT's reduced-precision floats back to IEEE work element by
element and bit by bit, at a few tens of MB/s. Here every chunk is read raw
with ``read_direct_chunk`` and its filters are undone in numba instead, giving
exactly the bytes h5py would return.

Supported (little-endian, chunked datasets), covering everything SWIFT writes
(``src/io_compression.c``, ``single_io.c``, ``distributed_io.c``,
``common_io_cells.c``):

+ an optional first filter:

  + N-bit: SWIFT's ``FMantissa*``, ``DMantissa*``, ``BFloat16``, ``HalfFloat``
    (any custom float layout, widened exactly as HDF5 converts it) and
    ``Nbit*`` (integers, sign-extended when signed);
  + scale-offset: SWIFT's ``DScale*`` (float32 and float64) and integers;

+ then shuffle, deflate and Fletcher-32 in any order, each optional.

Everything else, and any chunk whose filters were skipped when writing, is read
with h5py as before. Every chunk's Fletcher-32 checksum is verified, as HDF5
does. Virtual datasets are mapped onto their source files when each source
supplies whole rows to one contiguous block, as in SWIFT's virtual snapshots;
anything else is read with h5py.

Set the environment variable ``SWIFTSIMIO_FAST_DECODE=0``, or
``swiftsimio._fast_decode.ENABLED = False``, to always read with h5py.
"""

import os
import zlib

import h5py
import numpy as np

from .optional_packages import jit, NUMBA_AVAILABLE

try:  # libdeflate inflates about twice as fast as zlib
    import deflate as _libdeflate
except (ImportError, ModuleNotFoundError):
    _libdeflate = None

__all__ = ["ENABLED", "MINIMUM_ELEMENTS", "ChecksumError", "read_ranges"]

#: Decode chunks here rather than with HDF5 (needs numba to be fast).
ENABLED = NUMBA_AVAILABLE and os.environ.get("SWIFTSIMIO_FAST_DECODE", "1") != "0"

#: Reads of fewer elements than this go to h5py, which is fast enough there.
MINIMUM_ELEMENTS = 1 << 16

NBIT, SHUFFLE, DEFLATE, FLETCHER32, SCALEOFFSET = 5, 2, 1, 3, 6


class ChecksumError(OSError):
    """A chunk's Fletcher-32 checksum does not match its data."""


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


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
        for _ in range(tlen):
            sum1 += (np.int64(data[i]) << 8) | np.int64(data[i + 1])
            sum2 += sum1
            i += 2
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
    Unpack ``precision``-bit values (at most 57) from a big-endian bit stream.

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
    mask = (np.uint64(1) << np.uint64(precision)) - np.uint64(1)
    for i in range(count):
        bit = (first + i) * precision
        b = start_byte + (bit >> 3)
        skip = bit & 7
        nbytes = (skip + precision + 7) >> 3
        w = np.uint64(0)
        for k in range(nbytes):
            w = w << np.uint64(8)
            if b + k < stream.size:
                w |= np.uint64(stream[b + k])
        out[i] = (w >> np.uint64(nbytes * 8 - skip - precision)) & mask


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
    values = np.empty(count, dtype=np.uint64)
    unpack_bits(stream, 21, minbits, values, first, count)
    sentinel = (np.uint64(1) << np.uint64(minbits)) - np.uint64(1)
    for i in range(count):
        v = values[i]
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
    if (
        dataset.chunks is None
        or dataset.dtype.byteorder == ">"
        or dataset.dtype.kind not in "fiu"
    ):
        return None
    if tuple(dataset.chunks[1:]) != tuple(dataset.shape[1:]):
        return None  # chunks must hold whole rows (SWIFT's always do)
    filters = _pipeline(dataset)
    ids = [f[0] for f in filters]
    dtype = dataset.dtype
    spec = dict(dtype=dtype, stages=[])
    rest = filters[1:] if ids and ids[0] in (NBIT, SCALEOFFSET) else filters
    for code, flags, values in rest:
        if code == SHUFFLE:
            spec["stages"].append(("shuffle", values[0]))
        elif code == DEFLATE:
            spec["stages"].append(("deflate", None))
        elif code == FLETCHER32:
            spec["stages"].append(("fletcher32", None))
        else:
            return None
    if not ids or ids[0] not in (NBIT, SCALEOFFSET):
        spec["kind"] = "plain"
        return spec
    cd = filters[0][2]
    if ids[0] == NBIT:
        if len(cd) != 8 or cd[1] != 0 or cd[3] != 1 or cd[5] != 0:
            return None  # only atomic, compressed, little-endian data
        ftype = dataset.id.get_type()
        precision, offset = ftype.get_precision(), ftype.get_offset()
        if cd[6] != precision or cd[7] != offset or precision > 57:
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
            if src.get_select_bounds()[0][0] != 0:
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
) -> None:
    """Decode values first..first+count of one raw chunk into out (flat)."""
    dtype = spec["dtype"]
    data = np.frombuffer(raw, dtype=np.uint8)
    for stage, parameter in reversed(spec["stages"]):
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
        out[:] = data[: values * dtype.itemsize].view(dtype)[first : first + count]
    elif kind == "nbit_shift":
        word = np.uint32 if dtype.itemsize == 4 else np.uint64
        if spec["precision"] == 18 and spec["shift"] == 14 and dtype.itemsize == 4:
            unpack_nbit18(data, out.view(np.uint32), first, count)
        else:
            bits = np.empty(count, dtype=np.uint64)
            unpack_bits(data, 0, spec["precision"], bits, first, count)
            out.view(word)[:] = (bits << np.uint64(spec["shift"])).astype(word)
        if dtype.itemsize == 4:
            canonical_nans(
                out.view(np.uint32), np.uint32(0x7F800000), np.uint32(0x007FFFFF)
            )
        else:
            canonical_nans(
                out.view(np.uint64),
                np.uint64(0x7FF0000000000000),
                np.uint64(0x000FFFFFFFFFFFFF),
            )
    elif kind == "nbit_float":
        bits = np.empty(count, dtype=np.uint64)
        unpack_bits(data, 0, spec["precision"], bits, first, count)
        wide = np.empty(count, dtype=np.uint64)
        widen_float(bits, *spec["layout"], *spec["target"], wide)
        word = np.uint32 if dtype.itemsize == 4 else np.uint64
        out.view(word)[:] = wide.astype(word)
    elif kind == "nbit_int":
        bits = np.empty(count, dtype=np.uint64)
        unpack_bits(data, 0, spec["precision"], bits, first, count)
        if spec["signed"]:
            extended = np.empty(count, dtype=np.uint64)
            sign_extend(bits, spec["precision"], extended)
            bits = extended
        out[:] = bits.astype(np.dtype(f"u{dtype.itemsize}")).view(dtype)
    else:
        minbits, minval = scaleoffset_header(data)
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


def _copy_rows(
    source: h5py.Dataset, lo: int, hi: int, out: np.ndarray, columns: object
) -> None:
    """Rows lo..hi of a regular dataset, with columns selected, into out (rows first)."""
    spec = decoder_for(source)
    if spec is None:
        rows = source[lo:hi]
        out[:] = _convert(
            np.ascontiguousarray(
                rows[(slice(None), columns)] if source.ndim > 1 else rows
            ),
            out.dtype,
        )
        return
    chunk_rows = source.chunks[0]
    row_values = int(np.prod(source.shape[1:])) if source.ndim > 1 else 1
    values = chunk_rows * row_values
    native = source.dtype.newbyteorder("=")
    scratch = np.empty(values * source.dtype.itemsize + 64, dtype=np.uint8)
    decoded = np.empty(values, dtype=native)
    position = 0
    for k in range(lo // chunk_rows, (hi - 1) // chunk_rows + 1):
        c0, c1 = max(lo, k * chunk_rows), min(hi, (k + 1) * chunk_rows)
        rows = c1 - c0
        filter_mask, raw = source.id.read_direct_chunk(
            (k * chunk_rows,) + (0,) * (source.ndim - 1)
        )
        if filter_mask != 0:  # a filter was skipped for this chunk: let HDF5 decode it
            block = source[c0:c1]
        else:
            flat = decoded[: rows * row_values]
            _decode_chunk(
                spec,
                raw,
                values,
                (c0 - k * chunk_rows) * row_values,
                rows * row_values,
                flat,
                scratch,
            )
            block = flat.reshape((rows,) + source.shape[1:])
        if source.ndim > 1:
            block = block[(slice(None), columns)]
        out[position : position + rows] = _convert(np.asarray(block), out.dtype)
        position += rows


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
    output = np.empty(output_shape, dtype=np.dtype(output_type).newbyteorder("="))
    if output.size < MINIMUM_ELEMENTS:
        return None
    blocks = _sources(handle)
    if blocks is None:
        return None
    if len(ranges):  # only the files the ranges touch
        blocks = [
            b for b in blocks if any(lo < b[1] and hi > b[0] for lo, hi in ranges)
        ]
    handles = {}
    try:
        for start, stop, fname, dname in blocks:
            if fname not in handles:
                handles[fname] = h5py.File(fname, "r")
            source = handles[fname][dname]
            if source.shape[0] < stop - start or source.shape[1:] != handle.shape[1:]:
                return None  # the source is not the rows this block maps
        position = 0
        for lo, hi in ranges:
            for start, stop, fname, dname in blocks:
                a, b = max(lo, start), min(hi, stop)
                if a >= b:
                    continue
                _copy_rows(
                    handles[fname][dname],
                    a - start,
                    b - start,
                    output[position : position + b - a],
                    columns,
                )
                position += b - a
    except (OSError, KeyError) as error:
        if isinstance(error, ChecksumError):
            raise
        return None  # e.g. a missing source file: h5py applies the fill value
    finally:
        for h in handles.values():
            h.close()
    return output
