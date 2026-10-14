"""
Tests for swiftsimio._fast_decode.

Chunks decoded without HDF5 must give exactly the bytes h5py gives, for every
filter pipeline SWIFT writes.
"""

import h5py
import numpy as np
import pytest

from swiftsimio import _fast_decode as fd
from swiftsimio.accelerated import read_ranges_from_file


@pytest.fixture(autouse=True)
def small_reads_too(monkeypatch):
    """Use the fast path even for the small test datasets."""
    monkeypatch.setattr(fd, "MINIMUM_ELEMENTS", 0)
    monkeypatch.setattr(fd, "ENABLED", True)


CHUNK = 1000
N = 4321  # not a multiple of the chunk: the last chunk is partly filled
RANGES = [(0, 1), (5, 999), (999, 1001), (1500, 1500), (1990, 3017), (4200, N)]


def reference(path, name, ranges=None, columns=np.s_[:]):
    """Read the ranges (and columns) with h5py."""
    ranges = RANGES if ranges is None else ranges
    with h5py.File(path) as f:
        d = f[name]
        parts = [d[lo:hi] for lo, hi in ranges]
    data = np.concatenate(parts)
    return np.ascontiguousarray(data[(slice(None), columns)] if data.ndim > 1 else data)


def fast_read(path, name, ranges, columns=np.s_[:], output_type=None):
    """Read the ranges with the fast decoder (None if it declines)."""
    with h5py.File(path) as f:
        d = f[name]
        rows = int(sum(hi - lo for lo, hi in ranges))
        if d.ndim > 1:
            shape = (rows,) + np.empty((1,) + d.shape[1:])[
                (slice(None), columns)
            ].shape[1:]
        else:
            shape = (rows,)
        return fd.read_ranges(
            d, np.array(ranges), shape, output_type or d.dtype, columns
        )


def assert_same(path, name, ranges=None, fast=True, columns=np.s_[:]):
    """Assert the fast decoder and swiftsimio give exactly what h5py gives."""
    ranges = RANGES if ranges is None else ranges
    ref = reference(path, name, ranges, columns)
    with h5py.File(path) as f:
        d = f[name]
        blocks = fd._sources(d)
        decodable = blocks is not None and all(
            fd.decoder_for(h5py.File(fname)[dname]) is not None
            for _, _, fname, dname in blocks
        )
    assert decodable == fast
    ours = fast_read(path, name, ranges, columns)
    if fast:
        assert ours is not None
    if ours is not None:
        assert ours.dtype == ref.dtype.newbyteorder("=") and ours.shape == ref.shape
        assert ours.tobytes() == ref.astype(ref.dtype.newbyteorder("=")).tobytes()
    # and through swiftsimio's entry point, fast path or not
    with h5py.File(path) as f:
        d = f[name]
        through = read_ranges_from_file(
            d, np.array(ranges), ref.shape, d.dtype, columns
        )
    assert through.tobytes() == ref.astype(ref.dtype.newbyteorder("=")).tobytes()


def sample(n, cols, rng, dtype):
    """
    Make test values.

    They span many orders of magnitude, with both signs, exact zeros, and
    values at the top and bottom of each type's range.
    """
    shape = (n, cols) if cols > 1 else (n,)
    x = rng.standard_normal(shape) * 10.0 ** rng.integers(-30, 30, shape)
    x.flat[::17] = 0.0
    x.flat[::23] *= -1
    info = np.finfo(dtype)
    x.flat[1] = info.max
    x.flat[2] = info.tiny  # smallest normal
    x.flat[3] = info.tiny / 8  # subnormal
    x.flat[4] = -info.max
    return x.astype(dtype)


def nbit_type(base, mantissa, esize, ebias):
    """Make a custom float type with the given exponent and mantissa."""
    t = base.copy()
    t.set_fields(
        esize + mantissa, mantissa, esize, 0, mantissa
    )  # spos, epos, esize, mpos, msize
    t.set_precision(1 + esize + mantissa)
    t.set_offset(0)
    t.set_ebias(ebias)
    return t


def write_nbit(path, name, data, file_type, shuffle=True, checksum=True):
    """Write a dataset with the N-bit, shuffle, deflate and checksum filters."""
    with h5py.File(path, "a") as f:
        if "/" in name:
            f.require_group(name.rsplit("/", 1)[0])
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((min(CHUNK, data.shape[0]),) + data.shape[1:])
        dcpl.set_filter(h5py.h5z.FILTER_NBIT, 0, ())
        if shuffle:
            dcpl.set_shuffle()
        dcpl.set_deflate(4)
        if checksum:
            dcpl.set_fletcher32()
        space = h5py.h5s.create_simple(data.shape)
        try:
            d = h5py.Dataset(
                h5py.h5d.create(f.id, name.encode(), file_type, space, dcpl)
            )
        except ValueError as error:
            if "unusual datatype" in str(error):
                pytest.skip(
                    "HDF5 refuses to create this float type without relaxed integrity checks"
                )
            raise
        d[...] = data


# --- N-bit floats (SWIFT FMantissa / DMantissa / BFloat16) ------------------


@pytest.mark.parametrize("mantissa", [5, 7, 9, 13, 17, 22, 23])
@pytest.mark.parametrize("cols", [1, 2, 6])
def test_nbit_float32(tmp_path, mantissa, cols):
    """Check nbit float32."""
    data = sample(N, cols, np.random.default_rng(mantissa + cols), np.float32)
    write_nbit(
        tmp_path / "f.hdf5", "x", data, nbit_type(h5py.h5t.IEEE_F32LE, mantissa, 8, 127)
    )
    # a full 23-bit mantissa is plain float32: HDF5 marks it "need not
    # compress" and the decoder leaves it to h5py
    assert_same(tmp_path / "f.hdf5", "x", fast=mantissa < 23)


@pytest.mark.parametrize("mantissa", [9, 21, 35, 52])
@pytest.mark.parametrize("cols", [1, 3])
def test_nbit_float64(tmp_path, mantissa, cols):
    """Check nbit float64."""
    data = sample(N, cols, np.random.default_rng(mantissa), np.float64)
    write_nbit(
        tmp_path / "f.hdf5",
        "x",
        data,
        nbit_type(h5py.h5t.IEEE_F64LE, mantissa, 11, 1023),
    )
    if 1 + 11 + mantissa > 57:  # beyond the decoder's 64-bit window: h5py
        assert_same(tmp_path / "f.hdf5", "x", fast=False)
    else:
        assert_same(tmp_path / "f.hdf5", "x")


def test_nbit_special_values(tmp_path):
    """Check nbit special values."""
    nans = np.array(
        [0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FFFC000], dtype=np.uint32
    ).view(np.float32)
    data = np.concatenate(
        [np.array([np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0], dtype=np.float32), nans]
        * 180
    )
    write_nbit(
        tmp_path / "f.hdf5", "x", data, nbit_type(h5py.h5t.IEEE_F32LE, 9, 8, 127)
    )
    assert_same(tmp_path / "f.hdf5", "x", [(0, data.size)])


@pytest.mark.parametrize(
    "shuffle,checksum", [(False, True), (True, False), (False, False)]
)
def test_nbit_pipeline_variants(tmp_path, shuffle, checksum):
    """Check nbit pipeline variants."""
    data = sample(N, 1, np.random.default_rng(1), np.float32)
    write_nbit(
        tmp_path / "f.hdf5",
        "x",
        data,
        nbit_type(h5py.h5t.IEEE_F32LE, 9, 8, 127),
        shuffle,
        checksum,
    )
    assert_same(tmp_path / "f.hdf5", "x")


# --- scale-offset (SWIFT DScale / FScale, integers) --------------------------


def write(path, name, data, **kwargs):
    """Write a chunked dataset with h5py's high-level options."""
    with h5py.File(path, "a") as f:
        f.create_dataset(
            name,
            data=data,
            chunks=(min(CHUNK, data.shape[0]),) + data.shape[1:],
            **kwargs,
        )


def write_scaleoffset(path, name, data, digits, fill=None, file_dtype=None):
    """
    Write a scale-offset dataset as SWIFT does.

    The filters are scale-offset, shuffle, deflate (level 4) and Fletcher-32;
    h5py's high-level API refuses Fletcher-32 with scale-offset.
    """
    file_dtype = np.dtype(file_dtype or data.dtype)
    with h5py.File(path, "a") as f:
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((CHUNK,) + data.shape[1:])
        if fill is not None:
            dcpl.set_fill_value(np.array(fill, dtype=file_dtype))
        if file_dtype.kind == "f":
            dcpl.set_scaleoffset(h5py.h5z.SO_FLOAT_DSCALE, digits)
        else:
            dcpl.set_scaleoffset(h5py.h5z.SO_INT, digits)
        dcpl.set_shuffle()
        dcpl.set_deflate(4)
        dcpl.set_fletcher32()
        space = h5py.h5s.create_simple(data.shape)
        tid = h5py.h5t.py_create(file_dtype)
        d = h5py.Dataset(h5py.h5d.create(f.id, name.encode(), tid, space, dcpl))
        d[...] = data


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("digits", [0, 2, 6])
@pytest.mark.parametrize("cols", [1, 3])
def test_scaleoffset_float(tmp_path, dtype, digits, cols):
    """Check scaleoffset float."""
    rng = np.random.default_rng(digits + cols)
    shape = (N, cols) if cols > 1 else (N,)
    data = (rng.uniform(-50, 150, shape)).astype(dtype)
    write_scaleoffset(tmp_path / "f.hdf5", "x", data, digits, fill=0)
    assert_same(tmp_path / "f.hdf5", "x")


def test_scaleoffset_fill_value_and_constant_chunks(tmp_path):
    """Check scaleoffset fill value and constant chunks."""
    data = np.random.default_rng(3).uniform(0, 5, N)
    data[100:400] = 0.0  # the fill value: stored as the all-ones sentinel
    data[2000:3000] = 1.25  # a whole chunk with one value: minbits == 0
    write_scaleoffset(tmp_path / "f.hdf5", "x", data, 6, fill=0.0)
    assert_same(tmp_path / "f.hdf5", "x")


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint32, np.uint64, np.int16])
def test_scaleoffset_int(tmp_path, dtype):
    """Check scaleoffset int."""
    rng = np.random.default_rng(7)
    info = np.iinfo(dtype)
    lo = max(info.min // 2, -(2**39)) if info.min < 0 else 0
    data = rng.integers(
        lo,
        lo + min(2 ** (8 * np.dtype(dtype).itemsize - 2), 2**40),
        N,
        dtype=dtype,
        endpoint=False,
    )
    data[2000:3000] = data[2000]  # constant chunk
    write_scaleoffset(tmp_path / "f.hdf5", "x", data, 0, fill=5)
    assert_same(tmp_path / "f.hdf5", "x")


# --- plain shuffle + gzip ---------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint8])
@pytest.mark.parametrize("shuffle", [True, False])
def test_plain(tmp_path, dtype, shuffle):
    """Check plain."""
    data = (np.random.default_rng(2).uniform(0, 200, (N, 3))).astype(dtype)
    write(
        tmp_path / "f.hdf5",
        "x",
        data,
        shuffle=shuffle,
        compression="gzip",
        compression_opts=4,
        fletcher32=True,
    )
    assert_same(tmp_path / "f.hdf5", "x")


# --- everything else goes to h5py -------------------------------------------


def test_big_endian_uses_h5py(tmp_path):
    """Check big endian uses h5py."""
    data = np.random.default_rng(4).uniform(0, 1, N)
    write_scaleoffset(tmp_path / "f.hdf5", "x", data, 6, fill=0.0, file_dtype=">f8")
    assert_same(tmp_path / "f.hdf5", "x", fast=False)


def test_integer_nbit_with_an_offset(tmp_path):
    # HDF5 converts an integer stored from bit 4 to a native integer by
    # dropping the offset, which is what unpacking the packed bits gives
    """Check integer nbit with an offset."""
    data = np.random.default_rng(5).integers(0, 2**20, N).astype(np.int64) << 4
    t = h5py.h5t.STD_I64LE.copy()
    t.set_precision(36)
    t.set_offset(4)
    write_nbit(tmp_path / "f.hdf5", "x", data, t)
    assert_same(tmp_path / "f.hdf5", "x")


def test_other_compressors_and_contiguous_use_h5py(tmp_path):
    """Check other compressors and contiguous use h5py."""
    data = np.random.default_rng(6).uniform(0, 1, N)
    write(tmp_path / "f.hdf5", "lzf", data, compression="lzf")
    with h5py.File(tmp_path / "f.hdf5", "a") as f:
        f.create_dataset("contiguous", data=data)
    assert_same(tmp_path / "f.hdf5", "lzf", fast=False)
    assert_same(tmp_path / "f.hdf5", "contiguous", fast=False)


def test_chunk_with_skipped_filter_uses_h5py(tmp_path):
    # A chunk written with the deflate filter skipped (filter mask bit 2)
    """Check chunk with skipped filter uses h5py."""
    data = np.random.default_rng(8).uniform(0, 1, N).astype(np.float32)
    write(tmp_path / "f.hdf5", "x", data, compression="gzip", compression_opts=4)
    raw = data[:CHUNK].tobytes()
    with h5py.File(tmp_path / "f.hdf5", "a") as f:
        f["x"].id.write_direct_chunk((0,), raw, filter_mask=1)
    assert_same(tmp_path / "f.hdf5", "x")


# --- checksums ---------------------------------------------------------------


def test_corrupt_chunk_raises(tmp_path):
    """Check corrupt chunk raises."""
    data = np.random.default_rng(9).uniform(0, 1, N).astype(np.float32)
    write_nbit(
        tmp_path / "f.hdf5", "x", data, nbit_type(h5py.h5t.IEEE_F32LE, 9, 8, 127)
    )
    with h5py.File(tmp_path / "f.hdf5", "a") as f:
        mask, raw = f["x"].id.read_direct_chunk((CHUNK,))
        bad = bytearray(raw)
        bad[len(bad) // 2] ^= 0x10
        f["x"].id.write_direct_chunk((CHUNK,), bytes(bad), filter_mask=mask)
    with h5py.File(tmp_path / "f.hdf5") as f, pytest.raises(OSError):
        f["x"][CHUNK : 2 * CHUNK]  # HDF5 itself rejects the chunk
    with pytest.raises(fd.ChecksumError):
        fast_read(tmp_path / "f.hdf5", "x", [(CHUNK, CHUNK + 10)])
    assert_same(
        tmp_path / "f.hdf5", "x", [(0, CHUNK), (2 * CHUNK, N)]
    )  # other chunks still read


def test_fletcher32_matches_hdf5_on_odd_lengths():
    # The stored checksum of every chunk in these files is HDF5's; check odd
    # byte counts and the 360-word folding directly against a slow reference
    """Check fletcher32 matches hdf5 on odd lengths."""

    def slow(data):
        """Compute HDF5's Fletcher-32 checksum the slow, obvious way."""
        s1 = s2 = 0
        words = [(data[i] << 8) | data[i + 1] for i in range(0, len(data) - 1, 2)]
        for i in range(0, len(words), 360):
            for w in words[i : i + 360]:
                s1 += w
                s2 += s1
            s1 = (s1 & 0xFFFF) + (s1 >> 16)
            s2 = (s2 & 0xFFFF) + (s2 >> 16)
        if len(data) % 2:
            s1 += data[-1] << 8
            s2 += s1
            s1 = (s1 & 0xFFFF) + (s1 >> 16)
            s2 = (s2 & 0xFFFF) + (s2 >> 16)
        s1 = (s1 & 0xFFFF) + (s1 >> 16)
        s2 = (s2 & 0xFFFF) + (s2 >> 16)
        return (s2 << 16) | s1

    rng = np.random.default_rng(10)
    for n in (1, 2, 3, 719, 720, 721, 5001):
        data = rng.integers(0, 256, n, dtype=np.uint8)
        assert int(fd.fletcher32(data, n)) == slow(data.tolist())


# --- virtual datasets --------------------------------------------------------


def test_virtual_dataset_over_files(tmp_path):
    """Check virtual dataset over files."""
    rng = np.random.default_rng(11)
    pieces = [sample(n, 6, rng, np.float32) for n in (1500, 37, 2784)]
    layout = h5py.VirtualLayout(
        shape=(sum(len(p) for p in pieces), 6), dtype=np.float32
    )
    start = 0
    for i, p in enumerate(pieces):
        write_nbit(
            tmp_path / f"part.{i}.hdf5",
            "PartType0/x",
            p,
            nbit_type(h5py.h5t.IEEE_F32LE, 9, 8, 127),
        )
        layout[start : start + len(p)] = h5py.VirtualSource(
            f"part.{i}.hdf5", "PartType0/x", shape=p.shape
        )
        start += len(p)
    with h5py.File(tmp_path / "virtual.hdf5", "w") as f:
        f.create_virtual_dataset("PartType0/x", layout)
    ranges = [(0, 10), (1400, 1600), (1530, 1540), (2000, start)]
    assert_same(tmp_path / "virtual.hdf5", "PartType0/x", ranges)


# --- every lossy scheme SWIFT can write (src/io_compression.c), in the
#     filter orders SWIFT uses (single_io.c, distributed_io.c,
#     common_io_cells.c) ---------------------------------------------------

SWIFT_FLOATS = {  # name: (base type, exponent bits, mantissa bits)
    "FMantissa9": (h5py.h5t.NATIVE_FLOAT, 8, 9),
    "FMantissa13": (h5py.h5t.NATIVE_FLOAT, 8, 13),
    "BFloat16": (h5py.h5t.NATIVE_FLOAT, 8, 7),
    "HalfFloat": (h5py.h5t.NATIVE_FLOAT, 5, 10),
    "DMantissa9": (h5py.h5t.NATIVE_DOUBLE, 11, 9),
    "DMantissa13": (h5py.h5t.NATIVE_DOUBLE, 11, 13),
    "DMantissa21": (h5py.h5t.NATIVE_DOUBLE, 11, 21),
}
ORDERS = {
    "compressed": [
        "lossy",
        "shuffle",
        "deflate",
        "fletcher32",
    ],  # snapshot_compression > 0
    "uncompressed": ["lossy", "fletcher32"],  # snapshot_compression = 0
}


def swift_float_type(base, e_size, m_size):
    """Make a float type as SWIFT's io_compression.c does."""
    t = base.copy()
    e_pos, m_pos = m_size, 0
    s_pos = e_pos + e_size
    t.set_fields(s_pos, e_pos, e_size, m_pos, m_size)
    t.set_offset(0)
    t.set_precision(m_size + e_size + 1)
    t.set_size(base.get_size())
    t.set_ebias((1 << (e_size - 1)) - 1)
    return t


def write_swift(path, name, data, file_type, order, lossy):
    """Write a dataset with filters in the given order, as SWIFT does."""
    with h5py.File(path, "a") as f:
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((min(CHUNK, data.shape[0]),) + data.shape[1:])
        for step in order:
            if step == "lossy":
                lossy(dcpl)
            elif step == "shuffle":
                dcpl.set_shuffle()
            elif step == "deflate":
                dcpl.set_deflate(4)
            else:
                dcpl.set_fletcher32()
        space = h5py.h5s.create_simple(data.shape)
        try:
            d = h5py.Dataset(
                h5py.h5d.create(f.id, name.encode(), file_type, space, dcpl)
            )
        except ValueError as error:
            if "unusual datatype" in str(error):
                pytest.skip(
                    "HDF5 refuses to create this type without relaxed integrity checks"
                )
            raise
        d[...] = data


def nbit(dcpl):
    """Add the N-bit filter."""
    dcpl.set_filter(h5py.h5z.FILTER_NBIT, 0, ())


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("scheme", SWIFT_FLOATS)
@pytest.mark.parametrize("cols", [1, 3])
def test_swift_float_schemes(tmp_path, scheme, order, cols):
    """Check SWIFT float schemes."""
    base, e_size, m_size = SWIFT_FLOATS[scheme]
    dtype = np.float32 if base == h5py.h5t.NATIVE_FLOAT else np.float64
    rng = np.random.default_rng(len(scheme) + cols)
    shape = (N, cols) if cols > 1 else (N,)
    top = 2.0 ** ((1 << (e_size - 1)) - 1)  # largest power of two the scheme holds
    data = rng.standard_normal(shape) * top ** rng.uniform(
        -1.05, 1.05, shape
    )  # includes over- and underflow
    data.flat[::19] = 0.0
    data.flat[::29] = -0.0
    data.flat[7] = np.inf
    data.flat[8] = -np.inf
    data.flat[9] = np.nan
    write_swift(
        tmp_path / "f.hdf5",
        "x",
        data.astype(dtype),
        swift_float_type(base, e_size, m_size),
        ORDERS[order],
        nbit,
    )
    assert_same(tmp_path / "f.hdf5", "x")


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("bits", [32, 36, 40, 44, 48, 56])
@pytest.mark.parametrize("signed", [True, False])
def test_swift_nbit_integers(tmp_path, bits, order, signed):
    # Nbit32..56: NATIVE_LLONG with its precision cut (ParticleIDs etc.)
    """Check SWIFT nbit integers."""
    rng = np.random.default_rng(bits)
    t = (h5py.h5t.NATIVE_INT64 if signed else h5py.h5t.NATIVE_UINT64).copy()
    t.set_precision(bits)
    t.set_offset(0)
    if signed:
        data = rng.integers(-(2 ** (bits - 1)), 2 ** (bits - 1), N, dtype=np.int64)
    else:
        data = rng.integers(0, 2**bits, N, dtype=np.uint64)
    write_swift(tmp_path / "f.hdf5", "x", data, t, ORDERS[order], nbit)
    assert_same(tmp_path / "f.hdf5", "x")


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("digits", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_swift_dscale(tmp_path, digits, dtype, order):
    """Check SWIFT dscale."""
    rng = np.random.default_rng(digits)
    data = rng.uniform(0, 25, (N, 3)).astype(dtype)  # e.g. coordinates in a 25 Mpc box
    file_type = h5py.h5t.py_create(np.dtype(dtype))

    def scaleoffset(dcpl):
        """Add the D-scale scale-offset filter."""
        dcpl.set_scaleoffset(h5py.h5z.SO_FLOAT_DSCALE, digits)

    write_swift(tmp_path / "f.hdf5", "x", data, file_type, ORDERS[order], scaleoffset)
    assert_same(tmp_path / "f.hdf5", "x")


@pytest.mark.parametrize("dtype", [np.int64, np.float64, np.int32])
def test_swift_cell_metadata_order(tmp_path, dtype):
    # common_io_cells.c: fletcher32, then shuffle, then deflate
    """Check SWIFT cell metadata order."""
    data = (np.random.default_rng(3).uniform(0, 1e6, (N, 3))).astype(dtype)
    write_swift(
        tmp_path / "f.hdf5",
        "x",
        data,
        h5py.h5t.py_create(np.dtype(dtype)),
        ["fletcher32", "shuffle", "deflate"],
        None,
    )
    assert_same(tmp_path / "f.hdf5", "x")


@pytest.mark.parametrize(
    "source,declared",
    [
        (np.float32, np.float64),
        (np.float64, np.float32),
        (np.int64, np.uint64),
        (np.uint64, np.int64),
        (np.int64, np.int32),
    ],
)
def test_virtual_dataset_with_another_type(tmp_path, source, declared):
    # SWIFT's virtual files can declare a type other than the sub-files'
    # (FLAMINGO: float32 data as float64; COLIBRE: int64 IDs as uint64);
    # h5py converts to the declared type, clamping integers that do not fit
    """Check virtual dataset with another type."""
    rng = np.random.default_rng(12)
    if np.dtype(source).kind == "f":
        pieces = [sample(n, 1, rng, source) for n in (1200, 900)]
    else:
        info = np.iinfo(source)
        pieces = [
            rng.integers(max(info.min, -(2**40)), min(info.max, 2**40), n).astype(
                source
            )
            for n in (1200, 900)
        ]
        pieces[0][:5] = [info.min, info.max, -1 if info.min < 0 else 0, 2**33, 7]
    t = (
        nbit_type(h5py.h5t.IEEE_F32LE, 9, 8, 127)
        if source == np.float32
        else h5py.h5t.py_create(np.dtype(source))
    )
    layout = h5py.VirtualLayout(shape=(sum(len(p) for p in pieces),), dtype=declared)
    start = 0
    for i, p in enumerate(pieces):
        if source == np.float32:
            write_nbit(tmp_path / f"part.{i}.hdf5", "x", p, t)
        else:
            write(
                tmp_path / f"part.{i}.hdf5",
                "x",
                p,
                shuffle=True,
                compression="gzip",
                fletcher32=True,
            )
        layout[start : start + len(p)] = h5py.VirtualSource(
            f"part.{i}.hdf5", "x", shape=p.shape
        )
        start += len(p)
    with h5py.File(tmp_path / "virtual.hdf5", "w") as f:
        f.create_virtual_dataset("x", layout)
    assert_same(tmp_path / "virtual.hdf5", "x", [(0, 7), (1000, 1500), (1900, start)])


@pytest.mark.parametrize("columns", [np.s_[:], 0, 2, np.s_[1:3], [0, 2]])
def test_column_selection(tmp_path, columns):
    # swiftsimio reads named columns (e.g. one dust grain type) one at a time
    """Check column selection."""
    data = sample(N, 3, np.random.default_rng(13), np.float32)
    write_nbit(
        tmp_path / "f.hdf5", "x", data, nbit_type(h5py.h5t.IEEE_F32LE, 9, 8, 127)
    )
    assert_same(tmp_path / "f.hdf5", "x", columns=columns)


def test_output_type_is_converted(tmp_path):
    """Check output type is converted."""
    data = sample(N, 1, np.random.default_rng(14), np.float32)
    write_nbit(
        tmp_path / "f.hdf5", "x", data, nbit_type(h5py.h5t.IEEE_F32LE, 9, 8, 127)
    )
    ours = fast_read(tmp_path / "f.hdf5", "x", RANGES, output_type=np.float64)
    assert ours.dtype == np.float64
    assert np.array_equal(
        ours, reference(tmp_path / "f.hdf5", "x").astype(np.float64), equal_nan=True
    )


def test_disabled_and_small_reads_use_h5py(tmp_path, monkeypatch):
    """Check disabled and small reads use h5py."""
    data = sample(N, 1, np.random.default_rng(15), np.float32)
    write_nbit(
        tmp_path / "f.hdf5", "x", data, nbit_type(h5py.h5t.IEEE_F32LE, 9, 8, 127)
    )
    monkeypatch.setattr(fd, "ENABLED", False)
    assert fast_read(tmp_path / "f.hdf5", "x", RANGES) is None
    monkeypatch.setattr(fd, "ENABLED", True)
    monkeypatch.setattr(fd, "MINIMUM_ELEMENTS", 10**9)
    assert fast_read(tmp_path / "f.hdf5", "x", RANGES) is None
    ref = reference(tmp_path / "f.hdf5", "x")
    with h5py.File(tmp_path / "f.hdf5") as f:  # swiftsimio still reads it, with h5py
        through = read_ranges_from_file(
            f["x"], np.array(RANGES), ref.shape, f["x"].dtype
        )
    assert through.tobytes() == ref.tobytes()


def test_unsupported_virtual_mapping_uses_h5py(tmp_path):
    # a virtual dataset whose source rows are interleaved (not SWIFT's layout)
    """Check unsupported virtual mapping uses h5py."""
    data = sample(2000, 1, np.random.default_rng(16), np.float32)
    write_nbit(
        tmp_path / "part.hdf5", "x", data, nbit_type(h5py.h5t.IEEE_F32LE, 9, 8, 127)
    )
    layout = h5py.VirtualLayout(shape=(2000,), dtype=np.float32)
    source = h5py.VirtualSource("part.hdf5", "x", shape=(2000,))
    layout[0:2000:2] = source[0:1000]
    layout[1:2000:2] = source[1000:2000]
    with h5py.File(tmp_path / "virtual.hdf5", "w") as f:
        f.create_virtual_dataset("x", layout)
    assert_same(tmp_path / "virtual.hdf5", "x", [(0, 2000)], fast=False)


def test_chunks_not_holding_whole_rows_use_h5py(tmp_path):
    # e.g. a (N, 3) dataset chunked one column at a time
    """Check chunks not holding whole rows use h5py."""
    data = np.random.default_rng(17).uniform(0, 1, (N, 3)).astype(np.float32)
    with h5py.File(tmp_path / "f.hdf5", "w") as f:
        f.create_dataset(
            "x", data=data, chunks=(CHUNK, 1), compression="gzip", shuffle=True
        )
    assert_same(tmp_path / "f.hdf5", "x", fast=False)
    assert_same(tmp_path / "f.hdf5", "x", fast=False, columns=1)


def test_empty_dataset(tmp_path):
    """Check empty dataset."""
    with h5py.File(tmp_path / "f.hdf5", "w") as f:
        f.create_dataset(
            "x",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float32,
            chunks=(CHUNK, 3),
            compression="gzip",
        )
        out = fd.read_ranges(f["x"], np.zeros((0, 2), np.int64), (0, 3), np.float32)
        assert out is None or (out.shape == (0, 3) and out.dtype == np.float32)
