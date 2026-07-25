import math

import pytest

from unitccl_cli.schema import BenchRecord, _parse_test_column, size_to_bytes


def test_size_to_bytes():
    assert size_to_bytes("1kB") == 1024
    assert size_to_bytes("4MB") == 4 * 1024**2
    assert size_to_bytes("64MB") == 64 * 1024**2


def test_size_to_bytes_invalid():
    with pytest.raises(ValueError):
        size_to_bytes("not-a-size")


def test_parse_test_column():
    algo, size_str = _parse_test_column("scaling/1kB_64MB/RING_Bcast/1kB")
    assert algo == "RING"
    assert size_str == "1kB"

    algo, size_str = _parse_test_column("scaling/1kB_64MB/BINE_Bcast/16kB")
    assert algo == "BINE"
    assert size_str == "16kB"


def test_bench_record_size_bytes():
    r = BenchRecord(collective="Bcast", algo="BINE", proto="SIMPLE", size_str="4MB", mean_ns=123.0)
    assert math.isclose(r.size_bytes, 4 * 1024**2)
