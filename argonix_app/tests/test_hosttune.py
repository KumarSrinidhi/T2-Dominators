"""Host tuning: hybrid-CPU detection from SYSTEM_CPU_SET_INFORMATION records (parsed without calling Windows)."""
import struct

from argonix.hosttune import _top_class_cpus


def rec(logical: int, eff: int, group: int = 0) -> bytes:
    body = struct.pack("<IHBBBBB", 256 + logical, group, logical, logical // 2, 0, 0, eff)
    return struct.pack("<II", 32, 0) + body + b"\0" * (32 - 8 - len(body))


def test_hybrid_cpu_keeps_only_the_performance_class():
    raw = b"".join(rec(i, 1) for i in range(16)) + b"".join(rec(i, 0) for i in range(16, 24))
    assert _top_class_cpus(raw) == list(range(16))


def test_uniform_cpu_and_other_groups_change_nothing():
    assert _top_class_cpus(b"".join(rec(i, 0) for i in range(12))) is None
    assert _top_class_cpus(b"".join(rec(i, 0) for i in range(4)) + rec(0, 1, group=1)) is None
    assert _top_class_cpus(b"") is None
