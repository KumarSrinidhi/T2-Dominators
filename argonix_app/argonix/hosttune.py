"""Per-process host tuning for Windows GPU hosts, applied by run.py at start-up (no-ops elsewhere).

Measured on the RTX 4050 host (i7-14650HX: 8 performance + 8 efficiency cores, Windows 11), 2026-09-26:
- `fine_timers`: Windows rounds thread waits up to its 15.6 ms tick (a Condition.wait(0.008) took 15 ms). 1 ms ticks
  gave +5-14 % detector throughput and a 20-40 % lower p95 in a 4-stream TensorRT case.
- `prefer_performance_cores`: pinned to the performance cores, one YOLO26s frame went from 8.0 to 4.9 ms of TensorRT
  inference and 45 to 26 ms with the fp16 .pt model (kernel launches and post-processing are CPU-bound; Windows had
  been running the process on efficiency cores). Opt out with ARGONIX_ANY_CORES=1.
Child processes (ffmpeg encoders) inherit the affinity.
"""
from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes
from typing import Optional


def fine_timers() -> bool:
    if sys.platform != "win32":
        return False
    try:
        return ctypes.windll.winmm.timeBeginPeriod(1) == 0
    except (AttributeError, OSError):
        return False


def performance_cpus() -> Optional[list[int]]:
    """Logical CPU numbers (group 0) of the highest efficiency class, or None when every core is the same kind or
    the information is unavailable (non-Windows, one processor group only is handled)."""
    if sys.platform != "win32":
        return None
    try:
        k = ctypes.windll.kernel32
        need = wintypes.ULONG(0)
        k.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
        buf = (ctypes.c_ubyte * need.value)()
        if not k.GetSystemCpuSetInformation(buf, need, ctypes.byref(need), None, 0):
            return None
    except (AttributeError, OSError):
        return None
    return _top_class_cpus(bytes(buf))


def _top_class_cpus(raw: bytes) -> Optional[list[int]]:
    """Parse SYSTEM_CPU_SET_INFORMATION records: DWORD Size, DWORD Type, then CpuSet {DWORD Id, WORD Group,
    BYTE LogicalProcessorIndex, BYTE CoreIndex, BYTE LastLevelCacheIndex, BYTE NumaNodeIndex, BYTE EfficiencyClass}."""
    off, cpus = 0, []
    while off + 8 <= len(raw):
        size = int.from_bytes(raw[off:off + 4], "little")
        if size <= 0:
            break
        if int.from_bytes(raw[off + 4:off + 8], "little") == 0 and off + 19 <= len(raw):  # CpuSetInformation
            group = int.from_bytes(raw[off + 12:off + 14], "little")
            if group == 0:
                cpus.append((raw[off + 14], raw[off + 18]))  # (logical index, efficiency class)
        off += size
    classes = {c for _, c in cpus}
    if len(classes) < 2:
        return None
    top = max(classes)
    return sorted(i for i, c in cpus if c == top)


def prefer_performance_cores() -> Optional[list[int]]:
    """Restrict this process to the performance cores on a hybrid CPU. Returns the CPUs used, or None if unchanged."""
    if os.environ.get("ARGONIX_ANY_CORES") == "1":
        return None
    cpus = performance_cpus()
    if not cpus:
        return None
    try:
        import psutil
        psutil.Process().cpu_affinity(cpus)
        return cpus
    except Exception:  # noqa: BLE001 - tuning is best effort
        return None
