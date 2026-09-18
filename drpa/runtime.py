"""Portable host-resource detection; does not allocate tensors or start jobs."""
from pathlib import Path
import os

def memory_limit_bytes():
    limits=[]
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            value=Path(name).read_text().strip()
            if value.isdigit() and 0<int(value)<2**60:limits.append(int(value))
        except OSError:pass
    try:
        limits.append(os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES"))
    except (ValueError,OSError,AttributeError):
        try:
            import psutil
            limits.append(psutil.virtual_memory().total)
        except ImportError:pass
    if not limits:raise RuntimeError("Unable to determine available host memory")
    return min(limits)
