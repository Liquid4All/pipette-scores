"""Process memory measurement for scoring/serving progress logs."""

import resource
import sys


def rss_mb() -> float:
    """Current RSS in MB (reads /proc/self/status on Linux, falls back to ru_maxrss)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except FileNotFoundError:
        pass
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024
