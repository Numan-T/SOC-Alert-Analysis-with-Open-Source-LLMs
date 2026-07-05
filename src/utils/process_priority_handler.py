import sys
import psutil
import os


def set_high_process_prio():
    """Sets priocess priority to high, to mitigate ressource throttling because of other processes (HIGH_PRIORITY_CLASS in Windows)."""
    try:
        p = psutil.Process(os.getpid())

        if os.name == 'nt':  # Windows
            p.nice(psutil.HIGH_PRIORITY_CLASS)
        else:  # Linux
            p.nice(-10)
        print("Set process priority to high.")

    except Exception as e:
        print(f"Warning: Couldn't set process priority to high: {e}")
