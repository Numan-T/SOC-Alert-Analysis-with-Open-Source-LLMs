import time
from datetime import timedelta

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()

    def elapsed_seconds(self):
        return f"{(self.end - self.start):.2f}"

    def elapsed_hhmmss(self):
        return f"{timedelta(seconds=self.end - self.start)}"
    

def seconds_to_hhmmss(seconds):
    return f"{timedelta(seconds=int(float(seconds)))}"
