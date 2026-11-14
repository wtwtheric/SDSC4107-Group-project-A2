"""日志与计时（spec §4：run.log 记录本次运行）。"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logger(
    name: str,
    logfile: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """建一个同时往 stdout 和 run.log 写的 logger；重复调用会重置 handler。"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if logfile is not None:
        path = Path(logfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def human_secs(seconds: float) -> str:
    """秒数转成 12.3s / 1m 23s / 1h 05m。"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rest = divmod(seconds, 60.0)
    if minutes < 60:
        return f"{int(minutes)}m {rest:.0f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(minutes)}m"


class Stopwatch:
    """with Stopwatch() as sw: ...；结束后读 sw.elapsed（秒）。"""

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start
