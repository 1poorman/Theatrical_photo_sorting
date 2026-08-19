# -*- coding: utf-8 -*-
"""
logger.py - 集中式运行日志

- 日志同时输出到控制台与 logs/server_<时间戳>.log（带轮转）
- 接管 uvicorn 各日志器，统一落盘
- 重定向 sys.stdout，使散落的 print() 也写入日志文件
"""
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
_APP_LOGGER_NAME = "theatrical"


class _StdoutTee:
    """将 stdout 同时写入原 stdout 与日志文件处理器，保证 print() 也落盘。"""

    def __init__(self, original, handler):
        self._original = original
        self._handler = handler

    def write(self, data):
        self._original.write(data)
        try:
            stream = self._handler.stream
            stream.write(data)
            stream.flush()
        except Exception:
            pass

    def flush(self):
        self._original.flush()
        try:
            self._handler.stream.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._original, "isatty", lambda: False)()


def setup_logging():
    """配置根日志与 uvicorn 日志，返回 (logger, log_file)。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"server_{timestamp}.log")

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    # 让 uvicorn 各日志器向 root 传播，统一落盘（uvicorn.run 传 log_config=None 时生效）
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.propagate = True
        lg.handlers = []

    # 把已有 stdout 的 print 也写入日志文件
    if not isinstance(sys.stdout, _StdoutTee):
        sys.stdout = _StdoutTee(sys.stdout, file_handler)

    logger = logging.getLogger(_APP_LOGGER_NAME)
    logger.info("Logging initialized. Log file: %s", log_file)
    return logger, log_file


def get_app_logger():
    """获取应用 logger（用于业务代码记录日志）。"""
    return logging.getLogger(_APP_LOGGER_NAME)
