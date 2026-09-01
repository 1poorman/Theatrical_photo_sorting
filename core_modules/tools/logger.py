# -*- coding: utf-8 -*-
"""
logger.py - 集中式运行日志

功能：
- 日志时间戳统一使用北京时间 (UTC+8)，与服务器系统时区无关
- 按天滚动：日志文件为 logs/server.log，每天北京时间 0 点切分为 server.log.YYYY-MM-DD，默认保留 30 天
- 单文件过大时触发备份保护（maxBytes 由自定义双轮转处理器保证）
- 同时输出到控制台，接管 uvicorn 日志，重定向 sys.stdout 使 print() 也落盘
- 提供 StepTimer：关键动作分步计时，便于性能分析与故障定位
"""
import functools
import logging
import os
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from logging.handlers import TimedRotatingFileHandler

# 项目根 = core_modules/tools/ 向上三级
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "server.log")
_BEIJING_TZ = timezone(timedelta(hours=8))  # 北京时间 UTC+8
_BACKUP_COUNT = 30          # 按天滚动保留天数
_MAX_BYTES = 100 * 1024 * 1024  # 单个日志文件大小保护上限
_APP_LOGGER_NAME = "theatrical"


def _beijing_time(*args):
    """日志时间戳统一转换为北京时间（供 logging.Formatter.converter 使用）。"""
    return datetime.now(_BEIJING_TZ).timetuple()


def _next_beijing_midnight_epoch():
    """计算北京时间次日 0 点对应的 epoch 秒（滚动切分时间点）。"""
    now = datetime.now(_BEIJING_TZ)
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.timestamp()


class _SafeTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    按北京时间 0 点滚动的文件处理器，附带大小保护：
    - 跨天或超过 _MAX_BYTES 时滚动，避免极端流量下单日文件过大
    - 不依赖系统时区，切分点始终为北京时间 0 点
    """

    def __init__(self, filename, max_bytes=_MAX_BYTES, **kwargs):
        super().__init__(filename, **kwargs)
        self._max_bytes = max_bytes
        self._day_prefix = datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d")
        # 覆盖父类按系统时区计算的滚动时间点
        self.rolloverAt = _next_beijing_midnight_epoch()
        # 启动时补滚动：文件末行不是今天的日志则先切档，避免跨重启混日期
        self._rotate_if_stale()

    def _last_line_date(self):
        """读取日志文件末行行首的 YYYY-MM-DD（北京时间时间戳），失败返回 None"""
        try:
            path = self.baseFilename
            size = os.path.getsize(path)
            if size == 0:
                return None
            with open(path, 'rb') as f:
                f.seek(-min(1024, size), 2)
                tail = f.read().decode('utf-8', errors='ignore')
            lines = [l for l in tail.strip().splitlines() if l.strip()]
            if not lines:
                return None
            m = re.match(r'^(\d{4}-\d{2}-\d{2})', lines[-1].strip())
            return m.group(1) if m else None
        except Exception:
            return None

    def _should_rotate_by_size(self):
        if self._max_bytes <= 0 or self.stream is None:
            return False
        try:
            self.stream.seek(0, 2)
            return self.stream.tell() >= self._max_bytes
        except Exception:
            return False

    def shouldRollover(self, record):
        today = datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d")
        if today != self._day_prefix or time.time() >= self.rolloverAt or self._should_rotate_by_size():
            self._day_prefix = today
            return True
        return False

    def doRollover(self):
        """完全接管滚动：备份文件名 = 被切档内容末行的日期（北京时间）。

        不用父类实现的原因：父类用系统时区解释 rolloverAt 生成文件名，
        服务器时区非北京时间时备份名会偏差一天。
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        suffix = self._last_line_date() or datetime.now(_BEIJING_TZ).strftime("%Y-%m-%d")
        dfn = self.baseFilename + "." + suffix
        if os.path.exists(dfn):
            try:
                os.remove(dfn)
            except OSError:
                pass
        try:
            os.rename(self.baseFilename, dfn)
        except OSError:
            pass
        if self.backupCount > 0:
            for f in self.getFilesToDelete():
                try:
                    os.remove(f)
                except OSError:
                    pass
        if not self.delay:
            self.stream = open(self.baseFilename, self.mode, encoding=self.encoding)
        self.rolloverAt = _next_beijing_midnight_epoch()

    def _rotate_if_stale(self):
        """启动时检查：日志文件末行日期不是今天则立即滚动。

        解决跨重启混日期问题：TimedRotatingFileHandler 只在运行中有写入时才滚动，
        若服务在午夜前后启停，不同日期的内容会混进同一个文件。
        """
        try:
            if self._last_line_date() and self._last_line_date() != self._day_prefix:
                self.doRollover()
        except Exception:
            pass  # 任何异常都不阻塞日志初始化


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

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] [%(module)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    formatter.converter = _beijing_time  # 时间戳使用北京时间，与系统时区无关

    file_handler = _SafeTimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
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
    logger.info("Logging initialized. Log file: %s (daily rotation, keep %d days)", LOG_FILE, _BACKUP_COUNT)
    return logger, LOG_FILE


def get_app_logger():
    """获取应用 logger（用于业务代码记录日志）。"""
    return logging.getLogger(_APP_LOGGER_NAME)


class StepTimer:
    """
    分步计时器：为一次请求/任务内的各步骤记录耗时，结束后输出汇总。

    用法:
        timer = StepTimer(logger, task_id="abc")
        with timer.step("detect"):
            ...
        timer.log_summary()   # 输出各步骤耗时与总耗时
    """

    def __init__(self, logger=None, task_id=""):
        self.logger = logger or get_app_logger()
        self.task_id = task_id or datetime.now(_BEIJING_TZ).strftime("%H%M%S%f")[:10]
        self.steps = []  # [(name, seconds), ...]
        self._t0 = time.perf_counter()

    @contextmanager
    def step(self, name):
        """上下文管理器：记录单个步骤的 START / DONE / FAIL 与耗时。"""
        prefix = f"[{self.task_id}] [{name}]"
        self.logger.info("%s START", prefix)
        start = time.perf_counter()
        try:
            yield
        except Exception as e:
            duration = time.perf_counter() - start
            self.logger.error("%s FAIL (%.2fs): %s", prefix, duration, e, exc_info=True)
            self.steps.append((name, duration, "FAIL"))
            raise
        duration = time.perf_counter() - start
        self.steps.append((name, duration, "OK"))
        self.logger.info("%s DONE in %.2fs", prefix, duration)

    def elapsed(self):
        return time.perf_counter() - self._t0

    def log_summary(self, extra=""):
        """输出整个任务的耗时汇总（各步骤 + 总耗时），便于事后性能分析。"""
        total = self.elapsed()
        failed = [s for s in self.steps if s[2] == "FAIL"]
        detail = "; ".join(f"{name}={sec:.2f}s({status})" for name, sec, status in self.steps)
        status = "FAIL" if failed else "OK"
        self.logger.info(
            "[%s] SUMMARY %s | total=%.2fs | %s%s",
            self.task_id, status, total, detail,
            f" | {extra}" if extra else "",
        )


def log_key_action(action, **kwargs):
    """记录关键动作（模型加载、索引构建等），kwargs 以 k=v 追加到日志。"""
    logger = get_app_logger()
    extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info("[ACTION] %s%s", action, f" | {extras}" if extras else "")


def timed(action_name=None):
    """装饰器：函数级耗时日志（服务启动、模型初始化等场景）。"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            name = action_name or func.__name__
            logger = get_app_logger()
            logger.info("[ACTION] %s START", name)
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                logger.error("[ACTION] %s FAIL (%.2fs): %s", name, time.perf_counter() - start, e, exc_info=True)
                raise
            logger.info("[ACTION] %s DONE in %.2fs", name, time.perf_counter() - start)
            return result
        return wrapper
    return decorator
