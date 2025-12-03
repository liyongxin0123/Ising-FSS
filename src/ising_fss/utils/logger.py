# -*- coding: utf-8 -*-
"""
日志记录器 

实现功能：
    - 上下文管理: 支持 `with ExperimentLogger(...)` 语法，确保实验结束时正确关闭句柄并输出摘要。
    - 多进程安全: 可选集成 `QueueListener`，在多进程环境下安全地聚合日志流。
    - 指标追踪: 提供 `log_metric` 方法，将训练曲线（Loss, Accuracy 等）实时写入 JSONL 文件，便于后续可视化。
    - 格式化: 控制台输出支持彩色高亮，文件输出保持纯文本结构化格式。
"""

from __future__ import annotations

import logging
import sys
import time
import json
import atexit
import threading
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import multiprocessing as mp

try:
    import yaml  # Optional: pretty-print config
except Exception:
    yaml = None

from logging.handlers import (
    RotatingFileHandler,
    TimedRotatingFileHandler,
    QueueHandler,
    QueueListener,
)

__all__ = [
    'setup_logger', 'get_logger', 'log_experiment',
    'ExperimentLogger', 'ProgressLogger', 'PerformanceMonitor'
]

# -----------------------------------------------------------------------------
# Colored terminal formatter (only affects console handler)
# -----------------------------------------------------------------------------
class ColoredFormatter(logging.Formatter):
    """
    控制台彩色格式化器：仅临时包装 levelname 字段以添加颜色码，
    并在返回前恢复，避免对 record 做持久性修改。
    """
    COLORS = {
        'DEBUG': '\033[36m',    # cyan
        'INFO': '\033[32m',     # green
        'WARNING': '\033[33m',  # yellow
        'ERROR': '\033[31m',    # red
        'CRITICAL': '\033[35m', # magenta
    }
    RESET = '\033[0m'

    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        orig_levelname = record.levelname
        try:
            color = self.COLORS.get(orig_levelname)
            if color:
                # only touch levelname (temporary)
                record.levelname = f"{color}{orig_levelname}{self.RESET}"
            return super().format(record)
        finally:
            record.levelname = orig_levelname  # restore

# -----------------------------------------------------------------------------
# Handler 工厂
# -----------------------------------------------------------------------------
def _make_console_handler(level: int, use_color: bool, utc: bool) -> logging.Handler:
    datefmt = '%Y-%m-%d %H:%M:%S'
    if use_color:
        fmt = '%(asctime)s | %(levelname)s | %(message)s'
        formatter = ColoredFormatter(fmt, datefmt=datefmt)
    else:
        fmt = '%(asctime)s | %(levelname)-8s | %(message)s'
        formatter = logging.Formatter(fmt, datefmt=datefmt)
    if utc:
        formatter.converter = time.gmtime  # type: ignore
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    return ch

def _make_file_handler(log_path: Path, level: int, rotate: Optional[Dict[str, Any]], utc: bool) -> logging.Handler:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    datefmt = '%Y-%m-%d %H:%M:%S'
    fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    if rotate:
        if 'when' in rotate:
            fh = TimedRotatingFileHandler(
                str(log_path),
                when=rotate.get('when', 'D'),
                interval=int(rotate.get('interval', 1)),
                backupCount=int(rotate.get('backupCount', 14)),
                encoding='utf-8',
                utc=utc
            )
        else:
            fh = RotatingFileHandler(
                str(log_path),
                maxBytes=int(rotate.get('maxBytes', 10_000_000)),
                backupCount=int(rotate.get('backupCount', 5)),
                encoding='utf-8'
            )
    else:
        fh = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)  # keep detailed records; logger.level controls emission
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    if utc:
        formatter.converter = time.gmtime  # type: ignore
    fh.setFormatter(formatter)
    return fh

# -----------------------------------------------------------------------------
# 全局：跟踪启动的 QueueListener（按 logger.name 存储），便于重复 setup 时清理
# -----------------------------------------------------------------------------
_QUEUE_LISTENERS: Dict[str, Dict[str, Any]] = {}
_QUEUE_LOCK = threading.Lock()

# -----------------------------------------------------------------------------
# setup_logger / get_logger
# -----------------------------------------------------------------------------
def setup_logger(
    name: str = 'ising_fss',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    use_color: bool = True,
    utc: bool = False,
    rotate: Optional[Dict[str, Any]] = None,
    mp_safe: bool = False
) -> logging.Logger:
    """
    配置并返回 logger。重复调用会覆盖同名 logger 的 handlers，并停止旧 listener（若存在）。
    mp_safe=True 时使用 QueueHandler + QueueListener 以便多进程日志聚合。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清理先前注册的全局 QueueListener（若存在）
    with _QUEUE_LOCK:
        prev = _QUEUE_LISTENERS.get(name)
        if prev is not None:
            prev_listener = prev.get("listener")
            prev_queue = prev.get("queue")
            try:
                if prev_listener is not None:
                    prev_listener.stop()
            except Exception:
                pass
            # remove mapping
            _QUEUE_LISTENERS.pop(name, None)

    # 移除并关闭现有 handlers（避免重复输出）
    for h in list(logger.handlers):
        try:
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        except Exception:
            pass
    logger.propagate = False

    # 仅在真实终端时启用颜色
    use_color = bool(use_color and hasattr(sys.stdout, "isatty") and sys.stdout.isatty())

    console_handler = _make_console_handler(level, use_color, utc)
    file_handler = _make_file_handler(Path(log_file), level, rotate, utc) if log_file else None

    if mp_safe:
        # 使用 multiprocessing context 创建 queue 更健壮
        try:
            ctx = mp.get_context()
            queue = ctx.Queue(-1)
        except Exception:
            queue = mp.Queue(-1)
        qh = QueueHandler(queue)
        qh.setLevel(level)
        logger.addHandler(qh)

        # QueueListener 将实际写入控制台 + 文件（按 handlers 列表）
        handlers = [console_handler]
        if file_handler is not None:
            handlers.append(file_handler)

        listener = QueueListener(queue, *handlers, respect_handler_level=True)
        try:
            listener.start()
        except Exception:
            # 若启动失败也不要阻塞程序
            pass

        # 保存 listener 引用，便于后续关闭
        with _QUEUE_LOCK:
            _QUEUE_LISTENERS[name] = {"queue": queue, "listener": listener}

        # 便于 ExperimentLogger.finish 能够找到并停止（把 listener 挂到 logger）
        try:
            setattr(logger, "_queue_listener", listener)
        except Exception:
            pass

        # Register atexit to stop listener (idempotent)
        def _cleanup(listener_ref=listener):
            try:
                listener_ref.stop()
            except Exception:
                pass
        atexit.register(_cleanup)
    else:
        logger.addHandler(console_handler)
        if file_handler is not None:
            logger.addHandler(file_handler)

    return logger

def get_logger(name: str = 'ising_fss') -> logging.Logger:
    """获取 logger（若未 setup，返回同名 logger 对象，但不自动配置 handlers）。"""
    return logging.getLogger(name)

# -----------------------------------------------------------------------------
# ExperimentLogger：实验上下文、配置与指标落盘
# -----------------------------------------------------------------------------
class ExperimentLogger:
    """
    实验级别日志帮助类：
      - 为每个 experiment 创建独立 logger（name = f"exp_{experiment_name}"）和 log file；
      - 提供 log_config（YAML/JSON）、log_metric（JSONL）、log_epoch、log_results；
      - 上下文管理（with）与 finish()（确保关闭 listener）。
    """
    def __init__(self,
                 experiment_name: str,
                 output_dir: str = 'logs',
                 level: int = logging.INFO,
                 use_color: bool = True,
                 utc: bool = False,
                 rotate: Optional[Dict[str, Any]] = None,
                 mp_safe: bool = False,
                 metrics_jsonl: Optional[str] = None):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now(timezone.utc if utc else None).strftime('%Y%m%d_%H%M%S')
        log_file = self.output_dir / f"{experiment_name}_{ts}.log"

        self.logger = setup_logger(
            name=f"exp_{experiment_name}",
            level=level,
            log_file=str(log_file),
            use_color=use_color,
            utc=utc,
            rotate=rotate,
            mp_safe=mp_safe
        )
        self.start_time = time.time()
        self.utc = utc

        self.metrics: Dict[str, List] = {}
        self._metrics_lock = threading.Lock()
        if metrics_jsonl:
            self.metrics_path = Path(metrics_jsonl)
        else:
            self.metrics_path = self.output_dir / f"{experiment_name}_{ts}.metrics.jsonl"

        # 记录启动信息
        self.logger.info("=" * 70)
        self.logger.info("实验开始: %s", experiment_name)
        now_str = datetime.now(timezone.utc).isoformat() if utc else datetime.now().isoformat()
        self.logger.info("时间: %s", now_str)
        self.logger.info("=" * 70)

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            # log stack
            self.logger.error("实验异常：%s", ''.join(traceback.format_exception(exc_type, exc, tb)).rstrip())
        try:
            self.finish()
        except Exception:
            pass
        # do not suppress exceptions
        return False

    def log_config(self, config: dict):
        """优先 YAML（若可用），否则 JSON，每行单独 INFO 输出以保持清晰格式。"""
        self.logger.info("配置参数:")
        dumped = None
        if yaml is not None:
            try:
                dumped = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
            except Exception:
                dumped = None
        if dumped is None:
            dumped = json.dumps(config, ensure_ascii=False, indent=2)
        for line in dumped.rstrip().splitlines():
            self.logger.info("  %s", line)

    def log_metric(self, name: str, value: float, step: Optional[int] = None,
                   save: bool = True, extra: Optional[dict] = None):
        """记录单个标量指标到内存和可选 JSONL（线程安全追加）。"""
        self.metrics.setdefault(name, []).append((step, float(value)))
        if step is not None:
            self.logger.info("Step %s | %s: %.6f", str(step), name, float(value))
        else:
            self.logger.info("%s: %.6f", name, float(value))

        if save:
            rec = {"t": time.time(), "name": name, "value": float(value)}
            if step is not None:
                rec["step"] = int(step)
            if extra:
                try:
                    rec.update(extra)
                except Exception:
                    rec["extra"] = str(extra)
            with self._metrics_lock:
                # atomic append (simple, cross-platform)
                with open(self.metrics_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, **kwargs):
        self.logger.info("Epoch %3d | Train Loss: %.6f | Val Loss: %.6f", epoch, float(train_loss), float(val_loss))
        self.log_metric("train_loss", float(train_loss), step=epoch, save=True, extra=kwargs)
        self.log_metric("val_loss", float(val_loss), step=epoch, save=True, extra=kwargs)

    def log_results(self, results: dict):
        self.logger.info("\n" + "="*70)
        self.logger.info("实验结果:")
        for k, v in results.items():
            if isinstance(v, float):
                self.logger.info("  %s: %.6f", k, v)
            else:
                self.logger.info("  %s: %s", k, v)

    def finish(self):
        """结束并清理：记录耗时并尝试停止与该 logger 关联的 QueueListener（若有）。"""
        elapsed = time.time() - self.start_time
        self.logger.info("\n" + "="*70)
        self.logger.info("实验完成: %s", self.experiment_name)
        self.logger.info("总耗时: %.2f秒 (%.2f分钟)", elapsed, elapsed / 60.0)
        self.logger.info("=" * 70)

        # 尝试停止 listener：优先查找 logger._queue_listener，再检查全局映射
        try:
            listener = getattr(self.logger, "_queue_listener", None)
            if listener is not None:
                try:
                    listener.stop()
                except Exception:
                    pass
        except Exception:
            pass

        with _QUEUE_LOCK:
            rec = _QUEUE_LISTENERS.pop(self.logger.name, None)
            if rec is not None:
                try:
                    l = rec.get("listener")
                    if l is not None:
                        l.stop()
                except Exception:
                    pass

# -----------------------------------------------------------------------------
# 装饰器：自动注入 ExperimentLogger（并确保 finish）
# -----------------------------------------------------------------------------
def log_experiment(func):
    import inspect
    sig = inspect.signature(func)
    takes_logger = 'logger' in sig.parameters

    def wrapper(*args, **kwargs):
        exp_name = kwargs.get('name', func.__name__)
        el = ExperimentLogger(exp_name)
        try:
            if takes_logger and 'logger' not in kwargs:
                kwargs['logger'] = el
            return func(*args, **kwargs)
        except Exception:
            el.logger.error("实验失败", exc_info=True)
            raise
        finally:
            try:
                el.finish()
            except Exception:
                pass
    return wrapper

# -----------------------------------------------------------------------------
# ProgressLogger
# -----------------------------------------------------------------------------
class ProgressLogger:
    """按步数或时间间隔打印进度的简单工具。"""
    def __init__(self, total: int, desc: str = "Progress",
                 logger: Optional[logging.Logger] = None,
                 log_every_n: int = 10,
                 log_every_seconds: Optional[float] = None):
        self.total = int(total)
        self.desc = desc
        self.logger = logger or get_logger()
        self.log_every_n = max(1, int(log_every_n))
        self.log_every_seconds = float(log_every_seconds) if log_every_seconds is not None else None

        self.current = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time

    def update(self, n: int = 1):
        self.current = min(self.total, self.current + int(n))
        now = time.time()
        should = (self.current % self.log_every_n == 0) or (self.current >= self.total)
        if self.log_every_seconds is not None:
            should = should or ((now - self.last_log_time) >= self.log_every_seconds)
        if should:
            self._log_progress(now)
            self.last_log_time = now

    def _log_progress(self, now: Optional[float] = None):
        if now is None:
            now = time.time()
        elapsed = max(1e-9, now - self.start_time)
        percent = 100.0 * self.current / max(1, self.total)
        speed = self.current / elapsed
        remaining = max(0, self.total - self.current)
        eta = remaining / max(speed, 1e-9)
        self.logger.info(
            "%s: %d/%d (%.1f%%) | %.2f it/s | ETA: %.1fs",
            self.desc, self.current, self.total, percent, speed, eta
        )

    def finish(self):
        elapsed = max(1e-9, time.time() - self.start_time)
        speed = self.total / elapsed
        self.logger.info(
            "%s 完成! 总计: %d | 耗时: %.2fs | 速度: %.2f it/s",
            self.desc, self.total, elapsed, speed
        )

# -----------------------------------------------------------------------------
# PerformanceMonitor
# -----------------------------------------------------------------------------
class PerformanceMonitor:
    """轻量性能监控（计时器/计数器/摘要）。"""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger()
        self.timers: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}

    def start_timer(self, name: str):
        self.timers[name] = time.time()

    def stop_timer(self, name: str, log: bool = True):
        if name not in self.timers:
            self.logger.warning("计时器 '%s' 未启动", name)
            return None
        start = self.timers.pop(name, 0.0)
        elapsed = time.time() - start
        if log:
            self.logger.info("⏱️  %s: %.4f秒", name, elapsed)
        return elapsed

    def count(self, name: str, value: int = 1):
        self.counters[name] = self.counters.get(name, 0) + int(value)

    def get_counter(self, name: str) -> int:
        return int(self.counters.get(name, 0))

    def reset(self):
        self.timers.clear()
        self.counters.clear()

    def summary(self):
        self.logger.info("\n" + "="*70)
        self.logger.info("性能统计:")
        if self.counters:
            self.logger.info("计数器:")
            for k, v in self.counters.items():
                self.logger.info("  %s: %d", k, v)
        self.logger.info("="*70)

# -----------------------------------------------------------------------------
# 模块自检示例
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    print("日志系统（改进版）演示")
    print("=" * 70)

    # 全局 logger（按天轮转）
    logger = setup_logger(
        'demo', level=logging.DEBUG, use_color=True, utc=False,
        log_file='demo_logs/demo.log',
        rotate={'when':'D','interval':1,'backupCount':7},
        mp_safe=False
    )
    logger.debug("调试消息")
    logger.info("普通消息")
    logger.warning("警告消息")
    logger.error("错误消息")

    # 实验日志（上下文 + JSONL 指标）
    with ExperimentLogger('test_experiment', output_dir='demo_logs', level=logging.DEBUG,
                          use_color=True, rotate={'maxBytes':2_000_000,'backupCount':3}, mp_safe=False) as exp:
        exp.log_config({'simulation': {'L': 32, 'algorithm': 'metropolis'}, 'training': {'epochs': 5}})
        for epoch in range(1, 6):
            exp.log_epoch(epoch, train_loss=1.0/epoch, val_loss=1.2/epoch)
            exp.log_metric('accuracy', 0.8 + 0.04*epoch, step=epoch)
        exp.log_results({'final_loss': 0.15, 'accuracy': 0.96, 'best_epoch': 5})

    # 进度
    prog = ProgressLogger(100, desc="模拟进度", log_every_n=25, log_every_seconds=1.0)
    for _ in range(100):
        time.sleep(0.01)
        prog.update()
    prog.finish()

    # 性能
    pm = PerformanceMonitor(logger)
    pm.start_timer("heavy")
    time.sleep(0.1)
    pm.stop_timer("heavy")
    pm.count("iters", 1000)
    pm.summary()
    print("\n✓ 演示完成！")

