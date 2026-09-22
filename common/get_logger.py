# -*- coding: utf-8 -*-
"""统一日志: 带时间戳/级别/进程名的格式, 多进程下每个进程独立 handler。
级别优先级: 环境变量 RGBD2OCC_LOGLEVEL > config.json 的 log_level > INFO。"""
import json
import logging
import os

_FMT = "%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s"



def attach_file(logger, path):
    """给 logger 增加文件 handler (主进程调用一次, 跑批日志留档可回溯)。"""
    import logging as _l, os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
    fh = _l.FileHandler(path, encoding="utf-8")
    fh.setFormatter(_l.Formatter(_FMT, datefmt="%H:%M:%S"))
    logger.addHandler(fh)
    return path


def _level():
    lv = os.environ.get("RGBD2OCC_LOGLEVEL")
    if not lv:
        try:
            cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "config.json")
            lv = json.load(open(cfg, encoding="utf-8")).get("log_level")
        except Exception:
            lv = None
    return getattr(logging, (lv or "INFO").upper(), logging.INFO)


def get_logger(name="rgbd2occ"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(_level())
        logger.propagate = False
    return logger
