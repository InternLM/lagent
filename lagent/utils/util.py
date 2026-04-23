import asyncio
import importlib
import inspect
import logging
import os
import os.path as osp
import re
import sys
import time
from functools import partial
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Generator, Iterable, List, Optional, Union, cast


def load_class_from_string(class_path: str, path=None):
    path_in_sys = False
    if path:
        if path not in sys.path:
            path_in_sys = True
            sys.path.insert(0, path)

    try:
        module_name, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls
    finally:
        if path and path_in_sys:
            sys.path.remove(path)


def create_object(config: Union[Dict, Any] = None):
    """Create an instance based on the configuration where 'type' is a
    preserved key to indicate the class (path). When accepting non-dictionary
    input, the function degenerates to an identity.
    """
    from ray.actor import ActorClass

    if config is None or not isinstance(config, dict):
        return config
    assert isinstance(config, dict) and 'type' in config

    config = config.copy()
    obj_type = config.pop('type')
    if isinstance(obj_type, str):
        obj_type = load_class_from_string(obj_type)
    if isinstance(obj_type, ActorClass):
        obj = cast(ActorClass, obj_type).remote(**config)
    elif inspect.isclass(obj_type):
        obj = obj_type(**config)
    else:
        assert callable(obj_type)
        obj = partial(obj_type, **config)
    return obj


async def async_as_completed(futures: Iterable[asyncio.Future]):
    """A asynchronous wrapper for `asyncio.as_completed`"""
    loop = asyncio.get_event_loop()
    wrappers = []
    for fut in futures:
        assert isinstance(fut, asyncio.Future)
        wrapper = loop.create_future()
        fut.add_done_callback(wrapper.set_result)
        wrappers.append(wrapper)
    for next_completed in asyncio.as_completed(wrappers):
        yield await next_completed


def filter_suffix(response: Union[str, List[str]], suffixes: Optional[List[str]] = None) -> str:
    """Filter response with suffixes.

    Args:
        response (Union[str, List[str]]): generated responses by LLMs.
        suffixes (str): a list of suffixes to be deleted.

    Return:
        str: a clean response.
    """
    if suffixes is None:
        return response
    batched = True
    if isinstance(response, str):
        response = [response]
        batched = False
    processed = []
    for resp in response:
        for item in suffixes:
            # if response.endswith(item):
            #     response = response[:len(response) - len(item)]
            if item in resp:
                resp = resp.split(item)[0]
        processed.append(resp)
    if not batched:
        return processed[0]
    return processed


def get_logger(
    name: str = 'lagent',
    level: str = 'debug',
    fmt: str = '%(asctime)s %(levelname)8s %(filename)20s %(lineno)4s - %(message)s',
    add_file_handler: bool = False,
    log_dir: str = 'log',
    log_file: str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()) + '.log',
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 3,
):
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))

    formatter = logging.Formatter(fmt)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if add_file_handler:
        if not osp.exists(log_dir):
            os.makedirs(log_dir)
        log_file_path = osp.join(log_dir, log_file)
        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def truncate_text(text, max_num=4000, side='middle'):
    """
    中英文混合场景下，根据 side 参数截断文本。总共保留 approx max_num 个“词/字”。

    定义“单位”逻辑：
    1. 连续的英文/数字被视为 1 个单位 (如 "Python", "123")
    2. 单个汉字或标点被视为 1 个单位 (如 "中", "。", ",")

    Args:
        text (str): 原始文本
        max_num (int): 截取的单位数量
        side (str): 截断模式，可选 'left', 'right', 'middle'
            'left': 保留尾部（截断头部）
            'right': 保留头部（截断尾部）
            'middle': 保留头尾（截断中间）
    """
    if not text or max_num <= 0:
        return ""

    # --- 核心正则 ---
    # 逻辑：匹配 (英文/数字/下划线/连字符 组成的词) 或 (非空白的单字符)
    # 注意：英文的正则必须放在前面，表示优先匹配完整单词
    pattern = re.compile(r"[a-zA-Z0-9_'-]+|[^\s]")

    # 获取所有匹配对象（包含位置信息）
    matches = list(pattern.finditer(text))
    total_units = len(matches)

    # 如果总数不够，返回全文
    if total_units <= max_num:
        return text

    parts = []

    if side == 'left':
        # 保留尾部 max_num 个单位（截断头部）
        # matches[-max_num] 是保留部分的第一个词
        start_idx = total_units - max_num
        start_pos = matches[start_idx].start()
        parts.append("(truncated)...")
        parts.append(text[start_pos:])

    elif side == 'right':
        # 保留头部 max_num 个单位（截断尾部）
        # matches[max_num - 1] 是保留部分的最后一个词
        end_pos = matches[max_num - 1].end()
        parts.append(text[:end_pos])
        parts.append("...(truncated)")

    else:  # middle
        # --- 智能截取 (保留头尾) ---
        head_count = max_num // 2
        tail_count = max_num - head_count

        # 1. 提取头部
        if head_count > 0:
            # matches[head_count - 1] 是头部想要保留的最后一个词
            head_span_end = matches[head_count - 1].end()
            parts.append(text[:head_span_end])

        # 2. 插入截断提示
        parts.append("...(truncated)...")

        # 3. 提取尾部
        if tail_count > 0:
            # matches[-tail_count] 是尾部想要保留的第一个词
            tail_idx = total_units - tail_count
            tail_span_start = matches[tail_idx].start()
            parts.append(text[tail_span_start:])

    return "".join(parts)


class GeneratorWithReturn:
    """Generator wrapper to capture the return value."""

    def __init__(self, generator: Generator):
        self.generator = generator
        self.ret = None

    def __iter__(self):
        self.ret = yield from self.generator
        return self.ret
