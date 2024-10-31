import asyncio
import importlib
import inspect
import logging
import os
import os.path as osp
import sys
import time
from functools import partial
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Generator, Iterable, List, Optional, Union


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
    if config is None or not isinstance(config, dict):
        return config
    assert isinstance(config, dict) and 'type' in config

    config = config.copy()
    obj_type = config.pop('type')
    if isinstance(obj_type, str):
        obj_type = load_class_from_string(obj_type)
    if inspect.isclass(obj_type):
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


def filter_suffix(response: Union[str, List[str]],
                  suffixes: Optional[List[str]] = None) -> str:
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
    fmt:
    str = '%(asctime)s %(levelname)8s %(filename)20s %(lineno)4s - %(message)s',
    add_file_handler: bool = False,
    log_dir: str = 'log',
    log_file: str = time.strftime('%Y-%m-%d.log', time.localtime()),
    max_bytes: int = 5 * 1024 * 1024,
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
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class GeneratorWithReturn:
    """Generator wrapper to capture the return value."""

    def __init__(self, generator: Generator):
        self.generator = generator
        self.ret = None

    def __iter__(self):
        self.ret = yield from self.generator
        return self.ret
