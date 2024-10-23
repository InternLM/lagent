import os
import json
from typing import Optional, List
import logging
from lagent.rag.settings import DEFAULT_CACHE_DIR


class Storage:
    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = DEFAULT_CACHE_DIR
        self.cache_dir = os.path.abspath(cache_dir)

        if not os.path.exists(self.cache_dir):
            try:
                os.makedirs(self.cache_dir)
                logging.info(f"Cache directory created at {self.cache_dir}")
            except OSError as e:
                logging.error(f"Failed to create cache directory: {e}")
                raise

    def _get_cache_path(self, cache_name: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_name}.json")

    def put(self, cache_name: str, data):
        cache_path = self._get_cache_path(cache_name)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f"Cache written successfully to {cache_path}")
            return cache_path
        except IOError as e:
            logging.error(f"Failed to write cache file {cache_path}: {e}")
            raise

    def get(self, cache_name: str):
        cache_path = self._get_cache_path(cache_name)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    logging.info(f"Cache loaded successfully from {cache_path}")
                    return data
            except IOError as e:
                logging.error(f"Failed to read cache file {cache_path}: {e}")
                raise
            except json.JSONDecodeError as e:
                logging.error(f"Failed to decode JSON from {cache_path}: {e}")
                raise
        return None

    def exist(self, cache_name: str) -> bool:
        cache_path = self._get_cache_path(cache_name)
        if os.path.exists(cache_path):
            print(f"path: {cache_path} exists")
            return True
        else:
            print(f"path: {cache_path} doesn't exists")
            return False
