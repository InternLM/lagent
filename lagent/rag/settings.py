import os

DEFAULT_CHUNK_SZIE: int = 1000
DEFAULT_OVERLAP: int = 200
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(BASE_DIR, 'cache')

CONNECT_THRES = 0.3

DEFAULT_EDGE_STRATEGY = {
    'strategy': 'connect_all'
}

DEFAULT_LLM_MAX_TOKEN = 4096

DEFAULT_RESOLUTIONS = [0.1, 0.3, 0.5, 0.7]
DEFAULT_NUM_CLUSTER = [2, 5, 7]
