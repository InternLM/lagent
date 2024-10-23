import copy
from typing import Optional
import os

from lagent.rag.schema import MultiLayerGraph
from lagent.rag.pipeline import BaseProcessor, register_processor
from lagent.rag.nlp import FaissDatabase
from lagent.rag.doc import Storage
from lagent.utils import create_object


@register_processor
class LoadGraph(BaseProcessor):
    name = 'LoadGraph'
    expected_input_type = str
    expected_output_type = MultiLayerGraph

    def __init__(self, storage: Storage = dict(type=Storage)):
        super().__init__(name='LoadGraph')

        self.storage = create_object(storage)

    def run(self, path: str) -> MultiLayerGraph:
        if not os.path.exists(path):
            raise ValueError(f"Path{path} doesn't exist")

        graph = self.storage.get('external_memory')
        graph = MultiLayerGraph.dict_to_multilayergraph(graph)

        for k, path in graph.layers_db.items():
            db = FaissDatabase.load(path)
            graph.layers_db[k] = db

        return graph


@register_processor
class SaveGraph(BaseProcessor):
    name = 'SaveGraph'
    expected_input_type = MultiLayerGraph
    expected_output_type = MultiLayerGraph

    def __init__(self, dir_name: Optional = None, storage: Storage = dict(type=Storage)):
        super().__init__(name='SaveGraph')
        self.storage = create_object(storage)
        self.dir_name = dir_name or self.storage.cache_dir

    def run(self, data: MultiLayerGraph) -> MultiLayerGraph:
        data_copy = copy.deepcopy(data)
        layers_db = data.layers_db
        for k, db in layers_db.items():
            path = f'{self.dir_name}\\{k}.pkl'
            path = db.save_local(file_path=path)
            layers_db[k] = path

        graph_dict = data.to_dict()
        self.storage.put('external_memory', graph_dict)

        return data_copy
