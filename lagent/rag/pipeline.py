from lagent.rag.schema import MultiLayerGraph

from typing import Dict, Type, Any, List
from abc import abstractmethod
import yaml


class BaseProcessor:
    name: str
    expected_input_type = MultiLayerGraph
    expected_output_type = MultiLayerGraph

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, data: MultiLayerGraph) -> MultiLayerGraph:
        raise NotImplementedError("Must implement run method.")


processors: Dict[str, Type['BaseProcessor']] = {}


def register_processor(processor_class):
    if not hasattr(processor_class, 'run'):
        raise ValueError(f"{processor_class.__name__} must implement a 'run' method.")

    if not callable(getattr(processor_class, 'run')):
        raise ValueError(f"'run' method in {processor_class.__name__} must be callable.")

    input_type = getattr(processor_class, 'expected_input_type', MultiLayerGraph)
    output_type = getattr(processor_class, 'expected_output_type', MultiLayerGraph)

    original_run = processor_class.run

    def new_run(self, data) -> Any:
        if input_type and not isinstance(data, input_type):
            raise TypeError(f"{self.name} expects input type '{input_type.__name__}', got '{type(data).__name__}'")
        result = original_run(self, data)
        if output_type and not isinstance(result, output_type):
            raise TypeError(f"{self.name} expects output type '{output_type.__name__}', got '{type(result).__name__}'")
        return result

    processor_class.run = new_run
    processors[processor_class.name] = processor_class
    return processor_class


class Pipeline:
    def __init__(self):
        self.processors = []

    def add_processor(self, processor: BaseProcessor):

        self.processors.append(processor)

    def run(self, initial_data: MultiLayerGraph) -> MultiLayerGraph:
        data = initial_data
        for processor in self.processors:
            data = processor.run(data)
        return data

    @classmethod
    def load_processors_from_config(cls, config_path: str, dependencies: Dict[str, Any]) -> List[BaseProcessor]:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)

        processors_list = []
        for proc_conf in config.get('processors', []):
            class_name = proc_conf['class']
            params = proc_conf.get('params', {})

            for key, value in params.items():
                if isinstance(value, str) and value in dependencies:
                    params[key] = dependencies[value]

            processor_class = processors.get(class_name)
            if not processor_class:
                raise ValueError(f"processor:'{class_name}' is not found")

            processor_instance = processor_class(**params)
            processors_list.append(processor_instance)

        return processors_list
