import copy
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import networkx as nx
import matplotlib.pyplot as plt


class Layer:
    """
       Represents a single layer within a multi-layer graph, encapsulating its own NetworkX graph.

       Attributes:
           layer_id (str): The unique identifier for the layer.
           graph (nx.Graph): The NetworkX graph representing the layer's nodes and edges.
    """
    def __init__(self, layer_id: str):
        self.layer_id = layer_id
        self.graph = nx.Graph()

    def add_node(self, node_id, **attributes):
        self.graph.add_node(node_id, **attributes)

    def add_edge(self, node_id_1, node_id_2, **attributes):
        self.graph.add_edge(node_id_1, node_id_2, **attributes)

    def get_node_attributes(self, attribute_name):
        return nx.get_node_attributes(self.graph, attribute_name)

    def get_nodes(self) -> List[Dict]:
        nodes = []
        _nodes = self.graph.nodes(data=True)
        for node in _nodes:
            node_id = node[0]
            node_attr_dict = copy.deepcopy(node[1])
            node_attr_dict['id'] = node_id
            nodes.append(node_attr_dict)

        return nodes

    def get_edges(self) -> List[Dict]:
        edges = []
        _edges = self.graph.edges(data=True)
        for edge in _edges:
            source_id = edge[0]
            target_id = edge[1]
            attr = copy.deepcopy(edge[2])
            attr['source'] = source_id
            attr['target'] = target_id
            edges.append(attr)

        return edges


class MultiLayerGraph:
    """
        Represents a multi-layer graph consisting of multiple Layer instances and inter-layer edges.
        This class facilitates the storage and organization of data produced by each processor in the pipeline by
        representing them as graphs within multiple layers.
        Ultimately, it aggregates these layered graphs into a comprehensive MultiLayerGraph structure, which serves
        as the external memory for the RAG system.

        Attributes:
            layers (Dict[str, Layer]): A dictionary mapping layer IDs to their corresponding Layer instances.
            interlayer_edges_by_layers (nx.DiGraph): A directed graph representing edges between different layers.
            layers_db (Dict[str, str]): A dictionary for storing vector databases associated with layers.
    """
    def __init__(self):
        # Each layer is an independent graph, stored as a dictionary {layer_id: Layer object}}
        self.layers = {}
        # Stores inter-layer edge mappings as a directed graph { (layer_id_1, layer_id_2): Graph }
        self.interlayer_edges_by_layers = nx.DiGraph()
        # Store vector database for some layers
        self.layers_db = {}

    def add_layer(self, layer_id):
        if layer_id not in self.layers:
            new_layer = Layer(layer_id)
            self.layers[layer_id] = new_layer
        return self.layers[layer_id]

    def add_node(self, layer_id, node_id, **attributes):
        if layer_id not in self.layers:
            self.add_layer(layer_id)
        self.layers[layer_id].add_node(node_id, **attributes)

    def add_edge(self, layer_id, node_id_1, node_id_2, **attributes):
        if layer_id in self.layers:
            self.layers[layer_id].add_edge(node_id_1, node_id_2, **attributes)

    def add_interlayer_edge(self, layer_id_1, source_node_id, layer_id_2, target_node_id, **attributes):
        if source_node_id not in self.layers[layer_id_1].graph.nodes:
            raise ValueError(f"node{source_node_id} doesn't exist in layer{layer_id_1}")
        if target_node_id not in self.layers[layer_id_2].graph.nodes:
            raise ValueError(f"node{target_node_id} doesn't exist in layer{layer_id_2}")

        self.interlayer_edges_by_layers.add_edge((layer_id_1, source_node_id), (layer_id_2, target_node_id),
                                                 **attributes)

    def get_interlayer_mappings(self, layer_id_1, layer_id_2):
        mappings = {}
        for (source, target) in self.interlayer_edges_by_layers.edges():
            if source[0] == layer_id_1 and target[0] == layer_id_2:
                if source[1] not in mappings:
                    mappings[source[1]] = []
                mappings[source[1]].append(target[1])
        return mappings

    def to_dict(self) -> Dict:
        graph_dict = {}
        for layer_id, layer in self.layers.items():
            graph_dict[layer_id] = {
                'nodes': layer.get_nodes(),
                'edges': layer.get_edges()
            }

        # Add interlayer mappings
        interlayer_mappings = {}
        for (source, target) in self.interlayer_edges_by_layers.edges():
            source_layer, source_node = source
            target_layer, target_node = target

            # Define a key for each layer pair (e.g., "Layer1 -> Layer2")
            layer_pair_key = f"{source_layer} -> {target_layer}"

            if layer_pair_key not in interlayer_mappings:
                interlayer_mappings[layer_pair_key] = {}

            if source_node not in interlayer_mappings[layer_pair_key]:
                interlayer_mappings[layer_pair_key][source_node] = []

            interlayer_mappings[layer_pair_key][source_node].append(target_node)

        # Only add interlayer_mappings if there are any
        if interlayer_mappings:
            graph_dict['interlayer_mappings'] = interlayer_mappings

        # 存储db
        if isinstance(list(self.layers_db.values())[0], str):
            graph_dict['layers_db'] = self.layers_db

        return graph_dict

    @classmethod
    def dict_to_multilayergraph(cls, graph_dict: Dict) -> 'MultiLayerGraph':
        # Initialize a new MultiLayerGraph object
        ml_graph = cls()

        # Iterate through each layer in the dictionary
        for layer_id, layer_data in graph_dict.items():
            if layer_id == 'interlayer_mappings':
                continue  # Skip interlayer mappings for now

            # Add the layer to the MultiLayerGraph
            layer = ml_graph.add_layer(layer_id)

            # Add nodes to the layer
            for node in layer_data.get('nodes', []):
                node_id = node.pop('id')  # Extract the node ID
                layer.add_node(node_id, **node)  # Add node with remaining attributes

            # Add edges to the layer
            for edge in layer_data.get('edges', []):
                source = edge.pop('source')  # Extract source node ID
                target = edge.pop('target')  # Extract target node ID
                layer.add_edge(source, target, **edge)  # Add edge with remaining attributes

        # Handle interlayer mappings if present
        interlayer_mappings = graph_dict.get('interlayer_mappings', {})
        for layer_pair, mappings in interlayer_mappings.items():
            # Split the layer pair key to get source and target layer IDs
            try:
                source_layer, target_layer = layer_pair.split(' -> ')
            except ValueError:
                raise ValueError(f"Invalid layer pair format: '{layer_pair}'. Expected format 'Layer1 -> Layer2'.")

            # Iterate through each source node and its corresponding target nodes
            for source_node, target_nodes in mappings.items():
                for target_node in target_nodes:
                    # Add interlayer edges
                    ml_graph.add_interlayer_edge(source_layer, source_node, target_layer, target_node)

        layers_db = graph_dict.get('layers_db', {})
        ml_graph.layers_db = layers_db

        return ml_graph

    def visualize_layers(self, layer_ids):
        plt.figure(figsize=(12, 12))
        pos = {}
        offset = 0

        labels = {}

        for layer_id in layer_ids:
            if layer_id in self.layers:
                layer_graph = self.layers[layer_id].graph
                layer_pos = nx.spring_layout(layer_graph, seed=42)
                layer_pos = {k: (v[0], v[1] + offset) for k, v in layer_pos.items()}
                pos.update(layer_pos)

                labels.update({node: node for node in layer_graph.nodes()})

                offset += 3

                nx.draw(layer_graph, pos,
                        nodelist=layer_graph.nodes(),
                        with_labels=False,
                        node_color=f"C{layer_id}",
                        node_size=500,
                        edge_color=f"C{layer_id}",
                        alpha=0.6)

                nx.draw_networkx_edges(layer_graph, pos,
                                       edgelist=layer_graph.edges(),
                                       edge_color=f"C{layer_id}",
                                       alpha=0.6)

        nx.draw_networkx_labels(self.layers[layer_ids[0]].graph, pos=pos, labels=labels, font_size=10)

        if self.interlayer_edges_by_layers is not None:
            interlayer_edges = self.interlayer_edges_by_layers.edges()
            valid_edges = [(source[1], target[1]) for source, target in interlayer_edges if
                           source[1] in pos and target[1] in pos]
            if valid_edges:
                nx.draw_networkx_edges(self.interlayer_edges_by_layers, pos, edgelist=valid_edges,
                                       edge_color="black", style="dotted", width=2, arrows=True)

        plt.title(f"Visualization of Layers: {layer_ids}")
        plt.axis('off')
        plt.show()


class Node(BaseModel):
    embedding: Optional[List[float]] = None
    content: Optional[str] = None
    id: str
    source_id: Optional[List[str]] = None
    description: Optional[str] = None
    entities: Optional[List[str]] = None
    entity_type: Optional[str] = None

    _extra_attributes: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__annotations__:
            super().__setattr__(name, value)
        else:
            self._extra_attributes[name] = value

    def __getattr__(self, name: str) -> Any:
        if name in self.__annotations__:
            return super().__getattr__(name)
        if name in self._extra_attributes:
            return self._extra_attributes[name]
        raise AttributeError(f"'Node' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        if name in self.__annotations__:
            super().__delattr__(name)
        elif name in self._extra_attributes:
            del self._extra_attributes[name]
        else:
            raise AttributeError(f"'Node' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        return {**{k: v for k, v in self.__dict__.items() if v is not None},
                **self._extra_attributes}

    @classmethod
    def dict_to_node(cls, data: Dict) -> 'Node':
        embedding = data.pop('embedding', None)
        entities = data.pop('entities', None)
        entity_type = data.pop('entity_type', None)
        source = data.pop('source', None)
        description = data.pop('description', None)
        _id = data.pop('id')
        content = data.pop('content', None)
        return cls(
            content=content,
            embedding=embedding,
            id=_id,
            source=source,
            description=description,
            entities=entities,
            entity_type=entity_type,
            **data
        )


class Relationship(BaseModel):
    weight: Optional[float] = None
    source: str
    target: str
    description: Optional[str] = None

    _extra_attributes: Dict[str, Any] = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__annotations__:
            super().__setattr__(name, value)
        else:
            self._extra_attributes[name] = value

    def __getattr__(self, name: str) -> Any:
        if name in self.__annotations__:
            return super().__getattr__(name)
        if name in self._extra_attributes:
            return self._extra_attributes[name]
        raise AttributeError(f"'Node' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        if name in self.__annotations__:
            super().__delattr__(name)
        elif name in self._extra_attributes:
            del self._extra_attributes[name]
        else:
            raise AttributeError(f"'Node' object has no attribute '{name}'")

    def to_dict(self) -> Dict[str, Any]:
        return {**{k: v for k, v in self.__dict__.items() if v is not None},
                **self._extra_attributes}

    @classmethod
    def dict_to_edge(cls, data: Dict) -> 'Relationship':
        weight = data.pop("weight")
        description = data.pop("description")
        source = data.pop('source')
        target = data.pop('target')
        if data:
            return cls(
                weight=weight,
                source=source,
                target=target,
                description=description,
                **data
            )
        else:
            return cls(
                weight=weight,
                source=source,
                target=target,
                description=description,
            )


class Community(BaseModel):
    community_id: str
    level: int
    nodes_id: List[str]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            if value is not None:
                setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class CommunityContext(BaseModel):
    community_id: str
    level: int
    nodes_info: Dict[str, Dict]
    edges_info: Dict[tuple[str, str], Dict]
    claims: Optional[Dict[str, Any]] = None
    context_str: str = ''
    context_size: int = 0
    exceed_token: bool = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class CommunityReport(BaseModel):
    community_id: str
    level: int
    report: str
    structured_report: Dict

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


class Chunk(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]
    token_num: int

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def dict_to_chunk(cls, data: Dict) -> 'Chunk':
        return cls(
            id=data['id'],
            content=data['content'],
            metadata=data['metadata'],
            token_num=data['token_num']
        )


class Document(BaseModel):
    content: List[Dict[str, Any]]
    id: str
    metadata: Dict[str, Any]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def dict_to_document(cls, data: Dict) -> 'Document':
        return cls(
            id=data['id'],
            content=data['content'],
            metadata=data['metadata']
        )
