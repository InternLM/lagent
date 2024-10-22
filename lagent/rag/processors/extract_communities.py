import leidenalg as la
import igraph as ig
import networkx as nx
from graspologic.partition import hierarchical_leiden
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from lagent.rag.doc import Storage
from lagent.rag.schema import Node, Layer, Community, MultiLayerGraph
from lagent.rag.pipeline import register_processor, BaseProcessor
from lagent.rag.settings import DEFAULT_RESOLUTIONS, DEFAULT_NUM_CLUSTER
from lagent.utils import create_object
from node2vec import Node2Vec
from sklearn.cluster import AgglomerativeClustering
import copy
from typing import List, Dict, Optional
import numpy as np


def nx_to_igraph(graph_nx: nx.Graph) -> ig.Graph:
    """
        Converts a NetworkX graph to an iGraph graph, preserving all node and edge attributes.

        Args:
            graph_nx (nx.Graph): The NetworkX graph object to convert.

        Returns:
            ig.Graph: The converted iGraph graph object.
    """
    ig_nodes = [str(node) for node in graph_nx.nodes()]

    ig_edges = [tuple(sorted([str(source), str(target)])) for source, target in graph_nx.edges()]

    graph_ig = ig.Graph()
    graph_ig.add_vertices(ig_nodes)
    graph_ig.vs["id"] = ig_nodes

    graph_ig.add_edges(ig_edges)

    edge_mapping = {}
    for idx, edge in enumerate(graph_ig.es):
        source = graph_ig.vs[edge.source]["id"]
        target = graph_ig.vs[edge.target]["id"]
        edge_key = tuple(sorted([source, target]))
        edge_mapping[edge_key] = idx

    for node_id, attrs in graph_nx.nodes(data=True):
        ig_node = graph_ig.vs.find(id=str(node_id))
        for attr_key, attr_val in attrs.items():
            ig_node[attr_key] = attr_val

    for source, target, attrs in graph_nx.edges(data=True):
        source_str = str(source)
        target_str = str(target)
        edge_key = tuple(sorted([source_str, target_str]))
        if edge_key in edge_mapping:
            edge_idx = edge_mapping[edge_key]
            ig_edge = graph_ig.es[edge_idx]
            for attr_key, attr_val in attrs.items():
                ig_edge[attr_key] = attr_val
        else:
            print(f"Warning: Edge ({source}, {target}) not found in iGraph.")

    return graph_ig


def generate_node_embeddings(graph_nx: nx.Graph,
                             dimensions: int = 32,
                             walk_length: int = 10,
                             num_walks: int = 100,
                             workers: int = 4) -> Dict[str, np.ndarray]:
    """
        Generates node embeddings using Node2Vec.

        Args:
            graph_nx (nx.Graph): The NetworkX graph object.
            dimensions (int, optional): The dimensionality of the embeddings. Defaults to 32.
            walk_length (int, optional): The length of each random walk. Defaults to 10.
            num_walks (int, optional): The number of walks per node. Defaults to 100.
            workers (int, optional): The number of parallel workers. Defaults to 4.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping node IDs to their embedding vectors.
    """
    node2vec = Node2Vec(graph_nx, dimensions=dimensions, walk_length=walk_length,
                        num_walks=num_walks, workers=workers, quiet=True)
    paras = {
        "window": 10,
        "min_count": 1,
        "batch_words": 4
    }
    model = node2vec.fit(**paras)
    embeddings = {str(node): model.wv[str(node)] for node in graph_nx.nodes()}
    return embeddings


def get_community_hierarchy(communities: List[Community], levels: List[int]) -> Dict[int, Dict[str, List[str]]]:
    """
        Constructs a hierarchical mapping of communities to their sub-communities.

        Args:
            communities (List[Community]): A list of Community instances.
            levels (List[int]): A list of unique levels in the community hierarchy.

        Returns:
            Dict[int, Dict[str, List[str]]]: A dictionary mapping each level to its communities and their sub-community IDs.
    """
    communities_by_level: Dict[int, Dict[str, List[str]]] = {}
    community_hierarchy: Dict[int, Dict[str, List[str]]] = {}

    for level in levels:
        communities_by_level[level] = {}
        for community in communities:
            if community.level == level:
                communities_by_level[level][community.community_id] = community.nodes_id

    # Sort levels in ascending order (smaller level means higher hierarchy)
    levels = list(sorted(levels))

    for index in range(len(levels) - 1):
        current_level = levels[index]
        lower_level = levels[index + 1]
        current_communities = communities_by_level[current_level]
        lower_communities = communities_by_level[lower_level]
        community_hierarchy[current_level] = {}

        for _id, nodes_id in current_communities.items():
            find_nodes_len = 0
            community_hierarchy[current_level][_id] = []
            for lower_id, lower_nodes_id in lower_communities.items():
                if set(lower_nodes_id).issubset(set(nodes_id)):
                    community_hierarchy[current_level][_id].append(lower_id)
                    find_nodes_len += len(lower_nodes_id)
                if find_nodes_len == len(nodes_id):
                    break

    return community_hierarchy


def get_level_communities(hierarchy: Dict[int, Dict],
                          layer: Layer) -> Dict[int, Dict[int, List]]:
    """
        Get a dictionary mapping each level to its communities and their containing node IDs.

        Args:
            hierarchy (Dict[int, Dict[str, List[str]]]): The community hierarchy mapping.
            layer (Layer): The layer containing community information.

        Returns:
            Dict[int, Dict[int, List[str]]]: A dictionary mapping each level to its communities and their node IDs.
    """
    result: Dict[int, Dict[int, List]] = {}
    id_map_entities = {}
    entities = layer.get_nodes()
    for entity in entities:
        id_map_entities[entity['id']] = entity

    for level, node_map in hierarchy.items():
        result[level] = {}
        for node_id, community_id in node_map.items():
            if community_id not in result[level].keys():
                result[level][community_id] = []
            assert node_id in id_map_entities
            result[level][community_id].append(node_id)
        result[level] = dict(sorted(result[level].items()))

    return result


def get_level_Communities(hierarchy: Dict[int, Dict],
                          layer: Layer) -> Dict[int, List[Community]]:
    """
        Converts communities into Community instances organized by level.

        Args:
            hierarchy (Dict[int, Dict[str, List[str]]]): The community hierarchy mapping.
            layer (Layer): The layer containing community information.

        Returns:
            Dict[int, List[Community]]: A dictionary mapping each level to a list of Community instances.
    """
    result: Dict[int, List[Community]] = {}
    dict_result = get_level_communities(hierarchy, layer)
    for level, communities in dict_result.items():
        result[level] = []
        for community_id, ids in communities.items():
            nodes_id = [_id for _id in ids]

            _community = Community(
                community_id=f'{str(level)}_{str(community_id)}',
                level=level,
                nodes_id=nodes_id
            )
            result[level].append(_community)

    return result


@register_processor
class CommunitiesDetector(BaseProcessor):
    name = 'CommunitiesDetector'

    def __init__(self, detector: Optional[Dict] = None, storage: Storage = dict(type=Storage)):
        super().__init__(name='CommunitiesDetector')
        if detector is None:
            detector = {}
        self._detector = detector.pop('name', '').lower() or 'custom_leidenalg'
        self._detector_config = detector
        self._storage = create_object(storage)

    def run(
            self,
            graph: MultiLayerGraph,
            detector: Optional[Dict] = None,
            **kwargs) -> MultiLayerGraph:
        """
            Do hierarchical clustering on a given graph
            Args:
                graph (MultiLayerGraph): The multi-layer graph containing entities and relationships.
                detector (Optional[Dict[str, Any]], optional): Configuration for the community detection algorithm.
                    If provided, it overrides the initial detector configuration. Defaults to None.
                **kwargs (Any): Additional keyword arguments for community detection.

            Returns:
                MultiLayerGraph: The updated graph with detected communities added as a new layer.
        """

        if detector is None:
            _detector = self._detector
            _detector_config = self._detector_config
        else:
            _detector = detector.pop('name', '').lower() or self._detector
            _detector_config = detector

        entity_layer = graph.layers['summarized_entity_layer']
        dict_entities = entity_layer.get_nodes()
        id_map_entities = {}
        for dict_entity in dict_entities:
            id_map_entities[dict_entity['id']] = dict_entity

        community_layer = graph.add_layer('community_layer')

        detector_map = {
            'custom_leidenalg': self.custom_leiden_hierarchical,
            'leidenalg': self.leiden_hierarchical,
            'embedding_cluster': self.hierarchical_clustering_embeddings,
        }
        alg = detector_map[_detector]
        if alg is None:
            raise TypeError
        hierarchy = {}

        hierarchy = alg(entity_layer.graph, **_detector_config)

        communities_by_level = get_level_Communities(hierarchy, entity_layer)

        edges = copy.deepcopy(entity_layer.graph.edges(data=True))

        for level, communities in communities_by_level.items():

            level_layer = graph.add_layer(f'level{level}_entity_layer')

            for community in communities:
                community_id = community.community_id
                attr = {
                    'level': community.level,
                    'nodes_id': community.nodes_id
                }
                community_layer.add_node(community_id, **attr)
                for node_id in community.nodes_id:
                    node = copy.deepcopy(id_map_entities[node_id])
                    _ = node.pop('id')
                    node['community'] = community_id
                    node['level'] = level
                    level_layer.add_node(node_id, **node)

                for edge in edges:
                    edge[2]['level'] = level
                    level_layer.add_edge(edge[0], edge[1], **(edge[2]))

        return graph

    def hierarchical_clustering_embeddings(self, graph: nx.Graph,
                                           n_clusters: Optional[List[int]] = None,
                                           **kwargs):
        """
            Performs hierarchical clustering based on node embeddings.

            Args:
                graph (nx.Graph): The NetworkX graph object.
                n_clusters (Optional[List[int]], optional): A list of cluster counts for each hierarchical level.
                    Defaults to [2, 3, 4].
                **kwargs (Any): Additional keyword arguments for clustering.

            Returns:
                Dict[int, Dict[str, int]]: A dictionary mapping each level to a mapping of node IDs to cluster labels.
        """
        if n_clusters is None:
            n_clusters = [2, 3, 4]
        embeddings = generate_node_embeddings(graph, **kwargs)

        hierarchy = {}
        node_ids = list(embeddings.keys())
        emb_matrix = np.array([embeddings[node] for node in node_ids])

        for i, k in enumerate(n_clusters):
            clustering = AgglomerativeClustering(n_clusters=k)
            labels = clustering.fit_predict(emb_matrix)
            hierarchy[i] = {node: label for node, label in zip(node_ids, labels)}

        return hierarchy

    def leiden_hierarchical(
            self,
            graph_nx: nx.Graph,
            max_cluster_size: Optional[int] = None,
            **kwargs) -> Dict[int, Dict[str, int]]:
        """
            Performs hierarchical community detection using the Leiden algorithm on a NetworkX graph.

            Args:
                graph_nx (nx.Graph): The NetworkX graph object.
                max_cluster_size (Optional[int], optional): The maximum size of any cluster. Defaults to 10.
                **kwargs (Any): Additional keyword arguments for the Leiden algorithm.

            Returns:
                Dict[int, Dict[str, int]]: A dictionary mapping each level to a mapping of node IDs to cluster labels.
        """
        if max_cluster_size is None:
            max_cluster_size = 10

        community_mapping = hierarchical_leiden(
            graph_nx, max_cluster_size=max_cluster_size, random_seed=0xDEADBEEF
        )
        results: dict[int, dict[str, int]] = {}
        for partition in community_mapping:
            results[partition.level] = results.get(partition.level, {})
            results[partition.level][partition.node] = partition.cluster

        return results

    def custom_leiden_hierarchical(
            self,
            graph: nx.Graph,
            resolutions: Optional[List] = DEFAULT_RESOLUTIONS,
            **kwargs) -> Dict[int, Dict[int, int]]:
        """
            Performs hierarchical community detection using a custom Leiden algorithm implementation.

            Args:
                graph (nx.Graph): The NetworkX graph object.
                resolutions (Optional[List[float]], optional): A list of resolution parameters for the Leiden algorithm.
                    Defaults to DEFAULT_RESOLUTIONS.
                **kwargs (Any): Additional keyword arguments for the Leiden algorithm.

            Returns:
                Dict[int, Dict[str, int]]: A dictionary mapping each level to a mapping of node IDs to cluster labels.
        """

        # TODO: get resolutions(strategy?)

        if resolutions is None:
            resolutions = DEFAULT_RESOLUTIONS

        ig_graph = nx_to_igraph(graph)

        hierarchy = {}
        for i, resolution in enumerate(resolutions):
            partition = la.find_partition(
                ig_graph,
                la.CPMVertexPartition,
                resolution_parameter=resolution
            )
            node_ids = ig_graph.vs["id"]
            hierarchy[i] = {node_id: cluster for node_id, cluster in zip(node_ids, partition.membership)}

        return hierarchy
