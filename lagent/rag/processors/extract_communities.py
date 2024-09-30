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
from node2vec import Node2Vec
from sklearn.cluster import AgglomerativeClustering
import copy
from typing import List, Dict, Optional
import numpy as np


def nx_to_igraph(graph_nx: nx.Graph) -> ig.Graph:
    """
    将 NetworkX 图转换为 iGraph 图，保留所有节点和边的属性。

    :param graph_nx: NetworkX 图对象
    :return: iGraph 图对象
    """
    # 将所有节点转换为字符串
    ig_nodes = [str(node) for node in graph_nx.nodes()]

    # 将所有边转换为字符串，并确保无向图中边的顺序一致
    ig_edges = [tuple(sorted([str(source), str(target)])) for source, target in graph_nx.edges()]

    # 创建一个空的 iGraph 图，并添加所有节点
    graph_ig = ig.Graph()
    graph_ig.add_vertices(ig_nodes)
    graph_ig.vs["id"] = ig_nodes  # 设置节点的 'id' 属性

    # 添加所有边到 iGraph 图中
    graph_ig.add_edges(ig_edges)

    # 创建边的映射字典，键为 (source, target)，值为边的索引
    edge_mapping = {}
    for idx, edge in enumerate(graph_ig.es):
        source = graph_ig.vs[edge.source]["id"]
        target = graph_ig.vs[edge.target]["id"]
        edge_key = tuple(sorted([source, target]))
        edge_mapping[edge_key] = idx

    # 复制节点属性
    for node_id, attrs in graph_nx.nodes(data=True):
        ig_node = graph_ig.vs.find(id=str(node_id))
        for attr_key, attr_val in attrs.items():
            ig_node[attr_key] = attr_val

    # 复制边属性
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
            # 如果边未找到，可以选择忽略或处理
            print(f"警告: 边 ({source}, {target}) 未在 iGraph 中找到。")

    return graph_ig


def generate_node_embeddings(graph_nx: nx.Graph,
                             dimensions: int = 32,  # 减少维度以适应描述较短的情况
                             walk_length: int = 10,  # 减少随机游走长度
                             num_walks: int = 100,  # 减少随机游走次数
                             workers: int = 4) -> Dict[str, np.ndarray]:
    """
    使用 Node2Vec 生成节点嵌入

    :param graph_nx: NetworkX 图对象
    :param dimensions: 嵌入维度
    :param walk_length: 随机游走长度
    :param num_walks: 每个节点的随机游走次数
    :param workers: 并行执行的工作线程数
    :return: 节点嵌入字典 {node_id: embedding}
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
    根据输入的communities，得到每个community的sub community
    """
    communities_by_level: Dict[int, Dict[str, List[str]]] = {}
    community_hierarchy: Dict[int, Dict[str, List[str]]] = {}

    # 首先获得每个level中不同community包含的节点
    for level in levels:
        communities_by_level[level] = {}
        for community in communities:
            if community.level == level:
                communities_by_level[level][community.community_id] = community.nodes_id

    # 默认level更小的为更高层次
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
    以字典的形式返回不同层次划分得到的community
    Args:
        hierarchy:
        layer:

    Returns:
        {level: {community_id: [nodes_id]}}
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
    将community以类的形式返回
    Args:
        layer:
        hierarchy:

    Returns:
        Dict[int, List[Community]]
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

    def __init__(self, detector: Optional[Dict] = None, storage: Optional[Storage] | Optional[Dict] = None):
        super().__init__(name='CommunitiesDetector')
        if detector is None:
            detector = {}
        self._detector = detector.pop('name', '').lower() or 'custom_leidenalg'
        self._detector_config = detector
        if isinstance(storage, Storage):
            self._storage = storage
        elif isinstance(storage, dict):
            self._storage = Storage(**storage)
        elif storage is None:
            self._storage = Storage()

    def run(
            self,
            graph: MultiLayerGraph,
            detector: Optional[Dict] = None,
            **kwargs) -> MultiLayerGraph:
        """
        根据给出实体做分层次聚类
        Args:
            graph:
            detector:
            **kwargs:

        Returns:

        """
        # 根据给出的detector，由对应方法得到不同层次社区并规范化输出

        if detector is None:
            _detector = self._detector
            _detector_config = self._detector_config
        else:
            _detector = detector.pop('name', '').lower() or self._detector
            _detector_config = detector

        # 目前只考虑节点为entity的聚类
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

        # save communities
        storage = self._storage
        list_dict_to_save = []
        for k, v in communities_by_level.items():
            list_dict_to_save.append({k: [_commu.to_dict() for _commu in v]})
        storage.put("level_communities_class", list_dict_to_save)

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

                # 添加边
                for edge in edges:
                    edge[2]['level'] = level
                    level_layer.add_edge(edge[0], edge[1], **(edge[2]))

        return graph

    def hierarchical_clustering_embeddings(self, graph: nx.Graph,
                                           n_clusters: Optional[List[int]] = None,
                                           **kwargs):
        """
        基于节点嵌入进行层次聚类

        :param graph:
        :param n_clusters: 聚类数列表
        :return: dict[level: {node_id: cluster}]
        """
        if n_clusters is None:
            n_clusters = [2, 3, 4]  # 可以根据需要调整
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
        利用 Leiden 算法在 networkx 图上手动进行分层次社区检测

        :param graph_nx: 必须是 networkx 的图对象
        :param resolutions: 分辨率参数列表，不同的分辨率对应不同的层次
        :return: dict[level: {node_index: cluster}] 层次化社区划分结果
        """
        if max_cluster_size is None:
            max_cluster_size = 10

        # 使用 graspologic 提供的 hierarchical_leiden 方法进行社区检测

        community_mapping = hierarchical_leiden(
            graph_nx, max_cluster_size=max_cluster_size, random_seed=0xDEADBEEF
        )
        results: dict[int, dict[str, int]] = {}
        for partition in community_mapping:
            results[partition.level] = results.get(partition.level, {})
            results[partition.level][partition.node] = partition.cluster

        # hierarchy = {}
        # for i, resolution in enumerate(resolutions):
        #     _, levels = hierarchical_leiden(graph_nx, resolution=resolution, **kwargs)
        #
        #     # 将结果以 {level: {node_index: cluster}} 的形式存储
        #     hierarchy[i] = {node: cluster for node, cluster in enumerate(levels[0])}

        return results

    def custom_leiden_hierarchical(
            self,
            graph: nx.Graph,
            resolutions: Optional[List] = DEFAULT_RESOLUTIONS,
            **kwargs) -> Dict[int, Dict[int, int]]:
        """
        利用leiden算法手动进行分层次社区检测

        :param graph:
        :param resolutions:
        :return: dict[level: {node_index: cluster}]
        """

        # TODO: get resolutions(strategy?动态调整？)
        # if resolutions is None:
        #     resolutions = DEFAULT_RESOLUTIONS
        #
        # ig_graph = nx_to_igraph(graph)
        #
        # hierarchy = {}
        # for i, resolution in enumerate(resolutions):
        #     partition = la.find_partition(
        #         ig_graph, la.CPMVertexPartition, resolution_parameter=resolution
        #     )
        #     hierarchy[i] = {node: cluster for node, cluster in enumerate(partition.membership)}
        #
        # # TODO: normalize output
        #
        # return hierarchy
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

    def cluster_nodes(self, detector_name: str, detector_config: Optional[Dict], graph):
        detector_map = {
            'leidenalg': self.custom_leiden_hierarchical
        }
        alg = detector_map[detector_name]
        if alg is None:
            raise TypeError
        hierarchy = {}
        hierarchy = alg(graph, **detector_config)

        return hierarchy

    def cluster_nodes_without_relationship(self, detector_name: str, detector_config: Optional[Dict],
                                           chunk_nodes: List[Node]):

        detector_without_edge_map = {
            'kmeans': self.kmeans_cluster
        }

        alg = detector_without_edge_map.get(detector_name)
        if alg is None:
            raise TypeError

        hierarchy = {}
        hierarchy = alg(chunk_nodes, **detector_config)

        return hierarchy

    def kmeans_cluster(self, nodes: List[Node], num_cluster: Optional[List] = None, **kwargs):
        """
        针对不含边的chunk_node
        Args:
            nodes:
            num_cluster:不同分辨率下的聚类数量列表
            **kwargs:

        Returns:

        """
        # TODO:考虑根据nodes数量以及实际应用情况手动调整num_cluster?
        if num_cluster is None:
            num_cluster = DEFAULT_NUM_CLUSTER
        # 提取内容
        contents = [node.content for node in nodes]
        ids = [node.id for node in nodes]

        # 计算TF-IDF矩阵
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(contents)

        # 初始化层次结果字典
        hierarchy = {}

        # 进行多层次聚类
        for i, n_clusters in enumerate(num_cluster):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(tfidf_matrix)
            clusters = kmeans.labels_

            # 将结果保存为 {node_index: cluster}
            hierarchy[i] = {index: cluster for index, cluster in enumerate(clusters)}

        return hierarchy
