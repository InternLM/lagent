from lagent.rag.schema import Node, Relationship, CommunityContext, CommunityReport
from lagent.rag.doc import Storage
from typing import Dict, Optional, List
import hashlib
import igraph as ig
import re


def normalize_edge(edge: Relationship) -> Relationship:
    """
    标准化边的表示，使得边在无向情况下具有相同的键。

    :param: edge
    :return: 标准化后的边 (sorted_source_id, sorted_target_id)
    """
    source = edge.source
    target = edge.target
    if source < target:
        return edge
    else:
        edge.source = target
        edge.target = source
        return edge


def name_to_id(name: str) -> str:
    _id = hashlib.md5(name.encode('utf-8')).hexdigest()
    return _id


def tuple_to_str(data: tuple[str, str]) -> str:
    return f"{data[0]},{data[1]}"


def str_to_tuple(data: str) -> tuple[str, str]:
    items = data.split(",")
    if len(items) != 2:
        raise ValueError("edge doesn't have two nodes when transformation")

    else:
        source = items[0]
        target = items[1]
        return (source, target)


def get_id_map_nodes(nodes: List[Node], storage: Optional[Storage], stage: str) -> Dict[str, Node]:
    """

    Args:
        nodes:
        storage:
        stage:说明当前存储的映射是哪一阶段的nodes

    Returns:

    """
    load_dict_items = storage.get(f"id_map_nodes_{stage}")
    id_map_nodes = {}
    if load_dict_items is None:

        map_items_dict = []
        for node in nodes:
            id_map_nodes[node.id] = node
            map_items_dict.append({node.id: node.to_dict()})
        storage.put(f"id_map_nodes_{stage}", map_items_dict)

    else:
        for map_item in load_dict_items:
            # 由存储时可知，对于每个元素只含有一个键值对
            k, v = map_item.popitem()
            id_map_nodes[k] = Node.dict_to_node(v)

    return id_map_nodes


def get_id_map_relationships(relationships: List[Relationship], storage: Optional[Storage], stage: str) -> Dict[
    str, Relationship]:
    load_dict_items = storage.get(f"id_map_relationships_{stage}")
    id_map_relationships = {}

    if load_dict_items is None:
        map_items_dict = []
        for edge in relationships:
            name = tuple_to_str((edge.source, edge.target))
            id_map_relationships[name] = edge
            map_items_dict.append({name: edge.to_dict()})
        storage.put(f"id_map_relationships_{stage}", map_items_dict)

    else:
        try:
            for map_item in load_dict_items:
                k, v = map_item.popitem()
                id_map_relationships[k] = Relationship.dict_to_edge(v)
        except ValueError as e:
            print(f"Caught an exception: {e}")

    return id_map_relationships


def create_igraph(nodes: List[Node], edges: Optional[List[Relationship]] = None):
    # given nodes and edges, create a graph for leidenalg

    # 创建一个空的无向图
    g = ig.Graph(directed=False)

    for node in nodes:

        # TODO:哪些属性应该被加入，以及属性命名
        vertex_attr = {'id': node.id, 'name': node.id}
        if node.description is not None:
            vertex_attr['description'] = node.description
        if hasattr(node, 'degree'):
            vertex_attr['degree'] = node.degree
        if hasattr(node, 'community'):
            vertex_attr['community'] = node.community

        g.add_vertex(**vertex_attr)

    if edges is None:
        return g

    for edge in edges:
        edge_attr = edge.to_dict()
        edge_attr['source'] = edge_attr.get('source')
        edge_attr['target'] = edge_attr.get('target')

        g.add_edge(**edge_attr)

    return g


def filter_nodes_by_commu(nodes: List[Node], community_id: str) -> List[Node]:
    result = []
    for node in nodes:
        if not hasattr(node, 'community'):
            raise AttributeError(f'{node} should have attribute community')
        if node.community == community_id:
            result.append(node)

    return result


def filter_relas_by_nodes(nodes: List[Node], relationships: List[Relationship]) -> List[Relationship]:
    result = []
    map_nodes = {}

    for node in nodes:
        map_nodes[node.id] = node

    for rela in relationships:
        if rela.source in map_nodes and rela.target in map_nodes:
            result.append(rela)

    return result


def get_communities_context_reports_by_level(level: int, community_contexts: List[CommunityContext],
                                             community_reports: List[CommunityReport]):
    """
    筛选community_contexts中在level层次的contexts, reports处理同理
    Args:
        level:
        community_contexts:
        community_reports:

    Returns:

    """
    result_contexts = []
    for community_context in community_contexts:
        if community_context.level == level:
            result_contexts.append(community_context)

    result_reports = []
    for community_report in community_reports:
        if community_report.level == level:
            result_reports.append(community_report)

    return result_contexts, result_reports


def replace_variables_in_prompt(prompt: str, prompt_variables: Dict[str, str]):
    """
    替换prompt中的变量为实际值，支持多种变量格式并包含错误检查

    :param prompt: 包含变量的prompt字符串，支持以下格式:
                   - {variable}
                   - {{variable}}（双花括号）
                   - ${variable}（美元符号）
    :param prompt_variables: 包含变量名及其对应值的字典
    :return: 替换变量后的完整prompt字符串
    :raises ValueError: 如果变量未在字典中找到，抛出异常
    """

    # 定义正则表达式模式，用于检测变量格式
    patterns = {
        'braces': re.compile(r'\{(\w+)\}'),         # {variable}
        'double_braces': re.compile(r'\{\{(\w+)\}\}'),  # {{variable}}
        'dollar': re.compile(r'\$\{(\w+)\}')        # ${variable}
    }

    # 使用正则表达式替换变量
    def replace_variable(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name in prompt_variables:
            return prompt_variables[var_name]
        else:
            return match.group(0)  # 如果变量不在字典中，返回原始变量格式

    try:
        # 尝试替换 {variable}
        prompt = patterns['braces'].sub(replace_variable, prompt)

        # 尝试替换 {{variable}}
        prompt = patterns['double_braces'].sub(lambda m: replace_variable(m), prompt)

        # 尝试替换 ${variable}
        prompt = patterns['dollar'].sub(lambda m: replace_variable(m), prompt)

    except ValueError as e:
        # 捕捉并处理变量未找到的错误
        raise ValueError(f"Error in prompts replacement: {e}")

    return prompt
