from lagent.rag.schema import Node, Relationship, CommunityContext, CommunityReport
from lagent.rag.doc import Storage
from typing import Dict, Optional, List
import hashlib
import igraph as ig
import re


def normalize_edge(edge: Relationship) -> Relationship:
    """
        Normalizes the representation of an edge to ensure consistent ordering in undirected graphs.

        Args:
            edge (Relationship): The relationship to normalize.

        Returns:
            Relationship: The normalized relationship with source and target IDs sorted.
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
    g = ig.Graph(directed=False)

    for node in nodes:

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
        Replaces variables in the prompt with actual values, supporting multiple variable formats with error checking.

        Args:
            prompt (str): The prompt string containing variables in formats like {variable}, {{variable}}, or ${variable}.
            prompt_variables (Dict[str, str]): A dictionary containing variable names and their corresponding values.

        Returns:
            str: The complete prompt string with variables replaced by their actual values.

        Raises:
            ValueError: If a variable in the prompt is not found in the provided dictionary.
    """

    patterns = {
        'braces': re.compile(r'\{(\w+)\}'),         # {variable}
        'double_braces': re.compile(r'\{\{(\w+)\}\}'),  # {{variable}}
        'dollar': re.compile(r'\$\{(\w+)\}')        # ${variable}
    }

    def replace_variable(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name in prompt_variables:
            return prompt_variables[var_name]
        else:
            return match.group(0)
    try:
        prompt = patterns['braces'].sub(replace_variable, prompt)

        prompt = patterns['double_braces'].sub(lambda m: replace_variable(m), prompt)

        prompt = patterns['dollar'].sub(lambda m: replace_variable(m), prompt)

    except ValueError as e:
        raise ValueError(f"Error in prompts replacement: {e}")

    return prompt
