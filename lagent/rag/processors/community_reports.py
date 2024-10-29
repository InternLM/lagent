import json
import re
from typing import Dict, Optional, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

from lagent.rag.doc import Storage
from lagent.utils import create_object
from lagent.llms.deepseek import DeepseekAPI
from lagent.rag.processors.extract_communities import get_community_hierarchy
from lagent.rag.prompts import COMMUNITY_REPORT_PROMPT
from lagent.rag.schema import MultiLayerGraph, Community, CommunityReport, CommunityContext, Node, Relationship
from lagent.rag.utils import filter_nodes_by_commu, filter_relas_by_nodes
from lagent.rag.utils import get_communities_context_reports_by_level, replace_variables_in_prompt
from lagent.rag.nlp import SimpleTokenizer
from lagent.rag.pipeline import register_processor, BaseProcessor
from lagent.rag.settings import DEFAULT_LLM_MAX_TOKEN


def get_levels_from_commu(communities: List[Community]):
    """
        Extracts levels from a list of communities.

        Args:
            communities (List[Community]): A list of Community instances.

        Returns:
            List[int]: A list of unique levels present in the communities.
    """
    levels = []
    for community in communities:
        if community.level not in levels:
            levels.append(community.level)

    return levels


def select_sub_communities(community_id: str, level: int, community_hierarchy: Dict[int, Dict],
                           sub_level_communities_context, sub_level_reports) -> List[
    Dict[str, CommunityContext | CommunityReport]]:
    """
        Retrieves sub-communities based on the given community ID and hierarchy, aggregating their contexts and reports.

        Args:
            community_id (str): The ID of the parent community.
            level (int): The current level in the community hierarchy.
            community_hierarchy (Dict[int, Dict[str, List[str]]]): Hierarchical mapping of communities.
            sub_level_communities_context (List[CommunityContext]): Contexts of sublevel communities.
            sub_level_reports (List[CommunityReport]): Reports of sublevel communities.

        Returns:
            List[Dict[str, Union[CommunityContext, CommunityReport]]]:
                A list of dictionaries containing the context and optional report of each sub-community.

        Raises:
            ValueError: If the specified community ID does not exist in the given level.
    """
    communities_by_level = community_hierarchy[level]
    if community_id not in communities_by_level:
        raise ValueError(f"community{community_id} doesn't exist in level{level} in the hierarchy")

    id_map_context = {}
    for context in sub_level_communities_context:
        id_map_context[context.community_id] = context

    id_map_reports = {}
    for report in sub_level_reports:
        id_map_reports[report.community_id] = report

    result = []
    sub_community_ids = communities_by_level[community_id]
    for sub_community_id in sub_community_ids:
        item = {}
        # Sub-communities may not have reports but must have context
        if sub_community_id not in id_map_context:
            raise ValueError(f"{sub_community_id} should exist in the given contexts")
        item['context'] = id_map_context[sub_community_id]
        if sub_community_id in id_map_reports:
            item['report'] = id_map_reports[sub_community_id]
        result.append(item)

    return result


def merge_info(level: int, communities: List[Community],
               level_nodes: List[Node], level_relas: List[Relationship]) -> List[CommunityContext]:
    """
        Aggregates information for each community at a given level, including nodes and relationships.

        Args:
            level (int): The current level being processed.
            communities (List[Community]): A list of Community instances.
            level_nodes (List[Node]): A list of Node instances at the current level.
            level_relas (List[Relationship]): A list of Relationship instances at the current level.

        Returns:
            List[CommunityContext]: A list of CommunityContext instances containing aggregated information.

        Raises:
            ValueError: If a community does not belong to the specified level.
    """
    merged_community_context: List[CommunityContext] = []
    for community in communities:
        if community.level != level:
            raise ValueError(f"community shouldn't exist in level_{level}")

        community_nodes = filter_nodes_by_commu(level_nodes, community.community_id)
        community_relas = filter_relas_by_nodes(community_nodes, level_relas)

        nodes_info = {}
        for node in community_nodes:
            nodes_info[node.id] = {
                'id': node.id,
                'description': node.description,
                'degree': node.degree
            }

        relas_info = {}
        for rela in community_relas:
            relas_info[(rela.source, rela.target)] = {
                'source': rela.source,
                'target': rela.target,
                'description': rela.description if rela.description is not None else '',
                'degree': rela.degree
            }

        merged_community_context.append(
            CommunityContext(community_id=community.community_id,
                             level=level,
                             nodes_info=nodes_info,
                             edges_info=relas_info
                             )
        )

    return merged_community_context


def get_context_str(nodes: Optional[List[Dict]] = None, relas: Optional[List[Dict]] = None,
                    sub_community_reports: Optional[List[CommunityReport]] = None) -> str:
    """
        Constructs a context string based on the provided nodes, relationships, and sub-community reports.

        Args:
            nodes (Optional[List[Dict[str, Any]]]): A list of node dictionaries.
            relas (Optional[List[Dict[str, Any]]]): A list of relationship dictionaries.
            sub_community_reports (Optional[List[CommunityReport]]): A list of sub-community reports.

        Returns:
            str: The constructed context string.
    """
    context_str = []

    if sub_community_reports is not None:
        sub_reports = [sub_community_report.report for sub_community_report in sub_community_reports]
        if sub_reports:
            sub_reports_df = pd.DataFrame(sub_reports).drop_duplicates()
            sub_reports_csv = sub_reports_df.to_csv(index=False, sep=',')
            context_str.append(f'--reports--\n{sub_reports_csv}')

    if nodes is not None:
        nodes_df = pd.DataFrame(nodes).drop_duplicates()
        nodes_csv = nodes_df.to_csv(index=False, sep=',')
        context_str.append(f'--nodes--\n{nodes_csv}')

    if relas and relas is not None:
        relas_df = pd.DataFrame(relas).drop_duplicates()
        relas_csv = relas_df.to_csv(index=False, sep=',')
        context_str.append(f'--relationships--\n{relas_csv}')

    return '\n\n'.join(context_str)


@register_processor
class CommunityReportsExtractor(BaseProcessor):
    """
        Processor that extracts and generates reports for communities within a multi-layer graph.

        The CommunityReportsExtractor processes a MultiLayerGraph to generate structured reports for each community
        based on their context. It handles different levels of communities, aggregates relevant information, and
        interacts with a language model to produce comprehensive reports.
    """

    name = 'CommunityReportsExtractor'

    def __init__(self,
                 llm=dict(type=DeepseekAPI),
                 max_tokens: int = DEFAULT_LLM_MAX_TOKEN,
                 tokenizer=dict(type=SimpleTokenizer),
                 prompt: str = COMMUNITY_REPORT_PROMPT):
        super().__init__(name='CommunityReportsExtractor')
        self.tokenizer = create_object(tokenizer)
        self.llm = create_object(llm)
        self.prompt = prompt
        self.max_tokens = max_tokens

    def run(self, graph: MultiLayerGraph) -> MultiLayerGraph:

        tokenizer = self.tokenizer

        community_layer = graph.layers['community_layer']
        dict_communities = community_layer.get_nodes()
        communities = []
        for dict_community in dict_communities:
            communities.append(Community(
                community_id=dict_community['id'],
                level=dict_community['level'],
                nodes_id=dict_community['nodes_id']
            ))

        levels = get_levels_from_commu(communities)

        merge_result: Dict[int, Any] = {}

        community_hierarchy = get_community_hierarchy(communities, levels)

        for level in levels:

            level_layer = graph.layers[f'level{level}_entity_layer']
            level_nodes = level_layer.get_nodes()
            level_relas = level_layer.get_edges()
            level_nodes = [Node.dict_to_node(node) for node in level_nodes]
            level_relas = [Relationship.dict_to_edge(rela) for rela in level_relas]
            level_communities = []
            for community in communities:
                if community.level == level:
                    level_communities.append(community)

            merged_communities_context = merge_info(level, level_communities, level_nodes, level_relas)

            for merged_community_context in merged_communities_context:

                context_str = self.set_context(contexts_info=[merged_community_context])
                merged_community_context.context_str = context_str
                merged_community_context.context_size = tokenizer.get_token_num(context_str)
                if tokenizer.get_token_num(context_str) > self.max_tokens:
                    merged_community_context.exceed_token = True

            merge_result[level] = merged_communities_context

        # Generate reports based on the aggregated contexts
        reports: List[CommunityReport] = []
        for level in levels:
            level_contexts = self.prepare_report_context(level=level, community_hierarchy=community_hierarchy,
                                                         community_contexts=merge_result[level],
                                                         community_reports=reports)
            local_reports = self.generate_reports(level_contexts, self.prompt)

            reports.extend(local_reports)

        storage = Storage()
        reports_storage = [report.to_dict() for report in reports]
        storage.put("community_reports", reports_storage)

        report_layer = graph.add_layer('community_report_layer')
        for report in reports:
            attr = {
                'community_id': report.community_id,
                'level': report.level,
                'report': report.report,
                'structured_report': report.structured_report
            }
            report_layer.add_node(report.community_id, **attr)

        return graph

    def set_context(self, contexts_info: List[CommunityContext],
                    sub_community_reports: Optional[List[CommunityReport]] = None,
                    max_tokens: Optional[int] = None) -> str:
        """
            Constructs the final context string for a community based on its context information and sub-community
            reports.

            Args:
                contexts_info (List[CommunityContext]): A list of CommunityContext instances used for constructing
                the target community context.
                sub_community_reports (Optional[List[CommunityReport]]): A list of CommunityReport instances for
                sub-communities.
                    Defaults to None.
                max_tokens (Optional[int]): The maximum number of tokens allowed for the context string.
                    If not provided, then there is no limit for the number of tokens.

            Returns:
                str: The constructed context string.
        """
        community_context: str = ''
        tokenizer = self.tokenizer

        # Aggregate all nodes and relationships
        nodes_info = {}
        relas_info = {}
        for community_info in contexts_info:
            nodes_info = nodes_info | community_info.nodes_info
            relas_info = relas_info | community_info.edges_info

        # Sort relationships based on degree in descending order
        sorted_relas_info = dict(sorted(relas_info.items(), key=lambda item: item[1]['degree'], reverse=True))
        sorted_nodes = []
        sorted_relas = []
        flag = False
        for k, rela_info in sorted_relas_info.items():

            source = rela_info['source']
            target = rela_info['target']
            source_node = nodes_info[source]
            target_node = nodes_info[target]
            if source_node not in sorted_nodes:
                sorted_nodes.append(source_node)
            if target_node not in sorted_nodes:
                sorted_nodes.append(target_node)

            sorted_relas.append(rela_info)

            new_context_str = get_context_str(sorted_nodes, sorted_relas,
                                              sub_community_reports=sub_community_reports)

            if max_tokens is not None:
                if tokenizer.get_token_num(new_context_str) > max_tokens:
                    flag = True
                    break
            community_context = new_context_str
        if flag is False and len(sorted_nodes) < len(nodes_info.keys()):
            for _id, node_info in nodes_info.items():
                if node_info not in sorted_nodes:
                    sorted_nodes.append(node_info)
                    new_context_str = get_context_str(sorted_nodes, sorted_relas)

                    if max_tokens is not None:
                        if tokenizer.get_token_num(new_context_str) > max_tokens:
                            break
                    community_context = new_context_str

        if community_context == '':
            # Handle cases where context is too long by reducing content or increasing token limit
            community_context = get_context_str(sorted_nodes, sorted_relas)

        return community_context

    def update_context(self, context_str: str, community_context: CommunityContext) -> CommunityContext:
        """
            Updates the CommunityContext instance with the provided context string and token count.

            Args:
                context_str (str): The context string to be added.
                community_context (CommunityContext): The CommunityContext instance to update.

            Returns:
                CommunityContext: The updated CommunityContext instance.
        """
        tokenizer = self.tokenizer
        community_context.context_str = context_str
        community_context.context_size = tokenizer.get_token_num(context_str)
        if community_context.context_size > self.max_tokens:
            community_context.exceed_token = True
        else:
            community_context.exceed_token = False

        return community_context

    def prepare_report_context(self, level: int, community_hierarchy: Dict,
                               community_contexts: List[CommunityContext],
                               community_reports: List[CommunityReport]) -> List[CommunityContext]:
        """
            Prepares the context for report generation based on the community hierarchy and existing reports.

            Args:
                level (int): The current level in the community hierarchy.
                community_hierarchy (Dict[int, Dict[str, List[str]]]): The community hierarchy mapping.
                community_contexts (List[CommunityContext]): A list of CommunityContext instances.
                community_reports (List[CommunityReport]): A list of existing CommunityReport instances.

            Returns:
                List[CommunityContext]: A list of CommunityContext instances with updated context strings.
        """

        tokenizer = self.tokenizer

        # Identify communities exceeding token limits
        _community_contexts = []
        indexes = []
        for index, _context in enumerate(community_contexts):
            if _context.exceed_token is True:
                indexes.append(index)
                _community_contexts.append(_context)
        if len(_community_contexts) == 0:
            return community_contexts

        if not community_reports:
            # If there are no sub-community reports, trim the current contexts
            results = []
            for community_context in community_contexts:
                community_context_str = self.set_context(community_contexts, max_tokens=self.max_tokens)
                results.append(self.update_context(community_context_str, community_context))

            return results

        sub_level_communities_context, sub_level_reports = get_communities_context_reports_by_level(level + 1,
                                                                                                    community_contexts,
                                                                                                    community_reports)
        if not sub_level_reports and not sub_level_communities_context:
            results = []
            for community_context in community_contexts:
                community_context_str = self.set_context(community_contexts, max_tokens=self.max_tokens)
                results.append(self.update_context(community_context_str, community_context))

            return results

        for k, _community_context in enumerate(_community_contexts):

            index = indexes[k]
            _community_id = _community_context.community_id
            exceed_flag = True

            _sub_communities = select_sub_communities(_community_id, level, community_hierarchy,
                                                      sub_level_communities_context,
                                                      sub_level_reports)
            # Sort sub-communities based on context size in descending order
            _sub_communities = sorted(_sub_communities, key=lambda x: x['context'].context_size, reverse=True)

            _substitute_reports = []
            _local_contexts = []
            for i, _sub_community in enumerate(_sub_communities):
                _sub_community_report = _sub_community.get('report', None)
                _sub_community_context = _sub_community['context']
                if _sub_community_report is not None:
                    _substitute_reports.append(_sub_community_report)
                else:
                    _local_contexts.append(_sub_community_context)
                    continue

                _remaining_contexts = []
                for j in range(i + 1, len(_sub_communities) - 1):
                    _remaining_contexts.append(_sub_communities[j])

                _context_str = self.set_context(contexts_info=_local_contexts + _remaining_contexts,
                                                sub_community_reports=_substitute_reports)

                if tokenizer.get_token_num(_context_str) <= self.max_tokens:
                    community_contexts[index].context_size = tokenizer.get_token_num(_context_str)
                    community_contexts[index].context_str = _context_str
                    exceed_flag = False
                    community_contexts[index].exceed_token = False
                    break

            if exceed_flag is True:
                _reports = []
                _context_str = ''
                for _sub_report in _substitute_reports:
                    _reports.append(_sub_report)

                    _context_str = get_context_str(sub_community_reports=_reports)

                    if tokenizer.get_token_num(_context_str) > self.max_tokens:
                        break
                community_contexts[index].context_size = tokenizer.get_token_num(_context_str)
                community_contexts[index].context_str = _context_str
                community_contexts[index].exceed_token = False

    def generate_reports(self, community_contexts: List[CommunityContext], prompt: str) -> List[CommunityReport]:
        llm = self.llm
        result = []

        def process_single_context(community_context: CommunityContext) -> CommunityReport:
            community_id = community_context.community_id
            if community_context.context_size == 0 or community_context.exceed_token is True:
                raise ValueError(f"Invalid context for community {community_id}")

            prompt_variables = {
                'input_text': community_context.context_str
            }
            _prompt = replace_variables_in_prompt(prompt, prompt_variables)
            messages = [{"role": "user", "content": _prompt}]
            output = {}
            output_str = ''
            try:
                response = llm.chat(messages)
                # Extract JSON part from the response using regex
                json_match = re.search(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', response, re.DOTALL)

                if json_match:
                    output_str = json_match.group(0)
                    try:
                        output = json.loads(output_str)
                    except json.JSONDecodeError as e:
                        raise e

                output_str = transform_output(output)
            except Exception as e:
                raise ValueError(f"Error when generating report for community {community_id}: {e}")

            return CommunityReport(
                community_id=community_id,
                level=community_context.level,
                report=output_str,
                structured_report=output
            )

        max_workers = min(32, (len(community_contexts) or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_context = {executor.submit(process_single_context, context): context for context in
                                 community_contexts}

            for future in as_completed(future_to_context):
                context = future_to_context[future]
                try:
                    report = future.result()
                    result.append(report)
                except Exception as e:
                    raise e
        return result


def transform_output(output: dict):
    title = output.get('title', 'Report')
    summary = output.get('summary', '')
    insights = output.get('findings', [])

    def finding_summary(finding: dict):
        if isinstance(finding, str):
            return finding
        return finding.get("summary")

    def finding_explanation(finding: dict):
        if isinstance(finding, str):
            return ""
        return finding.get("explanation")

    report_sections = "\n\n".join(
        f"## {finding_summary(f)}\n\n{finding_explanation(f)}" for f in insights
    )

    return f"# {title}\n\n{summary}\n\n{report_sections}"
