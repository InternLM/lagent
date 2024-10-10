import json
import re
from typing import Dict, Optional, List, Any

import pandas as pd

from lagent.rag.doc import Storage
from lagent.rag.processors.extract_communities import get_community_hierarchy
from lagent.rag.prompts import COMMUNITY_REPORT_PROMPT
from lagent.rag.schema import MultiLayerGraph, Community, CommunityReport, CommunityContext, Node, Relationship
from lagent.rag.utils import filter_nodes_by_commu, filter_relas_by_nodes
from lagent.rag.utils import get_communities_context_reports_by_level, replace_variables_in_prompt
from lagent.rag.nlp import SimpleTokenizer
from lagent.rag.pipeline import register_processor, BaseProcessor
from lagent.rag.settings import DEFAULT_LLM_MAX_TOKEN


def get_levels_from_commu(communities: List[Community]):
    levels = []
    for community in communities:
        if community.level not in levels:
            levels.append(community.level)

    return levels


def select_sub_communities(community_id: str, level: int, community_hierarchy: Dict[int, Dict],
                           sub_level_communities_context, sub_level_reports) -> List[
    Dict[str, CommunityContext | CommunityReport]]:
    """
    根据给出的community_id以及community_hierarchy，得到对应的子社区并将其contexts与reports信息聚合，以字典的形式返回
    Args:
        level:
        community_id:
        community_hierarchy:
        sub_level_communities_context:
        sub_level_reports:

    Returns:

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
        # 子社区中可以不存在reports，但是必须含有context
        if sub_community_id not in id_map_context:
            raise ValueError(f"{sub_community_id} should exist in the given contexts")
        item['context'] = id_map_context[sub_community_id]
        if sub_community_id in id_map_reports:
            item['report'] = id_map_reports[sub_community_id]
        result.append(item)

    return result


def merge_info(level: int, communities: List[Community],
               level_nodes: List[Node], level_relas: List[Relationship], level_claims: Optional = None) -> List[
    CommunityContext]:
    """
    聚合给定level中每个community的信息，包括nodes、edges，并以context类的形式返回
    Args:
        level: 当前处理的level层次
        communities: 输入communities信息
        level_nodes:
        level_relas:
        level_claims:

    Returns:List[CommunityContext]

    """
    # merge得到的聚合消息需要有node.degree，node.name，node_details，edge_details,claim

    # 目前可以暂时使用字典形式，merged_CommunityContext: Dict[community_id, Dict[info_name, info]]
    merged_community_context: List[CommunityContext] = []
    for community in communities:
        if community.level != level:
            raise ValueError(f"community shouldn't exist in level_{level}")
        # if community.community_id not in merged_CommunityContext:
        #     merged_CommunityContext[community.community_id] = {}

        # 对于每个community，找到其内部的节点和边，并将其信息添加到context中。
        # 其中节点需要添加degree，description，name（id？），边需要添加source，target，description，degree，weight？
        community_nodes = filter_nodes_by_commu(level_nodes, community.community_id)
        community_relas = filter_relas_by_nodes(community_nodes, level_relas)
        community_claims = None
        # if level_claims is not None:
        #     community_claims = filter_claims_by_nodes(community_nodes, level_claims)

        nodes_info = {}
        for node in community_nodes:
            nodes_info[node.id] = {
                'id': node.id,
                'description': node.description,
                'degree': node.degree
            }
        # merged_community_context[community.community_id]['nodes_info'] = nodes_info

        relas_info = {}
        for rela in community_relas:
            relas_info[(rela.source, rela.target)] = {
                'source': rela.source,
                'target': rela.target,
                'description': rela.description if rela.description is not None else '',
                'degree': rela.degree
            }
        # merged_community_context[community.community_id]['relas_info'] = relas_info

        if community_claims is not None:
            pass

        merged_community_context.append(
            CommunityContext(community_id=community.community_id,
                             level=level,
                             nodes_info=nodes_info,
                             edges_info=relas_info,
                             claims=community_claims,
                             )
        )

    return merged_community_context


def get_context_str(nodes: Optional[List[Dict]] = None, relas: Optional[List[Dict]] = None,
                    claims: Optional[List[Dict]] = None,
                    sub_community_reports: Optional[List[CommunityReport]] = None) -> str:
    """
    根据community包含的信息获得信息string
    Args:
        nodes:
        relas:
        claims:
        sub_community_reports:

    Returns:

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

    if claims is not None:
        claims_df = pd.DataFrame(claims).drop_duplicates()
        claims_csv = claims_df.to_csv(index=False, sep=',')
        context_str.append(claims_csv)

    return '\n\n'.join(context_str)


@register_processor
class CommunityReportsExtractor(BaseProcessor):
    name = 'CommunityReportsExtractor'

    def __init__(self, llm, max_tokens: Optional[int] = None, tokenizer: Optional = None, prompt: Optional[str] = None):
        super().__init__(name='CommunityReportsExtractor')
        self.max_tokens = max_tokens or DEFAULT_LLM_MAX_TOKEN
        self.tokenizer = tokenizer or SimpleTokenizer()
        self.llm = llm
        if prompt is None:
            prompt = COMMUNITY_REPORT_PROMPT
        self.prompt = prompt

    def run(self, graph: MultiLayerGraph) -> MultiLayerGraph:
        """
        根据不同层次的communities上下文生成report
        Args:
            graph:

        Returns:

        """

        tokenizer = self.tokenizer
        # 首先获得不同level的nodes以及edges
        # 目前暂定在final_graph之后，会得到不同level对应的nodes（如果不加对每一个level的其他处理的化，实际上都是同样的nodes）

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

            # TODO
            level_claims = None

            # 整合信息需要以community为单位
            merged_communities_context = merge_info(level, level_communities, level_nodes, level_relas, level_claims)

            # 对每个community根据自身的context设置context_str
            for merged_community_context in merged_communities_context:

                context_str = self.set_context(contexts_info=[merged_community_context])
                merged_community_context.context_str = context_str
                merged_community_context.context_size = tokenizer.get_token_num(context_str)
                if tokenizer.get_token_num(context_str) > self.max_tokens:
                    merged_community_context.exceed_token = True

            merge_result[level] = merged_communities_context

        # 首先可以对获取得到的context进行检验：是否符合level，如果local_contexts未超过max_tokens，可以不用子社区的报告进行替代
        # 在local_contexts超过max_tokens限制的前提下，如果子社区reports为None，则只能对local_context做裁剪（这种情况只能发生在最底层level中）
        # 如果存在子社区，则根据子社区的上下文以及reports得到生成社区reports的context（首先合并子社区信息，之后进行裁剪）
        reports: List[CommunityReport] = []
        for level in levels:
            # 遍历当前level的每个community， 根据当前得到的reports以及社区的层次结构以及当前社区的context获取该community对应的context
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
        根据输入的community上下文信息列表，得到最终的上下文string并返回该community对应的上下文string
        Args:
            contexts_info:
            sub_community_reports:
            max_tokens:

        Returns:

        """
        community_context: str = ''
        # 对于max_tokens，如果需要在该阶段设置，则在后续的进一步进行上下文裁剪步骤不需要，然而这可能会面临信息丢失
        # 目前可以设置当前不对local_context进行裁剪，如果需要则在需要输入参数max_tokens
        # max_tokens = max_tokens or self.max_tokens
        tokenizer = self.tokenizer

        # 首先获取全部nodes，edges，claims信息
        nodes_info = {}
        relas_info = {}
        claims_info = {}
        for community_info in contexts_info:
            nodes_info = nodes_info | community_info.nodes_info
            relas_info = relas_info | community_info.edges_info
            if community_info.claims is not None:
                claims_info = claims_info | community_info.claims

        # 根据边的degree来排序
        sorted_relas_info = dict(sorted(relas_info.items(), key=lambda item: item[1]['degree'], reverse=True))
        sorted_nodes = []
        sorted_relas = []
        sorted_claims = None
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

            # TODO:claims

            new_context_str = get_context_str(sorted_nodes, sorted_relas, sorted_claims,
                                              sub_community_reports=sub_community_reports)

            if max_tokens is not None:
                if tokenizer.get_token_num(new_context_str) > max_tokens:
                    flag = True
                    break
            community_context = new_context_str
        if flag is False and len(sorted_nodes) < len(nodes_info.keys()):
            # 说明存在度数为0的节点信息没有被加入
            for _id, node_info in nodes_info.items():
                if node_info not in sorted_nodes:
                    sorted_nodes.append(node_info)
                    new_context_str = get_context_str(sorted_nodes, sorted_relas, sorted_claims)

                    if max_tokens is not None:
                        if tokenizer.get_token_num(new_context_str) > max_tokens:
                            break
                    community_context = new_context_str

        if community_context == '':
            # 一般是chunk_node，可能description较长导致的
            # 暂时的处理是缩短生成的description或增加max_llm_token
            community_context = get_context_str(sorted_nodes, sorted_relas, sorted_claims)

        return community_context

    def generate_context(self, context_str: str, community_context: CommunityContext) -> CommunityContext:
        """
        根据给出的context_str，将其添加到Community_context类中
        Args:
            community_context: 需要添加context_str的context
            context_str: 通过set_context得到的str

        Returns:

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
        对于给定level，根据要求的上下文长度，获得每个community生成report的上下文

        对于每个community：当merge_info满足上下文长度直接返回；否则根据子社区的上下文以及report生成上下文
        Returns:

        """

        tokenizer = self.tokenizer

        # 筛选需要上下文长度超过限制token数的_community_context
        _community_contexts = []
        indexes = []
        for index, _context in enumerate(community_contexts):
            if _context.exceed_token is True:
                indexes.append(index)
                _community_contexts.append(_context)
        if len(_community_contexts) == 0:
            # 说明每个community的上下文长度均未超过
            return community_contexts

        if not community_reports:
            # 当子社区的reports为空，只能对当前的context进行裁剪
            results = []
            for community_context in community_contexts:
                community_context_str = self.set_context(community_contexts, max_token=self.max_token)
                results.append(self.generate_context(community_context_str, community_context))

            return results

        sub_level_communities_context, sub_level_reports = get_communities_context_reports_by_level(level + 1,
                                                                                                    community_contexts,
                                                                                                    community_reports)
        if not sub_level_reports and not sub_level_communities_context:
            # 说明level对应的是最低层次，直接对context做裁剪
            results = []
            for community_context in community_contexts:
                community_context_str = self.set_context(community_contexts, max_token=self.max_token)
                results.append(self.generate_context(community_context_str, community_context))

            return results

        for k, _community_context in enumerate(_community_contexts):

            # 对每个community进行处理，首先得到contex在community_contexts中的index
            index = indexes[k]
            _community_id = _community_context.community_id
            exceed_flag = True

            # 得到community对应的sub_communities的上下文+reports信息，以list[dict]的形式返回
            _sub_communities = select_sub_communities(_community_id, level, community_hierarchy,
                                                      sub_level_communities_context,
                                                      sub_level_reports)
            # 从规模更大的sub_community开始获得上下文
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

                # 得到剩下sub_communities对应的context，如果满足token要求就可以直接返回
                _remaining_contexts = []
                for j in range(i + 1, len(_sub_communities) - 1):
                    _remaining_contexts.append(_sub_communities[j])

                # 根据contexts+reports得到context_str
                _context_str = self.set_context(contexts_info=_local_contexts + _remaining_contexts,
                                                sub_community_reports=_substitute_reports)

                if tokenizer.get_token_num(_context_str) <= self.max_token:
                    community_contexts[index].context_size = tokenizer.get_token_num(_context_str)
                    community_contexts[index].context_str = _context_str
                    exceed_flag = False
                    community_contexts[index].exceed_token = False
                    break

            if exceed_flag is True:
                # 说明无法将全部的sub_communities中的report信息加入到其中，优先加入report
                _reports = []
                _context_str = ''
                for _sub_report in _substitute_reports:
                    _reports.append(_sub_report)

                    # 将reports转化为str形式
                    _context_str = get_context_str(sub_community_reports=_reports)

                    if tokenizer.get_token_num(_context_str) > self.max_token:
                        break
                community_contexts[index].context_size = tokenizer.get_token_num(_context_str)
                community_contexts[index].context_str = _context_str
                community_contexts[index].exceed_token = False

    def generate_reports(self, community_contexts: List[CommunityContext], prompt: str) -> List[CommunityReport]:
        """
        根据给出的community上下文生成对应的report
        Args:
            prompt:
            community_contexts:

        Returns:

        """
        llm = self.llm
        result = []

        for community_context in community_contexts:
            community_id = community_context.community_id
            if community_context.context_size == 0 or community_context.exceed_token is True:
                raise ValueError
            prompt_variables = {
                'input_text': community_context.context_str
            }
            _prompt = replace_variables_in_prompt(prompt, prompt_variables)
            messages = [{"role": "user", "content": _prompt}]
            output = {}
            output_str = ''
            try:
                response = llm.chat(messages)
                # 使用正则表达式提取JSON部分
                json_match = re.search(r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}', response, re.DOTALL)

                # 检查是否成功匹配
                if json_match:
                    output_str = json_match.group(0)  # 提取JSON部分

                    # 将字符串转换为字典
                    try:
                        output = json.loads(output_str)
                    except json.JSONDecodeError as e:
                        print(f"JSON解码错误: {e}")
                else:
                    print("未找到JSON数据")
                output_str = transform_output(output)
            except Exception as e:
                print("error when generating report")

            result.append(
                CommunityReport(community_id=community_id, level=community_context.level,
                                report=output_str, structured_report=output)
            )

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
