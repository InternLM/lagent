from typing import Optional, List, Dict
from lagent.rag.schema import Chunk, Node, Relationship, MultiLayerGraph
from lagent.rag.doc import Storage
from lagent.rag.utils import normalize_edge, replace_variables_in_prompt
from lagent.rag.pipeline import register_processor, BaseProcessor
from lagent.rag.prompts import ENTITY_EXTRACTION_PROMPT

DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_ITEM_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]


@register_processor
class EntityExtractor(BaseProcessor):
    name = 'EntityExtractor'

    def __init__(self, llm, entity_types: Optional[List] = None, **kwargs):
        super().__init__('EntityExtractor')
        # TODO
        self.entity_types = entity_types or DEFAULT_ENTITY_TYPES
        self.llm = llm
        self.extraction_prompt = kwargs.get("prompts") or ENTITY_EXTRACTION_PROMPT
        self.tuple_delimiter = kwargs.get("tuple_delimiter", DEFAULT_TUPLE_DELIMITER)
        self.item_delimiter = kwargs.get("item_delimiter", DEFAULT_ITEM_DELIMITER)
        self.completion_delimiter = kwargs.get("completion_delimiter", DEFAULT_COMPLETION_DELIMITER)

    def run(self, graph: MultiLayerGraph, prompt_variables: Dict = None, **kwargs) -> MultiLayerGraph:
        """
        从chunk中抽取实体与实体之间的关系
        Args:
            graph:
            prompt_variables:
            **kwargs:

        Returns:

        """
        if prompt_variables is None:
            prompt_variables = {}
        entity_layer = graph.add_layer('entity_layer')

        chunk_layer = graph.layers['chunk_layer']
        chunks_dict = chunk_layer.get_nodes()
        chunks = []
        for chunk_dict in chunks_dict:
            chunk = Chunk(
                id=chunk_dict['id'],
                content=chunk_dict['content'],
                metadata=chunk_dict['metadata'],
                token_num=chunk_dict['token_num']
            )
            chunks.append(chunk)

        entity_types = prompt_variables.get("entity_types", self.entity_types)
        prompt_variables = {
            "tuple_delimiter": prompt_variables.get("tuple_delimiter", DEFAULT_TUPLE_DELIMITER),
            "item_delimiter": prompt_variables.get("item_delimiter", DEFAULT_ITEM_DELIMITER),
            "completion_delimiter": prompt_variables.get("completion", DEFAULT_COMPLETION_DELIMITER),
            "entity_types": ",".join(entity_types)
        }
        history = kwargs.get("history", [])

        entities = []
        relationships = []
        chunk_to_entities: Dict[str, List[str]] = {}
        prompt = replace_variables_in_prompt(self.extraction_prompt, prompt_variables)
        for index, chunk in enumerate(chunks):
            try:

                prompt = replace_variables_in_prompt(prompt, {"input_text": chunk.content})
                # TODO: get messages(history?)
                messages = [*history, {"role": "user", "content": prompt}]
                response = self.llm.chat(messages, **kwargs)
                _entities, _relationships = self.process_response(response)

                chunk_to_entities[chunk.id] = []
                for _entity in _entities:
                    _entity = dict_to_entity(_entity)
                    _entity.source_id = [chunk.id]
                    entities.append(_entity)
                    chunk_to_entities[chunk.id].append(_entity.content)
                for _relationship in _relationships:
                    _relationship = dict_to_relationship(_relationship)
                    relationships.append(_relationship)

            except Exception as e:
                raise ValueError

        id_map_entities, id_map_relas = merge_graph(entities, relationships)
        entities = list(id_map_entities.values())
        relationships = list(id_map_relas.values())

        # save
        # storage = Storage()
        # entities_dict = [entity.to_dict() for entity in entities]
        # relationships_dict = [rela.to_dict() for rela in relationships]
        # storage.put('entities', entities_dict)
        # storage.put('relationships', relationships_dict)
        # storage.put('chunk_to_entities', chunk_to_entities)

        for entity in entities:
            node_attr = {
                'source_id': entity.source_id,
                'description': entity.description,
                'entity_type': entity.entity_type
            }
            entity_layer.add_node(entity.id, **node_attr)

        for relationship in relationships:
            edge_attr = {
                'description': relationship.description,
                'weight': relationship.weight
            }
            entity_layer.add_edge(relationship.source, relationship.target, **edge_attr)

        nodes = entity_layer.graph.nodes(data=True)
        id_map_nodes = {}
        for node in nodes:
            id_map_nodes[node[0]] = node[1]

        for id_degree in entity_layer.graph.degree():
            node_id = id_degree[0]
            degree = id_degree[1]
            id_map_nodes[node_id]['degree'] = degree

        nodes = entity_layer.graph.nodes(data=True)
        id_map_nodes = {}
        for node in nodes:
            id_map_nodes[node[0]] = node[1]

        edges = entity_layer.graph.edges(data=True)
        for edge in edges:
            source = edge[0]
            target = edge[1]
            degree1 = id_map_nodes[source]['degree']
            degree2 = id_map_nodes[target]['degree']
            degree = degree1 + degree2
            edge[2]['degree'] = degree

        for chunk_id, entities_id in chunk_to_entities.items():
            for entity_id in entities_id:
                graph.add_interlayer_edge('chunk_layer', chunk_id, 'entity_layer', entity_id)

        return graph

    def process_response(self, response: str):
        """
            Parses the LLM response to extract entities and relationships.
            Args:
                response (str): The output text from the language model.
            Returns:
                Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: A tuple containing lists of entity dictionaries
                and relationship dictionaries.
            Raises:
                ValueError: If the response is incomplete or improperly formatted.
        """
        tuple_delimiter = self.tuple_delimiter
        item_delimiter = self.item_delimiter
        completion_delimiter = self.completion_delimiter

        # Check if the response ends with the completion delimiter
        if not response.endswith(completion_delimiter):
            raise ValueError("LLM output is incomplete or not correctly terminated.")

        response = response.rstrip(completion_delimiter).strip()

        if response.startswith("'''") and response.endswith("'''"):
            response = response[3:-3].strip()

        items = response.split(item_delimiter)

        entities = []
        relationships = []

        for item in items:
            item = item.strip()
            if not item:
                continue

            if item.startswith('\n'):
                item = item[1:].strip()
            if item.endswith('\n'):
                item = item[:-1].strip()
            if item.startswith("(") and item.endswith(")"):
                item = item[1:-1].strip()

            item = item.strip("'").strip()

            parts = item.split(tuple_delimiter)

            if parts[0] == "\"entity\"":
                if len(parts) < 4:
                    raise ValueError(f"Unexpected format for entity: {item}")
                entity = {
                    "entity_name": parts[1],
                    "entity_type": parts[2],
                    "description": parts[3]
                }
                entities.append(entity)

            elif parts[0] == "\"relationship\"":
                if len(parts) < 5:
                    raise ValueError(f"Unexpected format for relationship: {item}")
                relationship = {
                    "source_entity": parts[1],
                    "target_entity": parts[2],
                    "description": parts[3],
                    "weight": parts[4]
                }
                relationships.append(relationship)

        return entities, relationships


def dict_to_entity(entity_dict: Dict) -> Optional[Node]:
    try:
        name = entity_dict.get("entity_name")

        # when getting an entity, source_id is not added so far(added in nodes)
        entity_type = entity_dict.get("entity_type")
        description = entity_dict.get("description")
    except Exception as e:
        print("error in dict_to_entity")
        return None
    return Node(
        type="entity",
        content=name,
        id=name,
        description=description,
        entity_type=entity_type
    )


def dict_to_relationship(relationship_dict: Dict) -> Optional[Relationship]:
    try:
        _weight = float(relationship_dict.get("weight"))
        _source_entity = relationship_dict.get("source_entity")
        _target_entity = relationship_dict.get("target_entity")
        # use id
        # _source_id = hashlib.md5(_source_entity.encode('utf-8')).hexdigest()
        # _target_id = hashlib.md5(_target_entity.encode('utf-8')).hexdigest()
        _description = relationship_dict.get("description", None)

    except Exception as e:
        print("error in dict_to_relationship", e)
        return None

    return Relationship(
        weight=_weight,
        source=_source_entity,
        target=_target_entity,
        description=_description
    )


def merge_graph(entities: List[Node], relationships: List[Relationship]):
    merged_entities: Dict[str, Node] = {}
    merged_relationships: Dict[(str, str), Relationship] = {}
    for entity in entities:
        if entity.id in merged_entities:
            existing_entity = merged_entities[entity.id]
            # merge descriptions
            if entity.description:
                if existing_entity.description:
                    # avoid redundancy
                    descri_list = existing_entity.description.split(" | ")
                    if entity.description not in descri_list:
                        existing_entity.description += f' | {entity.description}'
                else:
                    existing_entity.description = f'{entity.description}'
            # merge source_ids
            if entity.source_id[0] not in existing_entity.source_id:
                existing_entity.source_id.append(entity.source_id[0])

        else:
            merged_entities[entity.id] = entity

    for relationship in relationships:
        relationship = normalize_edge(relationship)
        key = (relationship.source, relationship.target)
        if key in merged_relationships:
            existing_relationship = merged_relationships[key]
            if relationship.description:
                if existing_relationship.description:
                    descri_list = existing_relationship.description.split(" | ")
                    if relationship.description not in descri_list:
                        existing_relationship.description += f' | {relationship.description}'
                else:
                    existing_relationship.description = f'{relationship.description}'
        else:
            merged_relationships[key] = relationship

    return merged_entities, merged_relationships
