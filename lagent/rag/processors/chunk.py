# Copyright (c) 2024 Alibaba Cloud.
# Licensed under the Tongyi Qianwen LICENSE AGREEMENT.
#
# Based on the code from Qwen-Agent project (https://github.com/alibaba/Qwen-Agent),
# variable names and some logic statements have been modified for clarity and performance.
# Original Author: Alibaba Cloud
# Modified by [kxZhou621] on [2024.10.15]

from lagent.rag.schema import Document, Chunk, MultiLayerGraph
from lagent.rag.settings import DEFAULT_CHUNK_SZIE, DEFAULT_OVERLAP
from lagent.rag.nlp import SimpleTokenizer as Tokenizer
from lagent.rag.pipeline import register_processor, BaseProcessor

from typing import Optional, Dict, List
import re


@register_processor
class ChunkSplitter(BaseProcessor):
    name = 'ChunkSplitter'

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(name='ChunkSplitter')
        if cfg is None:
            cfg = {}
        self.cfg = cfg
        self.chunk_size = self.cfg.get('chunk_size', DEFAULT_CHUNK_SZIE)
        self.overlap_len = self.cfg.get('overlap', min(int(self.chunk_size * 0.2), DEFAULT_OVERLAP))

    def run(self, graph: MultiLayerGraph) -> MultiLayerGraph:
        all_chunks = []
        chunk_layer = graph.add_layer('chunk_layer')

        document_layer = graph.layers.get('document_layer', None)
        if document_layer is None:
            raise ValueError("when building chunks, document_layer doesn't exist")
        for document in document_layer.get_nodes():
            document = Document(
                id=document['id'],
                content=document['content'],
                metadata=document['metadata']
            )
            chunks = self.split_into_chunks(document)
            all_chunks.extend(chunks)

        # add nodes and interlayer_edges
        for chunk in all_chunks:
            node_attr = {
                'content': chunk.content,
                'metadata': chunk.metadata,
                'token_num': chunk.token_num
            }
            chunk_layer.add_node(chunk.id, **node_attr)

            graph.add_interlayer_edge('document_layer', chunk.metadata['source'], 'chunk_layer', chunk.id)

        return graph

    def get_overlap(self, chunk: list) -> Optional[str]:
        overlap_len = self.overlap_len
        overlap = ''
        flag = False
        length = len(chunk)
        for i in range(len(chunk) - 1, -1, -1):
            # 逆序遍历chunk
            text = chunk[i][0]
            if flag is True:
                break
            if len(text) <= overlap_len:
                if overlap:
                    overlap = f'{text}\n{overlap}'
                else:
                    overlap = f'{text}'
                overlap_len -= len(text)
            else:
                sentences = re.split(r'\. |。', text)
                for j in range(len(sentences) - 1, -1, -1):
                    sentence = sentences[j]
                    if len(sentence) <= overlap_len:
                        if overlap:
                            overlap = f'{sentence}\n{overlap}'
                        else:
                            overlap = f'{sentence}'
                        overlap_len -= len(sentence)
                    else:
                        min_sentences = re.split(r'[，,]', sentence)
                        for z in range(len(min_sentences) - 1, -1, -1):
                            min_sentence = min_sentences[z]
                            if len(min_sentence) <= overlap_len:
                                if overlap:
                                    overlap = f'{min_sentence}, {overlap}'
                                else:
                                    overlap = f'{min_sentence}'
                                overlap_len -= len(min_sentence)
                            else:
                                flag = True
                            if flag is True:
                                break
                    if flag is True:
                        break
        return overlap

    def split_into_chunks(self, document: Document) -> List[Chunk]:

        res = []
        chunk = []
        chunk_size = self.chunk_size
        available_token = self.chunk_size
        tokenizer = Tokenizer()
        content = document.content
        para_flag = False
        for page in content:
            page_num = page['page_num']
            if not chunk or f'[page: {str(page_num)}]' != chunk[0]:
                # add page tag
                chunk.append(f'[page: {str(page_num)}]')
            i = 0
            page_content = page['content']['text']
            len_para = len(page_content)
            while i < len_para:
                para = page_content[i].strip()
                token_num = tokenizer.get_token_num(para)
                if token_num < available_token:
                    chunk.append([para, page_num])
                    available_token -= token_num
                    i += 1
                    para_flag = True
                else:
                    if para_flag:
                        if isinstance(chunk[-1], str) and re.fullmatch(r'^\[page: \d+\]$', chunk[-1]) is not None:
                            chunk.pop()  # Redundant page information
                        res.append(Chunk(content='\n'.join([x if isinstance(x, str) else x[0] for x in chunk]),
                                         id=f"{document.id}{len(res)}",
                                         token_num=chunk_size - available_token,
                                         metadata={'source': document.id}))
                        overlap = self.get_overlap(chunk)
                        if overlap is not None and overlap != '':
                            chunk = [f'[page: {str(chunk[-1][1])}]', overlap]
                            available_token = chunk_size - tokenizer.get_token_num(overlap)
                            para_flag = False
                        else:
                            chunk = []
                            available_token = chunk_size
                            para_flag = False
                    else:
                        sentences = []
                        sentences = re.split(r'\. |。', para)
                        j = 0
                        while j < len(sentences):
                            sentence = sentences[j]
                            sentence = sentence.strip()
                            num = tokenizer.get_token_num(sentence)
                            if num == 0:
                                continue
                            if num <= available_token:
                                chunk.append([sentence, page_num])
                                available_token -= num
                                j += 1
                                para_flag = True
                            else:
                                if para_flag is False:
                                    chunk.append([sentence, page_num])
                                if isinstance(chunk[-1], str) and re.fullmatch(r'^\[page: \d+\]$',
                                                                               chunk[-1]) is not None:
                                    chunk.pop()
                                res.append(Chunk(content='\n'.join([x if isinstance(x, str) else x[0] for x in chunk]),
                                                 id=f"{document.id}{len(res)}",
                                                 token_num=chunk_size - available_token,
                                                 metadata={'source': document.id}))
                                overlap = self.get_overlap(chunk)
                                if overlap and overlap != '':
                                    chunk = [f'[page: {str(chunk[-1][1])}]', overlap]
                                    available_token = chunk_size - tokenizer.get_token_num(overlap)
                                    para_flag = False
                                else:
                                    chunk = []
                                    available_token = chunk_size
                                    para_flag = False
                        i += 1
        if para_flag:
            if isinstance(chunk[-1], str) and re.fullmatch(r'^\[page: \d+\]$', chunk[-1]) is not None:
                chunk.pop()

            res.append(Chunk(content='\n'.join([x if isinstance(x, str) else x[0] for x in chunk]),
                             id=f"{document.id}{len(res)}",
                             token_num=chunk_size - available_token,
                             metadata={'source': document.id}))
        return res
