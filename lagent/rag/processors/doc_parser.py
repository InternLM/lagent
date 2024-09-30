import os
from typing import Optional, Dict, Union, List
from urllib.parse import urlparse
import hashlib
import requests

from lagent.rag.schema import Document, MultiLayerGraph
from lagent.rag.doc import Storage
from lagent.rag.doc import PdfParser, DocxParser
from lagent.rag.pipeline import register_processor, BaseProcessor


@register_processor
class DocParser(BaseProcessor):
    name = 'DocParser'
    expected_input_type = List
    expected_output_type = MultiLayerGraph

    def __init__(self, cfg: Optional[Dict] = None, storage: Optional[Storage] = None):
        super().__init__(name='DocParser')
        if cfg is None:
            cfg = {}
        self.cfg = cfg  # 可选的配置字典
        self.storage = storage or Storage()  # 默认缓存目录

    def run(self, files: List[Union[str, dict]]) -> MultiLayerGraph:

        if isinstance(files[0], dict):
            files = [file.get('path', '') for file in files]

        multilayergraph = MultiLayerGraph()
        document_layer = multilayergraph.add_layer("document_layer")

        all_documents = []

        for file in files:
            # 生成文件的唯一缓存名
            cache_name = hashlib.md5(file.encode('utf-8')).hexdigest()

            # 检查缓存中是否已经存在分块的内容
            cached_document = self.storage.get(cache_name)
            if cached_document is not None and cached_document != []:
                cached_document = Document.dict_to_document(cached_document)
                print(f"Loading from cache: {cache_name}")
                all_documents.append(cached_document)

            # 如果缓存不存在，则处理文件
            if self.is_url(file):
                local_path = self.handle_url(file)
                if not local_path:
                    raise Exception(f"Failed to download or find the file: {file}")
            else:
                local_path = file
                if not os.path.exists(local_path):
                    raise FileNotFoundError(f"File not found: {local_path}")

            if local_path:
                document = self.parse_file(local_path)
                all_documents.append(document)

        for document in all_documents:
            node_attr = {
                'content': document.content,
                'metadata': document.metadata,
            }
            document_layer.add_node(document.id, **node_attr)

        return multilayergraph

    def is_url(self, path_or_url: str) -> bool:
        """检查字符串是否是URL"""
        try:
            result = urlparse(path_or_url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def handle_url(self, url: str) -> Optional[str]:
        """处理URL，下载文件到本地，如果已经存在则直接返回本地路径"""
        local_filename = os.path.basename(urlparse(url).path)
        try:
            if not os.path.exists(local_filename):
                print(f"Downloading {url}...")
                response = requests.get(url)
                response.raise_for_status()  # 检查HTTP请求的状态码
                with open(local_filename, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {local_filename}")
            else:
                print(f"File {local_filename} already exists.")
            return local_filename
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")
            return None
        except IOError as e:
            print(f"Failed to write file {local_filename}: {e}")
            return None

    def parse_file(self, file_path: str) -> Optional[Document]:
        """根据文件扩展名选择适当的解析器，并返回一个 Document 实例"""
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        try:
            if ext == '.txt':
                document = self.parse_text(file_path)
                return document
            elif ext == '.pdf':
                return self.parse_pdf(file_path)
            elif ext == '.docx':
                return self.parse_docx(file_path)
            else:
                print(f"Unsupported file type: {ext}")
                return None
        except Exception as e:
            print(f"Failed to parse file {file_path}: {e}")
            return None

    def parse_text(self, file_path: str) -> Optional[Document]:
        """解析文本文件并返回 Document 实例"""
        content = []
        try:

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    content.append(line.strip())  # 去除每行的前后空白符并添加到列表中
            full_content = '\n'.join(content)  # 将列表中的行合并为一个字符串
            cleaned_content = self.clean_content(full_content)
            paras = cleaned_content.split('\n')
            content = [{'page_num': 1, 'content': {'text': [f'{para}' for para in paras if para.strip() != '']}}]
            document_id = os.path.basename(file_path)
            try:
                # 尝试创建 Document 实例
                document = Document(
                    content=content,
                    id=document_id,
                    metadata={"file_type": "text", "file_path": file_path}
                )
            except Exception as e:
                # 如果实例化失败，打印错误信息
                print(f"Error instantiating Document for file {file_path}: {e}")
                return None

            return document
        except IOError as e:
            print(f"Failed to read text file {file_path}: {e}")
            return None

    def parse_pdf(self, file_path: str) -> Document:
        """解析PDF文件并返回 Document 实例"""
        # TODO
        try:
            pdf_parser = PdfParser()
            content = pdf_parser.parse(file_path=file_path)  # 这里应该是实际提取的内容
            content = [{'page_num': 1, 'content': content}]
            document = Document(
                content=content,
                id=os.path.basename(file_path),
                metadata={"file_type": "pdf", "file_path": file_path}
            )
            return document
        except Exception as e:
            print(f"Failed to parse PDF file {file_path}: {e}")
            return None

    def parse_docx(self, file_path: str) -> Document:
        """解析Word文档并返回 Document 实例"""
        # TODO
        try:
            doxc_parser = DocxParser()
            content = doxc_parser.parse(file_path=file_path)  # 这里应该是实际提取的内容
            content = [{'page_num': 1, 'content': content}]
            document = Document(
                content=content,
                id=os.path.basename(file_path),
                metadata={"file_type": "pdf", "file_path": file_path}
            )
            return document
        except Exception as e:
            print(f"Failed to parse DOCX file {file_path}: {e}")
            return None

    def clean_content(self, content: str) -> str:
        """对文件内容进行清洗处理"""
        # 这里可以实现一些基本的清洗逻辑，例如去除多余的空格、换行符等
        # TODO
        return content.strip()

