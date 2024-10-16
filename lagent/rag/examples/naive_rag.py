from lagent.rag.pipeline import BaseProcessor
from lagent.agents.graph_rag_agent import GraphRagAgent
from lagent.agents.naive_rag_agent import NaiveRAGAgent


def main():
    processors_config = {
        'dependencies': {
            'llm': {
                'type': "lagent.llms.deepseek.DeepseekAPI",
                'model_type': "deepseek-chat",
                'key': "your_api_key",
                'max_tokens': 4096
            },
            'embedder': {
                'type': "lagent.rag.nlp.sentence_transformer_embedder.SentenceTransformerEmbedder",
                'model_name': 'all-MiniLM-L6-v2',
                'device': 'cpu',
                'prefix': "",
                'suffix': "",
                'batch_size': 32,
                'normalize_embeddings': True,
                'model_path': 'sentence_transformer_min.pkl'
            },
            'storage': {
                'type': "lagent.rag.doc.storage.Storage"
            },
            'tokenizer': {
                'type': "lagent.rag.nlp.tokenizer.SimpleTokenizer"
            }
        },
        'processors': [
            {
                'type': "lagent.rag.processors.doc_parser.DocParser",
                'params': {
                    'storage': 'storage'
                }
            },
            {
                'type': "lagent.rag.processors.chunk.ChunkSplitter",
                'params': {
                    'cfg': {
                        'chunk_size': 1000,
                        'overlap': 200
                    }
                }
            },
            {
                'type': "lagent.rag.processors.build_db.BuildDatabase",
                'params': {
                    'embedder': 'embedder'
                }
            },
            {
                'type': "lagent.rag.processors.dump_load.SaveGraph",
                'params': {
                    'storage': "storage"
                }
            }
        ]
    }

    agent = NaiveRAGAgent(processors_config=processors_config)
    agent.init_external_memory(data=['your_path'])

    query = 'What innovations credited to the Sumerians helped define early civilization in Mesopotamia?'
    agent.forward(query=query)


if __name__ == '__main__':
    main()
