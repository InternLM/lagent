from lagent.agents.graph_rag_agent import GraphRagAgent
from lagent.agents.naive_rag_agent import NaiveRAGAgent
from lagent.rag.schema import MultiLayerGraph
from lagent.rag.processors import BuildDatabase, SaveGraph
from lagent.rag.doc import Storage


def test_graph():
    agent = GraphRagAgent(processors_config='your_path')
    agent.init_external_memory(data=['your_path'])

    query = 'What innovations credited to the Sumerians helped define early civilization in Mesopotamia?'
    result = agent.forward(query=query)
    print(result)

