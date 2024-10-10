from lagent.rag.pipeline import BaseProcessor
from lagent.agents.graph_rag_agent import GraphRagAgent
from lagent.agents.naive_rag_agent import NaiveRAGAgent


def test_naive():
    agent = NaiveRAGAgent(processors_config='your_path')
    agent.init_external_memory(data=['your_path'])

    query = 'What innovations credited to the Sumerians helped define early civilization in Mesopotamia?'
    agent.forward(query=query)
