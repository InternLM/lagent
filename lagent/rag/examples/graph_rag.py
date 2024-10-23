from lagent.agents.graph_rag_agent import GraphRagAgent


def main():

    agent = GraphRagAgent()
    agent.init_external_memory(data=['./test_file1.txt'])

    query = 'What innovations credited to the Sumerians helped define early civilization in Mesopotamia?'
    response = agent.forward(query=query)
    print(response)


if __name__ == '__main__':
    main()
