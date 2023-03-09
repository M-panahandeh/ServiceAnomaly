import networkx as nx

# Define the two graphs as lists of edges
graph1_edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
graph2_edges = [(1, 2), (2, 3), (3, 4), (4, 5), (2, 3), (3, 4)]

# Convert the edge lists to directed graphs
graph1 = nx.DiGraph(graph1_edges)
graph2 = nx.DiGraph(graph2_edges)

# Check if graph2 is an edge-induced subgraph isomorphic to graph1 using the VF2 algorithm
gm = nx.algorithms.isomorphism.DiGraphMatcher(graph1, graph2)
if gm.subgraph_is_isomorphic():
    print("graph2 is an edge-induced subgraph isomorphic to graph1")
else:
    print("graph2 is not an edge-induced subgraph isomorphic to graph1")