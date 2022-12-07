import numpy as np
import ast

def ChromosomeToCycle(chromosome):
    nodes = []
    for j in range(len(chromosome)):
        i = chromosome[j]
        if i > 0:
            nodes.append(2*i-1)
            nodes.append(2*i)
        else:
            nodes.append(-2*i)
            nodes.append(-2*i-1)
    return nodes

def ColoredEdges(P):
    Edges = []
    for chromosome in P:
        Nodes = ChromosomeToCycle(chromosome)
        #         print(Nodes)
        for j in range(0, len(chromosome) - 1):
            Edges.append([Nodes[2 * j + 1], Nodes[2 * j + 2]])
        Edges.append([Nodes[-1], Nodes[0]])
    return Edges

def two_BreakOnGenomeGraph(GenomeGraph, i1 , i2 , i3 , i4):
    new_graph = []
    for pair in GenomeGraph:
        if (pair[1] == i1 and pair[0]==i2) or (pair[0] == i1 and pair[1]==i2):
            # check the order
            if pair[0] < pair[1]:
                new_graph.append([np.min([i1, i3]), np.max([i1, i3])])
            else:
                new_graph.append([np.max([i1, i3]), np.min([i1, i3])])
        elif (pair[1] == i3 and pair[0]==i4) or (pair[0] == i3 and pair[1]==i4):
            if pair[0] < pair[1]:
                new_graph.append([np.min([i2, i4]), np.max([i2, i4])])
            else:
                new_graph.append([np.max([i2, i4]), np.min([i2, i4])])
        else:
            new_graph.append(pair)
    return new_graph

def two_BreakOnGenome(P, i1, i2, i3, i4):
    #     seq = ChromosomeToCycle(P)
    #     blackEdges = [[seq[i], seq[i+1]] for i in range(0, len(seq)-1, 2)]
    coloredEdges = ColoredEdges([P])
    print(coloredEdges)
    two_breaks = two_BreakOnGenomeGraph(coloredEdges, i1, i2, i3, i4)
    print(two_breaks)
    P = 0

    component_count = 0
    remaining = [str(element) for element in two_breaks]
    print(remaining)
    print("-------------")
    while remaining:
        component_count += 1
        queue = [ast.literal_eval(remaining.pop())]  # Undirected graph, so we can choose a remaining node arbitrarily.
        for i in range(len(two_breaks)):
            node1 = queue[-1]
            node2 = two_breaks[i]
            if str(node2) in remaining:  # It hasn't been selected
                if np.abs(node1[1] - node2[0]) == 1:
                    queue.append(node2)
            if np.abs(node2[1] - queue[0][0]) == 1:  # Cycle completed
                print(queue)
                break

        remaining -= [str(element) for element in queue]
        print("Remaining", remaining)

    #         while queue:
    #             # Select an element from the queue and get its remaining children.
    #             current = queue.pop()
    #             new_nodes = {node for node in graph[current] if node in remaining}
    #             # Add the new nodes to the queue, remove them from the remaining nodes.
    #             queue |= new_nodes
    #             remaining -= new_nodes
    return P

P = [+1, -2, -4 ,+3]
parameters = [1, 6, 3, 8]
result = two_BreakOnGenome(P, parameters[0], parameters[1], parameters[2], parameters[3])
result