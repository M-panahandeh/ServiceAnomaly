import netgraph as netgraph
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import networkx as nx
# class definition
from Tools.scripts.dutree import display


class DAG:
    edges = []
    #labels=[]
    G = nx.DiGraph()

    def GrphGeneration(self,listOfRelations):
        self.G.clear()
        self.edges=[]
        for i in listOfRelations:
            edge = []
            edge += [i['parent']]
            edge += [i['child']]
            self.edges.append(tuple(edge))
            #print(listOfRelations[i]['parent'])
            #generate graph
            self.G.add_edges_from(self.edges)
            #print(self.edges)

        #print(self.G.edges)
        #check if it's DAG or not:
        if not (self.IsDAG()):
               dict= self.Conver2DAG()
               for node in self.G.nodes():
                   self.G.nodes[node]['label'] = dict[node]
               plt.tight_layout()
               nx.draw_networkx(self.G, arrows=True, with_labels=True, labels=dict,edge_color='black', width=1, linewidths=1,node_size=500, node_color='blue')

        else:
                # No cycle
                plt.tight_layout()
                nx.draw_networkx(self.G, arrows=True,edge_color='black', width=1, linewidths=1, node_size=500, node_color='blue')



    def IsDAG(self):
        if (nx.is_directed_acyclic_graph(self.G)):
            return True
        else:
            return False



    def Conver2DAG(self):

                cycle_node = []
                #  Check to see if its not DAG
                self.G.add_edges_from(self.edges)
                # get list of loops
                for cycle in nx.simple_cycles(self.G):
                    cycle_node.append(cycle)
                i = -1
                #list of new altered edges
                DAG_Edges = []
                id_dict = {}
                for elem in self.edges:
                    tup = []
                    if (not elem[0] in id_dict.values()):
                        # print(id_dict.values())
                        # get new value in dictionary
                        i = i + 1
                        id_dict[i] = elem[0]
                        tup.insert(0, i)
                    else:
                        # get key in dic for the tup
                        tup.insert(0, list(id_dict.keys())[list(id_dict.values()).index(elem[0])])
                    if (not elem[1] in id_dict.values() or (elem[1] in cycle and elem[0] in cycle)):
                        # print(id_dict.values())
                        # get new value in dictionary
                        i = i + 1
                        id_dict[i] = elem[1]
                        tup.insert(1, i)
                    else:
                        # get key in dic for the tup
                        tup.insert(1, list(id_dict.keys())[list(id_dict.values()).index(elem[1])])
                    DAG_Edges.append(tuple(tup))
                #Replace edges with new edges
                self.edges = DAG_Edges
                #regenerate graph
                self.G = nx.DiGraph()
                self.G.add_edges_from(self.edges)
                #return dictionary to be used in visualization and labeling nodes
                return id_dict



    def GraphVisualize(self):


                plt.show()



    def GraphEditEdge(self):
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos=pos, edge_color=('black', 'r','black', 'black'),labels={node: node for node in self.G.nodes()})
        plt.axis('off')
        plt.show()

