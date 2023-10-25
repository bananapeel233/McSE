import numpy as np

class unionSet(object):
    def __init__(self, graph):
        self.graph = graph
        self.parent = np.asarray(range(len(graph)))
        self.rank = np.ones(len(graph))
        for i in range(len(graph)):
            for j in range(len(graph[0])):
                if graph[i][j] > 0:
                    self.union(i, j)


    def get_root(self, i):
        while i != self.parent[i]:
            i = self.parent[i]

        return i

    def union(self, i, j):
        i_root = self.get_root(i)
        j_root = self.get_root(j)
        if i_root == j_root:
            return

        if self.rank[i_root] == self.rank[j_root]:
            self.parent[i_root] = j_root
            self.rank[j_root] += 1
        elif self.rank[i_root] > self.rank[j_root]:
            self.parent[j_root] = i_root
        else:
            self.parent[i_root] = j_root

    def is_connected(self, i, j):
        return self.get_root(i) == self.get_root(j)

    def get_subConnectedGraph(self, i):
        nodes = []
        for j in range(len(self.parent)):
            if self.is_connected(i, j):
                nodes.append(j)

        return np.asarray(nodes)

    def visualize(self):
        print("unionSet:", self.parent)

