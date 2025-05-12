import heapq
import math
from typing import List, Optional

class Graph:
    def __init__(self):
        """
        Need adjacency list for keeping track of neighbors
        """
        self.adj_list = {}

    def add_edge(self, u: str, v: str, weight: int) -> None:
        """
        Add a weighted edge from u to v.
        Use adjacency list to keep track of a node's neighbors.
        Assuming the graph is undirected, each time we add an edge
        we have to make sure both nodes are in the adjacency list
        """
        if u not in self.adj_list:
            self.adj_list[u] = []
        self.adj_list[u].append((v, weight))

        if v not in self.adj_list:
            self.adj_list[v] = []
        self.adj_list[v].append((u, weight))

    def dijsktras_with_k_edges(self, start, end, k):
        """
        Implement dijsktras with an edge limit
        """
        # first, we should make sure we don't look for an impossible path
        if start not in self.adj_list or end not in self.adj_list:
            return float('inf'), []
        # implement a priority queue to keep track of path
        # this allows us to pop the priority node (lowest weight edge)
        # each time we have to make a decision
        pq = [(0, start, 0, [start])]
        heapq.heapify(pq)
        # before we go through the priority queue,
        # initialize the visited list with 0 distance traveled over 0 edges
        visited = {}
        visited[start] = {0: 0}

        while pq:
            # grab our variables from the priority tuple (node) in the queue
            dist, node, edges_used, path = heapq.heappop(pq)
            # if we reach the end node, return the distance and path
            if node == end:
                return dist, path
            # if we've used all k edges, run out the queue
            # this will force the infinite distance to return
            if edges_used >= k:
                continue
            # if we've already found a better path to this node,
            # with the same or fewer edges, skip
            if node in visited and edges_used in visited[node] and dist > visited[node][edges_used]:
                continue
            # explore neighbors and calculate distance
            for neighbor, weight in self.adj_list[node]:
                # skip edges with negative weight
                if weight < 0:
                    continue
                new_dist = dist + weight
                new_edges = edges_used + 1
                # if a neighbor is not in the visited list,
                # add it
                if neighbor not in visited:
                    visited[neighbor] = {}
                # if we haven't explore the path to the neighbor,
                # or if the path to the neighbor is more optimal,
                # we should visit the neighbor and add the path to the pq
                if new_edges not in visited[neighbor] or new_dist < visited[neighbor][new_edges]:
                    visited[neighbor][new_edges] = new_dist
                    new_path = path.copy()
                    new_path.append(neighbor)
                    heapq.heappush(pq, (new_dist, neighbor, new_edges, new_path))
        # if we can't reach the end node using at most k edges,
        # return infinity which is an 'impossible' distance
        return float('inf'), []

    def floyds_with_k_indexes(self, k) -> list[list[float]]:
        """
        Implement Floyd's algorithm to find shortest paths with only k intermediate indexes.
        """
        # we can just use a list of nodes here
        # in the Dijkstras implementation, we used a pq for dfs
        # floyds will create a matrix so we just iterate over all combinations
        nodes = list(self.adj_list.keys())
        n = len(nodes)
        # if k is bigger than n, set k to n
        k = min(k, n)
        # initialize weight matrix
        dist = [[float('inf')] * n for _ in range(n)]
        # all diagonal values should be zero
        # since the path to self is zero
        for i in range(n):
            dist[i][i] = 0
        if k == 0:
            return dist
        # populate the matrix based on the edges in the adjacency list
        # this is our freebie population
        # since we know the path across individual edges
        for start_idx, node in enumerate(nodes):
            for end, weight in self.adj_list.get(node, []):
                end_idx = nodes.index(end)
                dist[start_idx][end_idx] = weight
        # core floyd's implementation
        # intermediate keeps track of how far away we can get
        for intermediate in range(k):
            # create a copy of the matrix to avoid using updates to the original
            new_dist = [row[:] for row in dist]
            for i in range(n):
                for j in range(n):
                    # check if the path through the intermediate vertex is shorter
                    # if so, add it to the matrix
                    if (dist[i][intermediate] != float('inf') and
                        dist[intermediate][j] != float('inf') and
                        dist[i][intermediate] + dist[intermediate][j] < dist[i][j]):
                        new_dist[i][j] = dist[i][intermediate] + dist[intermediate][j]
            dist = new_dist
        return dist

def run_dijkstras(g: Graph, k) -> None:

    i=1
    # for each node, run the algorithm for all possible pairs
    for start in list(g.adj_list.keys())[0:len(g.adj_list)-1]:
        # start at one to prevent self pairings
        # the inner loop needs to shift the starting index to be start+1
        for end in list(g.adj_list.keys())[i:len(g.adj_list)]:
            dist, path = g.dijsktras_with_k_edges(start, end, k)
            if dist == float('inf'):
                print(f"No path from {start} to {end} using at most {k} edges")
            else:
                print(f"Shortest path from {start} to {end} using at most {k} edges:")
                print(f"  Distance: {dist}")
                print(f"  Path: {' -> '.join(path)}")
            print()
        i+=1

def run_floyds(g: Graph, k: int) -> None:
    """
    Runs Floyd's algorithm with a given graph
    Print resulting weight matrix

    Args:
        g (Graph): we are looking for the shortest paths on this graph
        k (int): the number of intermediate vertexes we are allowed to use
    """
    dist_mat = g.floyds_with_k_indexes(k)
    for row in dist_mat:
        print(" ".join(f"{x} " if x != math.inf else "inf" for x in row))


if __name__ == "__main__":
    # create our graph and add the edges to the adjacency list
    g = Graph()
    g.add_edge('A','B', 2)
    g.add_edge('B','C', 1)
    g.add_edge('B','D', 3)
    g.add_edge('C','D', -3)
    g.add_edge('D','E', 2)
    # i is used to prevent duplicate node pairs
    run_dijkstras(g, 0)
    run_floyds(g, 0)
