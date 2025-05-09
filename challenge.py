import heapq

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
                new_dist = dist + weight
                new_edges = edges_used + 1             
                # if a neighbor is not in the adjacency list,
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

if __name__ == "__main__":
    """
    Run dijkstras
    """
    # create our graph and add the edges to the adjacency list
    g = Graph()
    g.add_edge('A', 'B', 1)
    g.add_edge('A', 'E', 2)
    g.add_edge('B', 'C', 4)
    g.add_edge('B', 'D', 2)
    g.add_edge('C', 'D', 1)
    g.add_edge('C', 'F', 2)
    g.add_edge('D', 'F', 3)
    g.add_edge('E', 'C', 1)
    g.add_edge('E', 'G', 4)
    g.add_edge('G', 'F', 1)
    # i is used to prevent duplicate node pairs
    i=1
    # for each node, run the algorithm for all possible pairs
    for start in list(g.adj_list.keys())[0:len(g.adj_list)-1]:
        # start at one to prevent self pairings
        # the inner loop needs to shift the starting index to be start+1
        for end in list(g.adj_list.keys())[i:len(g.adj_list)-1]:
            k=1
            dist, path = g.dijsktras_with_k_edges(start, end, k)
            
            if dist == float('inf'):
                print(f"No path from {start} to {end} using at most {k} edges")
            else:
                print(f"Shortest path from {start} to {end} using at most {k} edges:")
                print(f"  Distance: {dist}")
                print(f"  Path: {' -> '.join(path)}")
            print()
        i+=1
