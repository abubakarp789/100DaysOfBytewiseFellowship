import sys

class GraphTraversal:
    def __init__(self):
        self.graph = {}
        self.visited = {}
        self.distance = {}
        self.previous = {}

    def add_edge(self, u, v, weight):
        if u not in self.graph:
            self.graph[u] = {}
        self.graph[u][v] = weight

    def dijkstra(self, start):
        # Initialize distance dictionary with infinity for all nodes
        for node in self.graph.keys():
            self.distance[node] = float('inf')

        self.distance[start] = 0
        self.previous[start] = None

        unvisited = set(self.graph.keys())

        while unvisited:
            current = None
            for node in unvisited:
                if current is None or self.distance[node] < self.distance[current]:
                    current = node

            unvisited.remove(current)

            for neighbor, weight in self.graph[current].items():
                new_distance = self.distance[current] + weight
                if neighbor not in self.distance or new_distance < self.distance[neighbor]:
                    self.distance[neighbor] = new_distance
                    self.previous[neighbor] = current

    def get_shortest_path(self, target):
        path = []
        current = target
        while current is not None:
            path.insert(0, current)
            current = self.previous[current]
        return path

def main():
    g = GraphTraversal()
    g.add_edge('A', 'B', 1)
    g.add_edge('A', 'C', 4)
    g.add_edge('B', 'C', 2)
    g.add_edge('B', 'D', 5)
    g.add_edge('C', 'D', 1)

    start_node = 'A'
    g.dijkstra(start_node)

    shortest_paths = {}
    for node in g.graph.keys():
        shortest_paths[node] = g.distance.get(node, sys.maxsize)

    print("Shortest Paths:", shortest_paths)

if __name__ == "__main__":
    main()