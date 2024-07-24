class GraphTraversal:
    def __init__(self):
        self.graph = {}
        self.visited = {}
        self.queue = []
        self.stack = []
        self.path = []
        self.cycle = False

    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)

    def bfs(self, node):
        self.queue.append(node)
        self.visited[node] = True
        while self.queue:
            node = self.queue.pop(0)
            self.path.append(node)
            for neighbor in self.graph[node]:
                if neighbor not in self.visited:
                    self.queue.append(neighbor)
                    self.visited[neighbor] = True

    def dfs(self, node):
        self.stack.append(node)
        self.visited[node] = True
        for neighbor in self.graph[node]:
            if neighbor not in self.visited:
                self.dfs(neighbor)
            elif neighbor in self.stack:
                self.cycle = True
        self.stack.pop()
        self.path.append(node)

def main():
    g = GraphTraversal()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(2, 3)
    g.add_edge(3, 3)

    g.bfs(2)
    print("BFS:", g.path)

    g.path = []
    g.visited = {}
    g.dfs(2)
    print("DFS:", g.path)

if __name__ == "__main__":
    main()