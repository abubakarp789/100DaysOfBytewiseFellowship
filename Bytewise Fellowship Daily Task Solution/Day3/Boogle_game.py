class BoggleGame:
    def __init__(self, board, words):
        self.board = board
        self.words = words
        self.visited = [[False for _ in range(len(board[0]))] for _ in range(len(board))]
        self.result = set()

    def solve(self):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                self.dfs(i, j, "")

    def dfs(self, i, j, current_word):
        if i < 0 or i >= len(self.board) or j < 0 or j >= len(self.board[0]) or self.visited[i][j]:
            return

        current_word += self.board[i][j]
        self.visited[i][j] = True

        if current_word in self.words:
            self.result.add(current_word)

        for x in range(-1, 2):
            for y in range(-1, 2):
                self.dfs(i + x, j + y, current_word)

        self.visited[i][j] = False

    def print_result(self):
        for word in self.result:
            print(word)


# Example usage
board = [
    ['A', 'B', 'U'],
    ['B', 'A', 'K'],
    ['A', 'R', 'I']
]

words = ['ABU', 'ABUBAKAR', 'BAKAR', 'BAKI']

game = BoggleGame(board, words)
game.solve()
game.print_result()