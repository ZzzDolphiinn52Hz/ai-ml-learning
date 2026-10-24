"""A deterministic GridWorld: 0=up, 1=right, 2=down, 3=left."""

from collections import deque

MAP = (
    ("S", ".", ".", "#", ".", ".", ".", "."),
    (".", "#", ".", "#", ".", "#", ".", "."),
    (".", "#", ".", ".", ".", "#", ".", "."),
    (".", ".", "#", "#", ".", ".", ".", "."),
    ("#", ".", ".", ".", ".", "#", ".", "."),
    (".", ".", "#", ".", ".", ".", ".", "."),
    (".", ".", ".", "#", "#", ".", "#", "."),
    (".", ".", ".", ".", ".", ".", ".", "G"),
)

ACTIONS = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1)
)
ACTION_NAMES = ("UP", "RIGHT", "DOWN", "LEFT")


class GridWorld:
    def __init__(self, layout=MAP):
        self.layout = tuple(layout)
        self.height = len(self.layout)
        self.width = len(self.layout[0])
        assert all(len(row) == self.width for row in self.layout)
        assert sum(row.count("S") for row in self.layout) == 1
        assert sum(row.count("G") for row in self.layout) == 1
        self.start = self.find_position("S")
        self.goal = self.find_position("G")
        self.num_states = self.height * self.width
        self.num_actions = len(ACTIONS)
        self.free_states = []
        for row in range(self.height):
            for col in range(self.width):
                if self.layout[row][col] not in ("#", "G"):
                    self.free_states.append(self.encode(row, col))

    def find_position(self, symbol):
        """Tìm vị trí của S hoặc G và đổi thành state number."""
        for row in range(self.height):
            for col in range(self.width):
                if self.layout[row][col] == symbol:
                    return self.encode(row, col)
        raise ValueError(f"Không tìm thấy ký tự {symbol} trong bản đồ")

    def encode(self, row, col):
        """Đổi tọa độ (row, col) thành một state number."""
        return row * self.width + col

    def decode(self, state):
        """Đổi state number thành tọa độ (row, col)."""
        return divmod(int(state), self.width)

    def is_wall(self, row, col):
        """Kiểm tra một ô có phải tường hoặc nằm ngoài bản đồ không."""
        return not (0 <= row < self.height and 0 <= col < self.width) or self.layout[row][col] == "#"

    def step(self, state, action):
        """Thực hiện một bước và trả về state mới, reward, done."""
        assert 0 <= int(action) < self.num_actions
        if int(state) == self.goal:
            return self.goal, 0.0, True
        row, col = self.decode(state)
        dr, dc = ACTIONS[int(action)]
        nr, nc = row + dr, col + dc
        if self.is_wall(nr, nc):
            return int(state), -10.0, False
        next_state = self.encode(nr, nc)
        if next_state == self.goal:
            return next_state, 100.0, True
        return next_state, -1.0, False

    def random_start(self, rng):
        return int(rng.choice(self.free_states))

    def shortest_path(self):
        """BFS is only for validation/reporting, NOT for training the agent."""
        queue = deque([(self.start, [self.start])])
        seen = {self.start}
        while queue:
            state, path = queue.popleft()
            if state == self.goal:
                return path
            for action in range(self.num_actions):
                next_state, _, _ = self.step(state, action)
                if next_state not in seen:
                    seen.add(next_state)
                    queue.append((next_state, path + [next_state]))
        return []
