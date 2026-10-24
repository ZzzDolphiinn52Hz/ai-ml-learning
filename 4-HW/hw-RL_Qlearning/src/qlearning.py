"""Tabular Q-learning with epsilon-greedy exploration."""

import numpy as np


class EpisodeStats:
    """Kết quả của một episode."""

    def __init__(self, steps, reward, success, epsilon):
        self.steps = steps
        self.reward = reward
        self.success = success
        self.epsilon = epsilon

class QLearningAgent:
    def __init__(self, env, alpha=0.15, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.998, max_steps=160,
                 seed=42):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.initial_epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Xóa bảng Q và bắt đầu học lại từ đầu."""
        self.Q = np.zeros((self.env.num_states, self.env.num_actions), dtype=float)
        self.epsilon = self.initial_epsilon
        self.history = []

    def choose_action(self, state, explore=True):
        """Chọn hành động ngẫu nhiên hoặc hành động tốt nhất hiện tại."""
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.env.num_actions))
        values = self.Q[state]
        # Nếu nhiều hành động cùng tốt, chọn ngẫu nhiên giữa chúng.
        best_value = max(values)
        best_actions = [action for action, value in enumerate(values)
                        if value == best_value]
        return int(self.rng.choice(best_actions))

    def update(self, state, action, reward, next_state, done):
        """Cập nhật một ô trong bảng Q theo công thức Bellman."""
        future_value = 0 if done else max(self.Q[next_state])
        target = reward + self.gamma * future_value
        old_value = self.Q[state, action]
        self.Q[state, action] = old_value + self.alpha * (target - old_value)

    def train_episode(self):
        """Cho robot chạy một episode và cập nhật bảng Q."""
        state = self.env.random_start(self.rng)
        total_reward = 0
        success = False
        steps_taken = 0
        for step in range(1, self.max_steps + 1):
            action = self.choose_action(state, explore=True)
            next_state, reward, done = self.env.step(state, action)
            self.update(state, action, reward, next_state, done)
            total_reward += reward
            steps_taken = step
            state = next_state
            if done:
                success = True
                break
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        result = EpisodeStats(steps_taken, total_reward, success, self.epsilon)
        self.history.append(result)
        return result

    def greedy_path(self, start=None, max_steps=120):
        """A deterministic preview of the learned greedy policy; no learning."""
        state = self.env.start if start is None else int(start)
        path = [state]
        visited = {state}
        for _ in range(max_steps):
            if state == self.env.goal:
                return path, True
            action = int(np.argmax(self.Q[state]))
            next_state, _, done = self.env.step(state, action)
            path.append(next_state)
            if done:
                return path, True
            if next_state in visited:
                return path, False
            visited.add(next_state)
            state = next_state
        return path, False
