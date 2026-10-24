"""Headless validation; does not require pygame."""
from gridworld import GridWorld
from qlearning import QLearningAgent


def verify(seed=42):
    env = GridWorld()
    agent = QLearningAgent(env, seed=seed)
    for _ in range(5000):
        agent.train_episode()
    path, success = agent.greedy_path()
    coordinates = [env.decode(s) for s in path]
    bfs_steps = len(env.shortest_path()) - 1
    print('Seed:', seed)
    print('Success:', success)
    print('Learned path:', coordinates)
    print('Steps:', len(path) - 1)
    print('BFS shortest steps:', bfs_steps)
    print('Last 100 episode success:', sum(x.success for x in agent.history[-100:]))
    assert success, 'The agent failed to reach the goal.'
    assert len(path)-1 == bfs_steps, 'The learned route is not shortest.'


if __name__ == '__main__':
    verify()
