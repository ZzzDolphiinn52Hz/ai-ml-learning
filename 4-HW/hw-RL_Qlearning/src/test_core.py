"""Unit tests for environment and Bellman update (pytest optional)."""

import numpy as np
from gridworld import GridWorld
from qlearning import QLearningAgent


def test_gridworld_wall_and_goal():
    env = GridWorld()
    # From start (0,0), UP is outside map; agent stays and gets -10.
    state, reward, done = env.step(env.start, 0)
    assert (state, reward, done) == (env.start, -10.0, False)
    # The goal is (7,7): coming from (7,6), RIGHT is terminal +100.
    state, reward, done = env.step(env.encode(7, 6), 1)
    assert (state, reward, done) == (env.goal, 100.0, True)


def test_q_bellman_update_and_terminal():
    env = GridWorld()
    agent = QLearningAgent(env, alpha=0.5, gamma=0.9)
    s, next_s = env.start, env.encode(0,1)
    agent.Q[next_s, 1] = 10
    agent.update(s, 1, -1, next_s, done=False)
    assert np.isclose(agent.Q[s, 1], 4.0)  # 0 + .5*(-1+.9*10)
    goal_neighbor = env.encode(7, 6)
    agent.update(goal_neighbor, 1, 100, env.goal, done=True)
    assert np.isclose(agent.Q[goal_neighbor, 1], 50.0)  # no bootstrap on terminal


def test_bfs_path_length():
    env = GridWorld()
    assert len(env.shortest_path()) - 1 == 14
