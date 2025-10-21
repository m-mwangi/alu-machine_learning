#!/usr/bin/env python3
"""Importing libraries"""
import numpy as np
import gym


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1,
          epsilon_decay=0.05):
    """
    Q-learning training algorithm
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = epsilon_greedy(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Assign penalty for falling in a hole
            if done and reward == 0:
                reward = -1

            # Q-table update
            Q[state, action] += alpha * (reward +
                                         gamma *np.max(Q[next_state]) -
                                         Q[state, action])

            state = next_state
            episode_reward += reward

            if done:
                break

        total_rewards.append(episode_reward)
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay))

    return Q, total_rewards
