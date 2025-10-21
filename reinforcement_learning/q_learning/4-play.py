#!/usr/bin.env python3
"""Importing libraries"""
import numpy as np
import gym


def play(env, Q, max_steps=100):
  """
  Uses Q-table to play an episode
  env: FrozenLakeEnv instance
  Q: numpy.ndarray containing the q-table
  max_steps: integer, maximum number of steps per episode
  returns: the total rewards for the episode
  """
  state = env.reset()
  total_reward = 0
  for step in range(max_steps):
    action = np.argmax(Q[state])
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    if done:
      break
    state = next_state
  return total_reward
