#!/usr/bin/env python3
"""Importing libraries"""
import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
  """
  Uses epsilon-greedy to determine the
  next action
  Q: numpy.ndarray containing the q-table
  state: integer, current state
  epsilon: float, the epsilon to use for the
  exploration function
  Sample p to determibe explore or exploit
  If exploring, pick next action randomly
  If exploiting, pick next action using Q
  """
  p = np.random.uniform(0, 1)
  if p < epsilon:
    return np.random.randint(Q.shape[1])
  return np.argmax(Q[state])
