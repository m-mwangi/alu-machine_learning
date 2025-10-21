#!/usr/bin/env python3
"""Importing libraries"""
import numpy as np
import gym


def q_init(env):
  """
  Initializes the Q table
  """
  return np.zeros((env.observation_space.n,
                   env.action_space.n))
