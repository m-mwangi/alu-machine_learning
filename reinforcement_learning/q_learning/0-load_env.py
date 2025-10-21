#!/usr/bin/env python3
"""Loading libraries"""
import numpy as np
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads frozenlakeEnv from gym
    """
    return gym.make('FrozenLake-v0', desc=desc,
                    map_name=map_name, is_slippery=is_slippery)
