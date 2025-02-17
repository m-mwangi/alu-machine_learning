#!/usr/bin/env python3
"""Importing pandas"""
import pandas as pd


df = pd.DataFrame(
    {
        "First": pd.Categorical([0.0, 0.5, 1.0, 1.5]),
        "Second": pd.Categorical(["one", "two", "three", "four"]),
    },
    index = ['A', 'B', 'C', 'D']
)