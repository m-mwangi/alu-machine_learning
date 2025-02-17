#!/usr/bin/env python3
"""Importing pandas"""
import pandas as pd


def from_file(filename, delimeter):
    """
    Function that loads data from a file
    as pd.DataFRame
    """
    df = pd.read_csv(filename, delimeter)
    return df