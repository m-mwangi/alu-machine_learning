#!/usr/bin/env python3
"""Importing pandas"""
import pandas as pd


def from_numpy(array):
    """
    A function that creates pd.DataFrame
    from np.ndarray
    array: np.ndarray to create pd.DataFrame
    26 columns labeled in alphabetical order
    and capitalized
    Returns newly created DataFrame
    """
    num_cols = array.shape[1]
    columns = [chr(ord('A') + i) for i in range(num_cols)]
    return pd.DataFrame(array, columns=columns)