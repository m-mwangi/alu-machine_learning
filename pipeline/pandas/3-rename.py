#!/usr/bin/env python3
"""Renaming columns"""
import pandas as pd


from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# rename the column Timestamp to Datetime
df = df.rename(columns={'Timestamp': 'Datetime'})

# convert the timestamp to datetime
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# display only the Datetime and Close columns
df = df[['Datetime', 'Close']]
print(df.head())