#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
# remove weighted_price column
df = df.drop(columns=['Weighted_Price'])
# Rename the Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})
# convert the timestamp to datetime
df['Date'] = pd.to_datetime(df['Date'], unit='s')
# Index the pd.DataFrame on the date values
df = df.set_index('Date')
# missing values in Close should be set to the previous row value
df['Close'] = df['Close'].fillna(method='ffill')
# missing values in High, Low, Open should be set to the previous row value
df['High'] = df['High'].fillna(df['Close'])
df['Low'] = df['Low'].fillna(df['Close'])
df['Open'] = df['Open'].fillna(df['Close'])
# missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
# plot the data from 2017 and beyond at daily intervals
# group the data by day and plot such that:
# High - max
# Low - min
# Close - mean
# Volume_(BTC) - sum
# Volume_(Currency) - sum
df['High'].plot()
df['Low'].plot()
df['Close'].plot()
df['Volume_(BTC)'].plot()
df['Volume_(Currency)'].plot()