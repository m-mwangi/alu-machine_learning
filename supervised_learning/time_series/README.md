# Time Series Forecasting
Time Series Forecasting is a statistical technique used to predict future values of a time series based on past observations. In simpler terms, it's like looking into the future of data points plotted over time.

## Task Involved
Given the coinbase and bitstamp datasets, a model is built that uses the past 24 hours of BTC data to predict the value of BTC at the close of the following hour (approximately how long the average transaction takes).

## Preprocessing steps
- Changing time window from 60s to 1 hr.

- Rescaling.

- Normalisation

The model uses RNN architecture(simpleRNN), MSE as cost function and uses tf.data.Dataset to feed data into the model.
