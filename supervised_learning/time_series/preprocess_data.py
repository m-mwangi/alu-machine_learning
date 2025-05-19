def preprocess_data(filepath, seq_len=24):
    """
    Preprocess  BTC datasets for RNN training.
    :param seq_len: Number of past hours to use for prediction.
    :return: Processed data and labels.
    """
    # Load and concatenate datasets
    dataframes = [pd.read_csv(filepath)]
    data = pd.concat(dataframes)
    # Handle missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    # Convert timestamp to datetime
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
    # Set timestamp as index
    data.set_index('Timestamp', inplace=True)

    # Resample data into hourly intervals
    data = data.resample('1H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume_(BTC)': 'sum',
        'Volume_(Currency)': 'sum',
        'Weighted_Price': 'mean'
    })
    # Normalize features
    scaler = (data - data.min()) / (data.max() - data.min())

    # Create sequences and labels
    sequences = []
    labels = []
    for i in range(len(scaler) - seq_len):
        sequences.append(scaler.iloc[i:i + seq_len].values)
        labels.append(scaler.iloc[i + seq_len]['Close'])

    return np.array(sequences), np.array(labels)
