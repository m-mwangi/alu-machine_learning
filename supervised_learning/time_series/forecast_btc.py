import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import numpy as np

# Load preprocessed data
preprocessed = np.load('preprocessed_data.npz')
sequences, labels = preprocessed['sequences'], preprocessed['labels']

# Split data into train and test sets
split_idx = int(0.8 * len(sequences))
X_train, X_test = sequences[:split_idx], sequences[split_idx:]
y_train, y_test = labels[:split_idx], labels[split_idx:]


# Create tf.data.Dataset
def create_dataset(X, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    return dataset.shuffle(len(X)).batch(batch_size).
    prefetch(tf.data.experimental.AUTOTUNE)


train_dataset = create_dataset(X_train, y_train)
test_dataset = create_dataset(X_test, y_test)


# Build RNN model
def create_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)  # Predicting close price
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Train model
model = create_model(X_train.shape[1:])
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Save model
model.save('btc_forecasting_model.h5')

# Evaluate model
eval_metrics = model.evaluate(test_dataset)
print(f"Test Loss: {eval_metrics[0]}, Test MAE: {eval_metrics[1]}")
