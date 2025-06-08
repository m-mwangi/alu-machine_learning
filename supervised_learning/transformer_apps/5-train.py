#!/usr/bin.env python3
"""Importing necessary libraries"""
import tensorflow.compat.v2 as tf


Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """
    Creates and trains a transformer model for machine translation of Portuguese to English.

    Args:
        N: The number of blocks in the encoder and decoder.
        dm: The dimensionality of the model.
        h: The number of heads.
        hidden: The number of hidden units in the fully connected layers.
        max_len: The maximum number of tokens per sequence.
        batch_size: The batch size for training.
        epochs: The number of epochs to train for.

    Returns:
        The trained transformer model.
    """

    # Create the dataset
    data = Dataset(batch_size, max_len)

    # Define the learning rate schedule
    def learning_rate_schedule(step_num):
        """Learning rate schedule."""
        warmup_steps = 4000
        if step_num < warmup_steps:
            return step_num / (warmup_steps * tf.math.sqrt(tf.cast(warmup_steps, tf.float32)))
        else:
            return tf.math.pow(step_num, -0.5) * tf.math.pow(warmup_steps, 0.5)

    # Create the optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_schedule, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    # Create the loss function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    def loss_function(real, pred):
        """Loss function."""
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        """Accuracy function."""
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    # Create the transformer model
    transformer = Transformer(N, dm, h, hidden, data.tokenizer_pt.vocab_size, data.tokenizer_en.vocab_size, max_len)

    # Train the model
    for epoch in range(epochs):
        batch_num = 1
        for (batch, (inputs, target)) in enumerate(data.data_train):
            # Create masks
            encoder_mask, combined_mask, decoder_mask = create_masks(inputs, target)

            with tf.GradientTape() as tape:
                # Forward pass
                predictions, _ = transformer(inputs, target, training=True, encoder_mask=encoder_mask, combined_mask=combined_mask, decoder_mask=decoder_mask)

                # Calculate loss
                loss = loss_function(target, predictions)

            # Calculate gradients
            gradients = tape.gradient(loss, transformer.trainable_variables)

            # Apply gradients
            optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

            # Print training information every 50 batches
            if batch_num % 50 == 0:
                accuracy = accuracy_function(target, predictions)
                print(f"Epoch {epoch + 1}, batch {batch_num}: loss {loss.numpy():.4f} accuracy {accuracy.numpy():.4f}")

            batch_num += 1

        # Print training information for the epoch
        accuracy = accuracy_function(target, predictions)
        print(f"Epoch {epoch + 1}: loss {loss.numpy():.4f} accuracy {accuracy.numpy():.4f}")

    return transformer
