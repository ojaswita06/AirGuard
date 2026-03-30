"""
architecture.py
===============
Defines the Bidirectional LSTM model.

Why Bidirectional?
  A regular LSTM reads the sequence left→right (past→future).
  A Bidirectional LSTM adds a second LSTM reading right→left,
  then concatenates both hidden states. This captures patterns
  that are clearer in reverse (e.g., AQI dips before a peak).

Architecture:
  Input (30, n_features)
      ↓
  BiLSTM(128) + Dropout(0.3)   ← returns full sequence for next BiLSTM
      ↓
  BiLSTM(64)  + Dropout(0.3)   ← returns only final timestep
      ↓
  Dense(32, relu)
      ↓
  Dense(1, linear)              ← regression output: next-day AQI
"""

import tensorflow as tf
import keras
from keras import layers, models, optimizers
from config import (
    LSTM_UNITS_1, LSTM_UNITS_2,
    DENSE_UNITS, DROPOUT_RATE, LEARNING_RATE
)


def build_bilstm_model(input_shape: tuple) -> tf.keras.Model:
    """
    Build and compile the BiLSTM regression model.

    Parameters
    ----------
    input_shape : (WINDOW_SIZE, n_features) — shape of one training sample

    Returns
    -------
    Compiled tf.keras.Model ready for training
    """
    inp = layers.Input(shape=input_shape, name="sequence_input")

    # First BiLSTM — return_sequences=True so the next BiLSTM gets a full sequence
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_1, return_sequences=True),
        name="bilstm_1"
    )(inp)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_1")(x)

    # Second BiLSTM — return_sequences=False, output is a single vector
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS_2, return_sequences=False),
        name="bilstm_2"
    )(x)
    x = layers.Dropout(DROPOUT_RATE, name="dropout_2")(x)

    # Fully connected layers for regression
    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense_1")(x)
    out = layers.Dense(1, activation="linear", name="aqi_output")(x)

    model = models.Model(inputs=inp, outputs=out, name="AirGuard_BiLSTM")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
        metrics=["mae"]
    )

    model.summary()
    return model
