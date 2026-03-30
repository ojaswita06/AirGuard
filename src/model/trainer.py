"""
trainer.py
==========
Handles the full training loop with professional Keras callbacks:

  EarlyStopping:
    Monitors val_loss. Stops training if it doesn't improve for PATIENCE epochs.
    Restores the best weights automatically — so you always get the best model.

  ModelCheckpoint:
    Saves the model to disk whenever val_loss improves.
    Protects against crashes mid-training.

  ReduceLROnPlateau:
    Halves the learning rate if val_loss stagnates for 4 epochs.
    Helps the model fine-tune after large initial improvements.
"""

import os
import tensorflow as tf
import keras
from keras import layers, models, optimizers, callbacks
from config import (
    EPOCHS, BATCH_SIZE, PATIENCE,
    MODEL_PATH, MODEL_DIR, VAL_SPLIT
)


def train_model(model, X_train, y_train):
    """
    Train the BiLSTM model with early stopping, checkpointing, and LR scheduling.

    Parameters
    ----------
    model   : Compiled tf.keras.Model
    X_train : np.ndarray shape (n_samples, window, features)
    y_train : np.ndarray shape (n_samples,)

    Returns
    -------
    history : Keras History object (contains loss curves)
    model   : Best model (weights restored by EarlyStopping)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=cb_list,
        verbose=1
    )

    print(f"\n[trainer] Best model saved to: {MODEL_PATH}")
    return history, model
