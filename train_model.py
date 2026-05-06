import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from preprocess import (
    load_csv,
    fit_scaler,
    transform,
    create_sequences
)

os.makedirs("models", exist_ok=True)

TRAIN_FILE = "data/raw/normal.csv"

# -----------------------------------
# Load data
# -----------------------------------

df = load_csv(TRAIN_FILE)

scaler = fit_scaler(df)

scaled = transform(df, scaler)

X, y = create_sequences(scaled)

# -----------------------------------
# Build model
# -----------------------------------

model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(X.shape[1], X.shape[2])
    ),

    Dropout(0.2),

    LSTM(32),

    Dense(16, activation="relu"),

    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# -----------------------------------
# Train
# -----------------------------------

early = EarlyStopping(
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X,
    y,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early]
)

# -----------------------------------
# Save model
# -----------------------------------

model.save("models/lstm_model.keras")

print("\nModel saved.")