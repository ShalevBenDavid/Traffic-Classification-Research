import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pyarrow.parquet as pq
import sys

# Read the Parquet file into a pandas DataFrame
parquet_file_path = '/home/shalev/Public/QUIC PCAP and FLUSH/Datasets/flash/1091.parquet'
data_frame = pq.read_table(parquet_file_path).to_pandas()

# Extract features and labels from the DataFrame
X = np.array(data_frame.iloc[:, 1].tolist())
y = data_frame.iloc[:, 2].values

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
class_labels = encoder.classes_

# Split the dataset to 80/20 for train/test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Reshape input data for LSTM
X_train_reshaped = X_train[..., np.newaxis]
X_test_reshaped = X_test[..., np.newaxis]

# The LSTM Model
model = Sequential([
    LSTM(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=True),  # LSTM layer
    Dropout(0.25),  # Dropout for regularization
    LSTM(32),  # LSTM layer
    Dense(np.unique(y_train).size, activation='softmax')  # Output layer
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train_reshaped, y_train, epochs=200, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

print("Evaluating model...")
y_pred = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)
report = classification_report(y_test, y_pred_classes, target_names=class_labels)

print(report)
