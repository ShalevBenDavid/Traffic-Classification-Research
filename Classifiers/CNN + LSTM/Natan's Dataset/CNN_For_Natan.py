import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Conv1D, AveragePooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pyarrow.parquet as pq

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

# The CNN Model
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),  # Input shape depends on the data
    AveragePooling1D(pool_size=2),
    Dropout(0.25),
    Conv1D(filters=128, kernel_size=3, activation='relu'),
    AveragePooling1D(pool_size=2),
    Dropout(0.25),
    Flatten(),
    Dense(np.unique(y_train).size, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model.fit(X_train[..., np.newaxis], y_train, epochs=200, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

print("Evaluating model...")
y_pred = model.predict(X_test[..., np.newaxis])
y_pred_classes = np.argmax(y_pred, axis=1)
report = classification_report(y_test, y_pred_classes, target_names=class_labels)

print(report)