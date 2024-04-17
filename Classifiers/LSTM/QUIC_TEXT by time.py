import csv
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,AveragePooling1D,Flatten,Dropout,LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report

from collections import defaultdict

TIME = 7.5
packet_counts_by_class = defaultdict(int)
flow_count_by_class = defaultdict(int)
total_packet_count = 0
total_flow_count = 0
folder_path = "/home/hybrid/Amit/new pretraining"
all_sessions = []
labels = []
flownum = 0  

for label in os.listdir(folder_path):
    label_folder_path = os.path.join(folder_path, label)
    if os.path.isdir(label_folder_path):
        for filename in os.listdir(label_folder_path):
            packet_count = 0
            flow_count = 0
            file_path = os.path.join(label_folder_path, filename)
            with open(file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter='\t')
                session_data = []
                for row in csvreader:
                    try:
                        time_diff = float(row[1])
                        if time_diff <= TIME:
                            packet_count += 1
                            flow_count += 1
                            session_data.append(row)
                            continue

                    except ValueError:
                        continue

                if session_data:
                    session_df = pd.DataFrame(session_data, columns=['timestamp', 'relative_time', 'length', 'direction'])
                    session_df['flownum'] = flownum  
                    session_df['label'] = label  
                    all_sessions.append(session_df)
                    packet_counts_by_class[label] += packet_count
                    total_packet_count += packet_count
                    flownum += 1  

all_data = pd.concat(all_sessions, ignore_index=True)

num_of_packets = 30

# Filter sessions with at least num_of_packets packets
df_filtered = all_data.groupby('flownum').filter(lambda x: len(x) >= num_of_packets)

# Generate sequences and labels
sequences = []
labels = []

for name, group in df_filtered.groupby('flownum'):
    session_data = group[['direction', 'length', 'relative_time']].values[:30]
    sequences.append(session_data)
    labels.append(group['label'].iloc[0])

# Convert lists to arrays
max_length = max(len(seq) for seq in sequences)
padded_sequences = np.array([np.pad(seq, ((0, max_length - len(seq)), (0, 0)), mode='constant') for seq in sequences])
sequences = padded_sequences
labels = np.array(labels)


# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)

# Split the dataset into 70% training and 30% for testing + validation
X_train, X_test_val, y_train, y_test_val = train_test_split(sequences, encoded_labels, test_size=0.2, random_state=42)

# Split the 30% test_val set equally into validation and test sets
X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

# Normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

num_samples, num_time_steps, num_features = X_train.shape


# Define the LSTM model
model_lstm = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(num_time_steps, num_features)),  # First Bidirectional LSTM layer
    Dropout(0.25),  # Dropout for regularization
    LSTM(64),  # Second LSTM layer
    Dense(np.unique(y_train).size, activation='softmax')  # Output layer
])


model_lstm.compile(optimizer=Adam(learning_rate=0.001),
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_lstm.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val), callbacks=[early_stopping])

print("Evaluating model...")
y_pred = model_lstm.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
report = classification_report(y_test, y_pred)

print(report)
