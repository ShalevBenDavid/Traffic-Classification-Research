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
sequences = np.array(sequences)
labels = np.array(labels)



# Encode labels
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
X_train, X_test, y_train, y_test = train_test_split(sequences, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42)

# Normalize features based only on the training set
scaler = MinMaxScaler(feature_range=(0, 1))
num_samples, num_time_steps, num_features = X_train.shape
X_train_reshaped = X_train.reshape(-1, num_features)
X_test_reshaped = X_test.reshape(-1, num_features)

# Fit on training set, transform both training and test sets
scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(num_samples, num_time_steps, num_features)
X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape[0], num_time_steps, num_features)

num_samples, num_time_steps, num_features = X_train_scaled.shape

print("Building and training LSTM model...")

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

# Note: Make sure X_train and X_test are scaled and reshaped appropriately before this step
model_lstm.fit(X_train_scaled, y_train, epochs=1000, batch_size=16, validation_split=0.5, callbacks=[early_stopping])

y_pred_lstm = model_lstm.predict(X_test_scaled)
y_pred_lstm = np.argmax(y_pred_lstm, axis=1)
report_lstm = classification_report(y_test, y_pred_lstm)

print(report_lstm)