import os
import csv
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Constants and initialize variables
folder_path = "/home/shalev/Public/pretraining"
all_sessions = []
labels = []
TIME = 1  # Time threshold for flow segmentation

# Load and preprocess data from each label directory and file
for label in os.listdir(folder_path):
    label_folder_path = os.path.join(folder_path, label)

    if os.path.isdir(label_folder_path):
        for filename in os.listdir(label_folder_path):
            file_path = os.path.join(label_folder_path, filename)
            with open(file_path, 'r') as csvfile:
                csvreader = csv.reader(csvfile, delimiter='\t')
                session_data = []
                for row in csvreader:
                    try:
                        time_diff = float(row[1])
                        if time_diff >= TIME:
                            continue

                    except ValueError:
                        continue

                    # Append packet only if within the time block
                    session_data.append(row)

                if time_diff >= TIME and session_data:
                    session_df = pd.DataFrame(session_data, columns=['Timestamp', 'Time Difference', 'Packet Size', 'Direction'])
                    all_sessions.append(session_df)
                    labels.append(label)

# Concatenate all session data for normalization
all_data = pd.concat(all_sessions, ignore_index=True)

# Scale 'Time Difference', 'Packet Size', and 'Direction' features using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(all_data[['Time Difference', 'Packet Size', 'Direction']])

# Normalize each session individually
for i in range(len(all_sessions)):
    all_sessions[i][['Time Difference', 'Packet Size', 'Direction']] = scaler.transform(all_sessions[i][['Time Difference', 'Packet Size', 'Direction']])

# Convert session data to sequences
sequences = [session[['Time Difference', 'Packet Size', 'Direction']].values for session in all_sessions]
# Pad sequences for uniform length
sequences_padded = pad_sequences(sequences, dtype='float32')

# Split data into training and test sets while preserving the class distribution (using stratify)
X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels, test_size=0.3, random_state=42, stratify=labels)

# Encode labels for classification
le = LabelEncoder()
le.fit(y_train)
y_train_le = le.transform(y_train)
y_test_le = le.transform(y_test)

# Build the CNN/LSTM model
model = Sequential([
    Conv1D(32, kernel_size=3, strides=1, activation='selu', input_shape=X_train[0].shape),
    MaxPooling1D(),
    Conv1D(32,kernel_size=3, strides=1, activation='selu'),
    MaxPooling1D(),
    Flatten(),
    #LSTM(32, input_shape=X_test[0].shape),
    Dense(5, activation='softmax')
])

# Compile the model with sparse categorical cross-entropy loss and Adam optimizer
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping criteria to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train_le, epochs=1000, batch_size=32, validation_data=(X_test, y_test_le),callbacks=[early_stopping])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test_le)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred.argmax(axis=1))  # Convert predicted labels back to original labels

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_labels))