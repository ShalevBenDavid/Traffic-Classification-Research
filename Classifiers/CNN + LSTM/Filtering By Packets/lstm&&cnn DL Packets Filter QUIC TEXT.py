import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Set the folder path where session files are located
folder_path = "/home/shalev/Public/pretraining"
all_sessions = []
labels = []
PACKETS = 30 # Number of packets to read from each session file

# Load data from each label directory and file
for label in os.listdir(folder_path):
    label_folder_path = os.path.join(folder_path, label)

    if os.path.isdir(label_folder_path):
        for filename in os.listdir(label_folder_path):
            file_path = os.path.join(label_folder_path, filename)

             # Read each session file (only first defined packets)
            session_df = pd.read_csv(file_path, delimiter='\t', names=['Timestamp', 'Time Difference', 'Packet Size', 'Direction'], nrows=PACKETS)

            if not session_df.empty:
                # Select only DL features
                session_df = session_df[['Packet Size', 'Direction']]
                all_sessions.append(session_df)
                labels.append(label)

# Concatenate all session data for normalization
all_data = pd.concat(all_sessions, ignore_index=True)

# Scale 'Packet Size' and 'Direction' features using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(all_data[['Packet Size', 'Direction']])

# Normalize each session individually
for i in range(len(all_sessions)):
    all_sessions[i][['Packet Size', 'Direction']] = scaler.transform(all_sessions[i][['Packet Size', 'Direction']])
sequences = [session[['Packet Size', 'Direction']].values for session in all_sessions]

# Convert session data to sequences
sequences = [session[['Packet Size', 'Direction']].values for session in all_sessions]
# Pad sequences for uniform length
sequences_padded = pad_sequences(sequences, dtype='float32')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels, test_size=0.3, random_state=42)

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

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define early stopping criteria to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train_le, epochs=1000, batch_size=32, validation_data=(X_test, y_test_le), callbacks=[early_stopping])

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test_le)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred.argmax(axis=1))

# Print classification report
print(classification_report(y_test, y_pred_labels))