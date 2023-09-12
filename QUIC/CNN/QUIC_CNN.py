import pandas as pd
import csv
import os
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

folder_path = "/home/shalev/Public/pretraining"
all_sessions = []
labels = []
for label in os.listdir(folder_path):
    label_folder_path = os.path.join(folder_path, label)
    if os.path.isdir(label_folder_path):
        for filename in os.listdir(label_folder_path):
            file_path = os.path.join(label_folder_path, filename)

            session_df = pd.read_csv(file_path, sep="\t", header=None, skiprows=4, nrows=150)
            session_df.columns = ['Timestamp', 'Time Difference', 'Packet Size', 'Direction']

            if not session_df.empty:
                all_sessions.append(session_df)
                labels.append(label)

all_data = pd.concat(all_sessions, ignore_index=True)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(all_data[['Time Difference', 'Packet Size', 'Direction']])

for i in range(len(all_sessions)):
    all_sessions[i][['Time Difference', 'Packet Size', 'Direction']] = scaler.transform(all_sessions[i][['Time Difference', 'Packet Size', 'Direction']])
sequences = [session[['Time Difference', 'Packet Size', 'Direction']].values for session in all_sessions]
from tensorflow.keras.preprocessing.sequence import pad_sequences
sequences_padded = pad_sequences(sequences, dtype='float32')
import numpy as np
labels = np.array(labels)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D,Flatten,MaxPooling1D,LSTM

X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels, test_size=0.3, random_state=42)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y_train)

y_train_le = le.transform(y_train)
y_test_le = le.transform(y_test)


model = Sequential()
model.add(Conv1D(32, kernel_size=3, strides=1, activation='selu', input_shape=X_train[0].shape))
model.add(MaxPooling1D())
model.add(Conv1D(32,kernel_size=3, strides=1, activation='selu'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train_le, epochs=50, batch_size=32, validation_data=(X_test, y_test_le),callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test_le)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)