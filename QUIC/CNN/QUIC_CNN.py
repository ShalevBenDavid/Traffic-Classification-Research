import pandas as pd
import csv
import os
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

folder_path = "/home/shalev/Public/pretraining"
all_sessions = []
labels = []
# Going over all sessions in the QUIC data-set.
for label in os.listdir(folder_path):
    label_folder_path = os.path.join(folder_path, label)
    if os.path.isdir(label_folder_path):
        for filename in os.listdir(label_folder_path):
            file_path = os.path.join(label_folder_path, filename)
            # Converting each session to Panda's DataFrame and restricting the number of rows. 
            session_df = pd.read_csv(file_path, sep="\t", header=None, skiprows=4, nrows=150)
            # Giving names to the columns (features).
            session_df.columns = ['Timestamp', 'Time Difference', 'Packet Size', 'Direction']
            # Adding to the collection.
            if not session_df.empty:
                all_sessions.append(session_df)
                labels.append(label)
# Merge all DataFrames to one DataFrame.
all_data = pd.concat(all_sessions, ignore_index=True)
# Normalizing the features to the range (0,1).
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(all_data[['Time Difference', 'Packet Size', 'Direction']])

for i in range(len(all_sessions)):
    all_sessions[i][['Time Difference', 'Packet Size', 'Direction']] = scaler.transform(all_sessions[i][['Time Difference', 'Packet Size', 'Direction']])
# Creating a list of the time series.
sequences = [session[['Time Difference', 'Packet Size', 'Direction']].values for session in all_sessions]
from tensorflow.keras.preprocessing.sequence import pad_sequences
sequences_padded = pad_sequences(sequences, dtype='float32')
import numpy as np
labels = np.array(labels)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D,Flatten,MaxPooling1D,LSTM
# Splitting the data to 30% Test, 70% Train.
X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels, test_size=0.3, random_state=42)

from sklearn.preprocessing import LabelEncoder
# Encoding the labels.
le = LabelEncoder()
le.fit(y_train)
# Converting all the labels to numerical values.
y_train_le = le.transform(y_train)
y_test_le = le.transform(y_test)

# Initializing the model.
model = Sequential()
# Adding layer to the model.
model.add(Conv1D(32, kernel_size=3, strides=1, activation='selu', input_shape=X_train[0].shape))
model.add(MaxPooling1D())
model.add(Conv1D(32,kernel_size=3, strides=1, activation='selu'))
model.add(MaxPooling1D())
# Adding a layer to conert the results of the previous layers to a 1 dimensional vector.
model.add(Flatten())
# Adding a final layer for the classification.
model.add(Dense(5, activation='softmax'))
# Compiling the model.
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# Training the model.
model.fit(X_train, y_train_le, epochs=50, batch_size=32, validation_data=(X_test, y_test_le),callbacks=[early_stopping])
# Computing and outputting the results.
loss, accuracy = model.evaluate(X_test, y_test_le)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)