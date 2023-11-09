import pandas as pd
import numpy as np
import csv
import os
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from collections import defaultdict

TIME = 1
packet_counts_by_class = defaultdict(int) 
flow_count_by_class = defaultdict(int)
total_packet_count = 0 
total_flow_count = 0
packet_count = 0
flow_count = 0
firstTime = False
if firstTime:
    folder_path = "Datasets\QUIC"
    all_sessions = []
    labels = []
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
                            packet_count += 1
                            if time_diff > TIME:
                                flow_count += 1
                                continue

                        except ValueError:
                            continue

                        session_data.append(row)

                    if time_diff >= TIME and session_data:
                        session_df = pd.DataFrame(session_data, columns=['Timestamp', 'Time Difference', 'Packet Size', 'Direction'])
                        all_sessions.append(session_df)
                        labels.append(label)
                        packet_counts_by_class[label] += packet_count
                        total_packet_count += packet_count

    all_data = pd.concat(all_sessions, ignore_index=True)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(all_data[['Time Difference', 'Packet Size', 'Direction']])

    for i in range(len(all_sessions)):
        all_sessions[i][['Time Difference', 'Packet Size', 'Direction']] = scaler.transform(all_sessions[i][['Time Difference', 'Packet Size', 'Direction']])
    sequences = [session[['Time Difference', 'Packet Size', 'Direction']].values for session in all_sessions]

    with open('Datasets\QUIC\QUIC_sequences.pkl', 'wb') as f:
        pickle.dump(sequences, f)
    with open('Datasets\QUIC\QUIC_labels.pkl', 'wb') as f:
        pickle.dump(labels, f)
    # save the packet counts by class to a file
    with open('Datasets\QUIC\QUIC_flow_counts_by_class.pkl', 'wb') as f:
        pickle.dump(flow_count_by_class, f)
else:
    # QUIC_sequences.pkl is compressed to a zip file due to its large size
    import zipfile
    zip_ref = zipfile.ZipFile('Datasets\QUIC\QUIC_sequences.zip', 'r')
    sequences = pickle.load(zip_ref.open('QUIC_sequences.pkl'))
    labels = pickle.load(open('Datasets\QUIC\QUIC_labels.pkl', 'rb'))
    flow_count_by_class = pickle.load(open('Datasets\QUIC\QUIC_flow_counts_by_class.pkl', 'rb'))

for label, flow_count in flow_count_by_class.items():
    print(f"Class '{label}': Number of flows = {flow_count}")
total_flow_count = sum(flow_count_by_class.values())
print(f"Total flows: {total_flow_count} in {len(flow_count_by_class)} classes")

from tensorflow.keras.preprocessing.sequence import pad_sequences
sequences_padded = pad_sequences(sequences, dtype='float32')
labels = np.array(labels)

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D,Flatten,MaxPooling1D,LSTM

X_train, X_test, y_train, y_test = train_test_split(sequences_padded, labels, test_size=0.3, random_state=42, stratify=labels)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(y_train)

y_train_le = le.transform(y_train)
y_test_le = le.transform(y_test)


model = Sequential()
#model.add(Conv1D(32, kernel_size=3, strides=1, activation='selu', input_shape=X_train[0].shape))
#model.add(MaxPooling1D())
#model.add(Conv1D(32,kernel_size=3, strides=1, activation='selu'))
#model.add(MaxPooling1D())
#model.add(Flatten())

model.add(LSTM(32, input_shape=X_test[0].shape))
model.add(Dense(5, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train, y_train_le, epochs=50, batch_size=16, validation_data=(X_test, y_test_le),callbacks=[early_stopping])

loss, accuracy = model.evaluate(X_test, y_test_le)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

y_pred = model.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred.argmax(axis=1))  # Convert predicted labels back to original labels
print("Classification Report:")

print(classification_report(y_test, y_pred_labels))

# save test accuracy and classification report to a file
with open('Datasets\QUIC\QUIC_LSTM_CNN_TDL_Time_Filter.txt', 'w') as f:
    f.write(f"Test Accuracy: {accuracy}\n")
    f.write(f"Classification Report:\n")
    f.write(classification_report(y_test, y_pred_labels))