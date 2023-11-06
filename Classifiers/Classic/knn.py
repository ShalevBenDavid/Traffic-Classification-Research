import csv
import os

import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

folder_path = (
    "../Datasets/QUIC"
)
all_sessions = []
labels = []
for label in os.listdir(folder_path):
    label_folder_path = os.path.join(folder_path, label)
    if os.path.isdir(label_folder_path):
        for filename in os.listdir(label_folder_path):
            file_path = os.path.join(label_folder_path, filename)

            session_df = pd.read_csv(
                file_path,
                sep="\t",
                header=None,
                skiprows=4,
                usecols=[0, 2, 3],
                nrows=250,
            )
            session_df.columns = ["Timestamp", "Packet Size", "Direction"]

            if not session_df.empty:
                all_sessions.append(session_df)
                labels.append(label)

# Merge all DataFrames to one DataFrame.
all_data = pd.concat(all_sessions, ignore_index=True)
# Normalizing the features to the range (0,1).
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(all_data[["Packet Size", "Direction"]])

for i in range(len(all_sessions)):
    all_sessions[i][["Packet Size", "Direction"]] = scaler.transform(
        all_sessions[i][["Packet Size", "Direction"]]
    )
# Creating a list of the time series.
sequences = [session[["Packet Size", "Direction"]].values for session in all_sessions]
from tensorflow.keras.preprocessing.sequence import pad_sequences

sequences_padded = pad_sequences(sequences, dtype="float32")
sequences_padded_2d = sequences_padded.reshape(sequences_padded.shape[0], -1)
import numpy as np

labels = np.array(labels)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    sequences_padded_2d, labels, test_size=0.3, random_state=42, stratify=labels
)

from sklearn.preprocessing import LabelEncoder

# Encoding the labels.
le = LabelEncoder()
le.fit(y_train)
# Converting all the labels to numerical values.
y_train_le = le.transform(y_train)
y_test_le = le.transform(y_test)

# Initialize and Train KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # Adjust n_neighbors as needed
knn_classifier.fit(X_train, y_train_le)

y_pred_le = knn_classifier.predict(X_test)
y_pred = le.inverse_transform(y_pred_le)

# Computing and outputting the results.
report = classification_report(y_test, y_pred)
print(report)
