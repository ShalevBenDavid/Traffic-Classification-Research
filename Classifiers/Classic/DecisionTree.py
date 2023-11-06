import csv
import os

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

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
                nrows=450,
            )
            session_df.columns = ["Timestamp", "Packet Size", "Direction"]

            if not session_df.empty:
                all_sessions.append(session_df)
                labels.append(label)

# Merge all DataFrames to one DataFrame.
all_data = pd.concat(all_sessions, ignore_index=True)

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

# Initialize and train Decision Tree Classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)
decision_tree_classifier.fit(X_train, y_train_le)

y_pred_le = decision_tree_classifier.predict(X_test)

y_pred = le.inverse_transform(y_pred_le)
# Computing and outputting the results.
report = classification_report(y_test, y_pred)
print(report)
