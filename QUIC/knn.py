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

folder_path = "../Datasets/QUIC"
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
                nrows=150,
            )
            session_df.columns = ["Timestamp", "Packet Size", "Direction"]

            if not session_df.empty:
                all_sessions.append(session_df)
                labels.append(label)

all_data = pd.concat(all_sessions, ignore_index=True)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(all_data[["Packet Size", "Direction"]])

labels_df = pd.DataFrame({"Label": labels})
label_mapping = {
    "Google Doc": 0,
    "Google Drive": 1,
    "Google Music": 2,
    "Google Search": 3,
    "Youtube": 4,
}
labels_df["Label"] = labels_df["Label"].map(label_mapping)

all_data = pd.concat([all_data, labels_df], axis=1)

X = all_data.drop("Label", axis=1)
y = all_data["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

knn_classifiers = []
for label_value in label_mapping.values():
    y_train_binary = y_train == label_value
    knn_classifier = KNeighborsClassifier(n_neighbors=5)
    knn_classifier.fit(X_train, y_train_binary)
    knn_classifiers.append(knn_classifier)


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

# Encode the categorical labels
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


# Make predictions on the testing data for each classifier
y_preds = []
for knn_classifier in knn_classifiers:
    y_pred = knn_classifier.predict(X_test)
    y_preds.append(y_pred)

# Combine predictions to get the final predicted labels
y_pred_final = [
    max(range(len(label_mapping)), key=lambda i: y_preds[i][j])
    for j in range(len(y_test))
]

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred_final)
print(f"Accuracy: {accuracy * 100:.2f}%")

print(classification_report(y_test_encoded, y_pred_final))
