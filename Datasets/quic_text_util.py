import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

folder_path = "C:\\Users\\chenh\\Downloads\\Traffic-Classification-Research\\Datasets\\QUIC"
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
    all_sessions[i][['Time Difference', 'Packet Size', 'Direction']] = scaler.transform(
        all_sessions[i][['Time Difference', 'Packet Size', 'Direction']])
# Creating a list of the time series.
sequences = [session[['Time Difference', 'Packet Size', 'Direction']].values for session in all_sessions]

labels = np.array(labels)

#Uncomment to save the proccessed data locally
#np.save('labels.npy', labels) 
#with open('sequences.pkl', 'wb') as f:
#    pickle.dump(sequences, f)

