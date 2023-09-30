import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_parquet('flash.parquet')

grouped = df.groupby(['flownum', 'Label2'])

all_sessions = []
labels = []

session_number = 1  # Initialize session number
for (flownum, label), session_df in grouped:
    if len(session_df) >= 32:  # Only including session with atleast 32 packets
        session_data = session_df[['Time', 'Length', "Direction"]].iloc[:32].copy()
        session_data['session_id'] = session_number
        session_number += 1
        all_sessions.append(session_data)
        labels.append(label)

scaler = MinMaxScaler(feature_range=(0, 1))

all_data = pd.concat(all_sessions, ignore_index=True)

scaler.fit(all_data[['Time', 'Length', "Direction"]])

for i in range(len(all_sessions)):
    all_sessions[i][['Time', 'Length', "Direction"]] = scaler.transform(all_sessions[i][['Time', 'Length', "Direction"]])

final_df = pd.concat(all_sessions, ignore_index=True)

final_df.to_parquet('all_sessions.parquet', engine='pyarrow')
