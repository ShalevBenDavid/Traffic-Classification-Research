import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

df = pd.read_parquet('./Datasets/quic_pcaps.parquet')

all_session = []

# Loop through each session grouped by 'flownum'
for _, session in df.groupby('flownum'):
    # Filter out rows with 'Time' greater than 1.0
    session = session[session['Time'] <= 1.0]
    # Select specific columns from the session
    session_data = session[['Time', 'Length', 'Direction', 'Label', 'flownum']]
    # Add the session data to the allsession list
    all_session.append(session_data)

scaler = MinMaxScaler()
encoder = LabelEncoder()

# Concatenate all sessions to a single DataFrame
all_data = pd.concat(all_session)
# Fit the encoder with the unique labels in the data
encoder.fit(all_data['Label'].unique())

# Fit the scaler with the specific columns in the data
scaler.fit(all_data[['Time', 'Length', 'Direction']])

# Loop through each session in all_session
for i, session in enumerate(all_session):
    # Transform the 'Label' column using the encoder
    all_session[i]['Label'] = encoder.transform(session['Label'])
    # Scale the specified columns using the scaler
    scaled_features = scaler.transform(session[['Time', 'Length', 'Direction']])
    all_session[i][['Time', 'Length', 'Direction']] = scaled_features

# Concatenate all transformed sessions to a single DataFrame
final_df = pd.concat(all_session)

final_df.to_csv('./Datasets/one_second_quic_pcaps.csv', index=False)
