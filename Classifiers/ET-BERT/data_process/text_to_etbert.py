import pandas as pd
import os

# Replace 'yourfile.txt' with the path to your text file
root_dir = 'C:\\Users\\code\\Traffic-Classification-Research\\Datasets\\QUIC'


full_df = pd.DataFrame()

flownum = 0
label_counter = 0
label_mapping = {}

for subdir,dirs,files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir,file)
        
        temp_df = pd.read_csv(file_path,sep='\s+',header=None)
        
        if subdir not in label_mapping:
            label_mapping[subdir] = label_counter
            label_counter+=1
        
        temp_df['label'] = label_mapping[subdir]
        
        temp_df['flownum'] = flownum
        flownum+=1
        
        full_df = pd.concat([full_df,temp_df],ignore_index=True)
        
full_df.columns = ['tempstamp','time','length','direction','label','flownum']
