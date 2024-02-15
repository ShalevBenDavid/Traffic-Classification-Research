import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import shutil


input_folder = '/home/shalev/Public/VIT/Data/QUIC_TEXT_32'
output_folder = '/home/shalev/Public/VIT/Data/QUIC_TEXT_JPG_32'

# open each folder, go over each file, resize it to 224x224, save it as a jpg and delete the original
for folder in os.listdir(input_folder):
    for file in os.listdir(os.path.join(input_folder, folder)):
        if file.endswith('.npy'):
            file_path = os.path.join(input_folder, folder, file)
            data = np.load(file_path)
            # Create output directory if it doesn't exist
            os.makedirs(os.path.join(output_folder, folder), exist_ok=True)
            for i in range(data.shape[0]):
                   # Extract the 2D array from the 4D array
                   image_data = data[i, 0, :, :]
                   print(image_data)
                   # Create an image from the array
                   image = Image.fromarray(image_data.astype('uint8'))
                   # Save the image as a JPEG file
                   output_path = os.path.join(output_folder, folder, f'{folder}_{i + 1}.jpg')
                   image.save(output_path, "JPEG")

# Split for train/test for QUIC PCAP
for folder in os.listdir('VIT/Data/QUIC_PCAP_JPG_32'):
    if folder != 'train' and folder != 'test' and folder != '.DS_Store':
        
        # Define label before the if-elif-else block
        label = None  # Initialize label to None

        if folder == "PlayMusic":
            label = "PlayMusic"
        elif folder == "HangoutChat":
            label = "HangoutChat"
        elif folder == "HangoutVoIP":
            label = "HangoutVoIP"
        elif folder == "YouTube":
            label = "YouTube"
        else:
            print("ERROR")

        if label is not None:  # Check if label is defined
            threshold = int(len(os.listdir('VIT/Data/QUIC_PCAP_JPG_32/' + folder)) * 0.7)

            for i, file in enumerate(os.listdir('VIT/Data/QUIC_PCAP_JPG_32/' + folder)):
                source_path = os.path.join('VIT/Data/QUIC_PCAP_JPG_32', folder, file)

                if i < threshold:
                    # copy to train folder
                    destination_path = os.path.join('VIT/Data/QUIC_PCAP_JPG_32/train', f'{folder}_{file}')
                    shutil.copyfile(source_path, destination_path)
                else:
                    # copy to test folder
                    destination_path = os.path.join('VIT/Data/QUIC_PCAP_JPG_32/test', f'{folder}_{file}')
                    shutil.copyfile(source_path, destination_path)


# Split for train/test for QUIC TEXT
for folder in os.listdir('/home/shalev/Public/VIT/Data/QUIC_TEXT_JPG_224/20sec'):
    if folder != 'train' and folder != 'test' and folder != '.DS_Store':
        
        # Define label before the if-elif-else block
        label = None  # Initialize label to None

        if folder == "0":
            label = "0"
        elif folder == "1":
            label = "1"
        elif folder == "2":
            label = "2"
        elif folder == "3":
            label = "3"
        elif folder == "4":
            label = "4"
        else:
            print("ERROR")

        if label is not None:  # Check if label is defined
            threshold = int(len(os.listdir('/home/shalev/Public/VIT/Data/QUIC_TEXT_JPG_224/20sec/' + folder)) * 0.7)

            for i, file in enumerate(os.listdir('/home/shalev/Public/VIT/Data/QUIC_TEXT_JPG_224/20sec/' + folder)):
                source_path = os.path.join('/home/shalev/Public/VIT/Data/QUIC_TEXT_JPG_224/20sec', folder, file)

                if i < threshold:
                    # copy to train folder
                    destination_path = os.path.join('/home/shalev/Public/VIT/Data/QUIC_TEXT_JPG_224/20sec/train', f'{label}' + '_' + f'{file}')
                    shutil.copyfile(source_path, destination_path)
                else:
                    # copy to test folder
                    destination_path = os.path.join('/home/shalev/Public/VIT/Data/QUIC_TEXT_JPG_224/20sec/test', f'{label}' + '_' + f'{file}')
                    shutil.copyfile(source_path, destination_path)