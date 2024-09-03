from sklearn.metrics import confusion_matrix
import seaborn as sns
import glob
from itertools import chain
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from vit_pytorch.efficient import ViT

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Define the training settings (hyperparameters)
batch_size = 64
epochs = 100
lr = 3e-5
gamma = 0.7
seed = 42

# Set random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)

# If GPU is available - use it! Otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs('data', exist_ok=True)

# Define the paths for the train/test data
train_dir = '/home/shalev/Public/VIT/Data/QUIC_TEXT_JPG_224/1sec/train'
test_dir = '/home/shalev/Public/VIT/Data/QUIC_TEXT_JPG_224/1sec/test'

# Create lists of file paths for the train/test data
train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

# Print the size of the train/test data
print(f"Train Data: {len(train_list)}")
print(f"Test Data: {len(test_list)}")

# Extract labels of training data to a list
labels = [path.split('/')[-1].split('_')[0] for path in train_list]

# Print uqique labels
print(f"Unique Labels: {np.unique(labels)}")
print(f"Total Labels: {len(labels)}")

# Show random images from the training set
random_idx = np.random.randint(1, len(train_list), size=9)
fig, axes = plt.subplots(3, 3, figsize=(16, 12))

for idx, ax in enumerate(axes.ravel()):
    print(train_list[idx])
    img = Image.open(train_list[idx])
    ax.set_title(labels[idx])
    ax.imshow(img)

# Split the training data into train and validation sets
train_list, valid_list = train_test_split(train_list, 
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)

# Print the sizes of each
print(f"Train Data: {len(train_list)}")
print(f"Validation Data: {len(valid_list)}")
print(f"Test Data: {len(test_list)}")

# Image augmentation functions for train/validation/test
train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)
val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

# Load dataset class
class MyDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        # Load the image and apply transformations
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        # Extract the label from the filename
        label = img_path.split("/")[-1].split("_")[0]
        #labels = ['HangoutChat','HangoutVoIP','PlayMusic','YouTube']
        labels = ['0','1','2','3','4']
        # Get label's index
        label = labels.index(label)

        return img_transformed, label

# Create dataset objects for train/validation/test
train_data = MyDataset(train_list, transform=train_transforms)
valid_data = MyDataset(valid_list, transform=test_transforms)
test_data = MyDataset(test_list, transform=test_transforms)
# Create data loaders for train/validation/test
train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

# Print dataset/loader sizes
print(len(train_data), len(train_loader))
print(len(valid_data), len(valid_loader))

# Define a Linformer transformer
efficient_transformer = Linformer(
    dim=128,
    seq_len=100+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

# Create the VIT model
model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=5,
    transformer=efficient_transformer,
    channels=3,
).to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()
# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# Define scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

# The training loop
for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    # The validation loop
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )

# Test on the test set using test_loader
epoch_test_accuracy = 0
epoch_test_loss = 0
with torch.no_grad():
    for data, label in test_loader:
        data = data.to(device)
        label = label.to(device)

        test_output = model(data)
        test_loss = criterion(test_output, label)

        acc = (test_output.argmax(dim=1) == label).float().mean()        
        epoch_test_accuracy += acc / len(test_loader)
        epoch_test_loss += test_loss / len(test_loader)

    print(
        f"Epoch : {epoch+1} - loss : {epoch_test_loss:.4f} - acc: {epoch_test_accuracy:.4f}\n"
    )

y_pred = []
y_true = []

# Iterate over test data
for inputs, labels in test_loader:
        # Move inputs/labels to the same device as the model
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Feed Network
        output = model(inputs)
        # Save Prediction
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)
        # Save Truth
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)

# Constant for classes
classes = ['0','1','2','3','4']

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sns.heatmap(df_cm, annot=True)
plt.show()

# Calculate the classification report
classification_rep = classification_report(y_true, y_pred, target_names=classes)

# Print accuracy, loss, and classification report
print(f"Test Loss: {epoch_test_loss:.4f}")
print(f"Test Accuracy: {epoch_test_accuracy:.4f}")
print("Classification Report:")
print(classification_rep)
