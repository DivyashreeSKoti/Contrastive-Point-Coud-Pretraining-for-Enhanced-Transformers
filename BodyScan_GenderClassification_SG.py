#!/usr/bin/env python
# coding: utf-8

import time

# Keep track of run time
start_time = time.time()

# ********* Start ************
# This section is to ensure to get the right directory when submitting a job
import sys
import os
current_dir = os.path.abspath('./')
sys.path.append(current_dir)

# Set the working directory to the directory containing the script
custom_path = current_dir

# Get the absolute path of the current script
script_dir = os.path.abspath(custom_path)

# ********* END ************

# ********* Start of main code *************
import DataLoader as DL
from PCT.model import Pct

#to plot data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from config.bodyscan_config import model_params
from model import create_model_st, create_model_pct

from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from itertools import islice
import numpy as np
import pandas as pd

print('arg: ',sys.argv)

folder_path = "../../../../../Data/3dbsf_txt/"

min_lines, max_lines = DL.get_min_max_lines(folder_path)
print(f"The minimum number of lines among all files: {min_lines}")
print(f"The maximum number of lines among all files: {max_lines}")
field = 'Male/Female'
fields = ['File', field] 
metaData_df = pd.read_excel('MetaData.xlsx', usecols=fields)

nan_flag = metaData_df.isna()
if nan_flag.values.any():
    print(nan_flag)
file_lst = metaData_df['File'].to_numpy()
file_lst_without_obj = np.array([s.replace(".obj", "") for s in file_lst])
file_lst_without_obj = file_lst_without_obj.tolist()

# Load data and corresponding targets
dataset0, targets0, labels0 = DL.load_dataset(folder_path, num_lines = min_lines)
metaData_df['File'] = file_lst_without_obj
# print('*****',metaData_df.shape)

# Zip labels0 and dataset0
zipped_data = list(zip(dataset0, labels0))
# print(labels0)
# Create a DataFrame from the zipped data
zipped_df = pd.DataFrame(zipped_data, columns=['Dataset', 'File'])

# Merge zipped_df with metaData_df on the 'File' column
merged_df = pd.merge(zipped_df, metaData_df, on='File')
# file_search = merged_df['File'][0]

# print('====',merged_df.shape)
# print('-----',np.where(np.array(labels0) == file_search)[0][0])
# print('>>>>>',dataset0[np.where(np.array(labels0) == file_search)[0][0]])
# print('//////',zipped_df.iloc[np.where(zipped_df['File'] == file_search)[0][0]]['Dataset'])
# print('\\\\\\',merged_df.iloc[np.where(merged_df['File'] == file_search)[0][0]]['Dataset'])

# print('//////',merged_df.iloc[np.where(merged_df['File'] == file_search)[0][0]][field])

# print('*****',metaData_df.iloc[np.where(metaData_df['File'] == file_search)][field])

# Separate the columns
labels = merged_df['File'].to_numpy()
dataset = np.stack(merged_df['Dataset'].to_numpy())
targets = merged_df[field].to_numpy()

# Get device to run on available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define global variable to keep track of uniquestring fr outputfile
file_str = ''

batch_size = 4
train_subsample_size = 8000
val_subsample_size = 2048
print('batch_size =', batch_size, 'subsample_size =', train_subsample_size, 'val_subsample_size =',val_subsample_size)
pretrained_path = model_params['pretrained_model_path']

# Funtion to custom data split 80:20
def call_splitter(val_target_loo):
    splitter = DL.DatasetSplitter(shuffle = True)
    train_data0,train_targets0, val_data, val_targets = splitter.split_dataset_by_index(dataset, targets, val_target=val_target_loo, binary=True)
    
    label_encoder = LabelEncoder()
    train_targets0 = label_encoder.fit_transform(train_targets0)
    val_targets = label_encoder.transform(val_targets)
    
    target_class, target_counts = np.unique(train_targets0, return_counts=True)
    target_counts_dict = dict(zip(target_class, target_counts))
    
    min_count_class = min(target_counts_dict, key=target_counts_dict.get)
    max_count_class = max(target_counts_dict, key=target_counts_dict.get)
    min_count = target_counts_dict[min_count_class]

    indices_of_max_count_class = np.where(train_targets0 == max_count_class)
    indices_of_min_count_class = np.where(train_targets0 == min_count_class)

    temp_train_data_max_count_class0 = train_data0[indices_of_max_count_class]
    temp_train_data_min_count_class0 = train_data0[indices_of_min_count_class]
    temp_train_targets_max_count_class0 = train_targets0[indices_of_max_count_class]
    temp_train_targets_min_count_class0 = train_targets0[indices_of_min_count_class]
   
    return temp_train_data_max_count_class0,temp_train_data_min_count_class0,temp_train_targets_max_count_class0,temp_train_targets_min_count_class0, min_count, val_data, val_targets

# Custom collate function to subssample the point cloud data
def custom_collate(subsample_size):
    def collate_fn(batch):
        subsamples = []
        for data, target in batch:
            num_samples = data.shape[0]
            current_subsample_size = min(subsample_size, num_samples)
            indices = random.sample(range(num_samples), subsample_size)
            subsample = data[indices]
            subsamples.append((subsample, target))

        data, targets = zip(*subsamples)
        data = torch.stack(data, dim=0)
        targets = torch.stack(targets, dim=0)

        return data, targets
    
    return collate_fn

# Defining the fine tuning model for Contrastive pretrained model
class FineTuneModel(nn.Module):
    def __init__(self, pretrained_model, additional_layers, final_layer):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.additional_layers = additional_layers
        self.final_layer = final_layer
    
    # get_embeddings is to get embeddings from standard set transformer
    def forward(self, inputs, get_embeddings=True, get_embeddings_additional_layer=False):
        _, outputs = self.pretrained_model(inputs, get_embeddings=get_embeddings)
        embeddings = self.additional_layers(outputs)
        outputs = self.final_layer(embeddings)
        if get_embeddings_additional_layer:
            return outputs, embeddings
        return outputs

def batching(temp_train_data_max_count_class0,temp_train_data_min_count_class0,temp_train_targets_max_count_class0,temp_train_targets_min_count_class0, min_count, val_data, val_targets, sub_sample_samples = False):
    # print(temp_train_targets_max_count_class0)
    # print(temp_train_targets_min_count_class0)
    if sub_sample_samples:
        temp_train_data_max_count_class_indices = np.random.choice(len(temp_train_data_max_count_class0), size=min_count, replace=False)
        temp_train_data_max_count_class = temp_train_data_max_count_class0[temp_train_data_max_count_class_indices]
        temp_train_targets_max_count_class = temp_train_targets_max_count_class0[temp_train_data_max_count_class_indices]
        print(temp_train_targets_max_count_class)
        train_data = np.concatenate((temp_train_data_max_count_class, temp_train_data_min_count_class0), axis=0)
        train_targets = np.concatenate((temp_train_targets_max_count_class, temp_train_targets_min_count_class0), axis=0)

    else:
        train_data = np.concatenate((temp_train_data_max_count_class0, temp_train_data_min_count_class0), axis=0)
        train_targets = np.concatenate((temp_train_targets_max_count_class0, temp_train_targets_min_count_class0), axis=0)

    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_targets))
    val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_targets))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate(subsample_size=train_subsample_size))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate(subsample_size=val_subsample_size))

    train_total_DLbatches = len(train_dataloader)
    val_total_DLbatches = len(val_dataloader)
    # print('train num batches',train_total_DLbatches)
    # print('val num batches',val_total_DLbatches)
    
    return train_dataloader, val_dataloader, train_total_DLbatches, val_total_DLbatches
    
# Define the batch size, training subsample size and validation subsample size
def call_dataloader(temp_train_data_max_count_class0,temp_train_data_min_count_class0,temp_train_targets_max_count_class0,temp_train_targets_min_count_class0, min_count, val_data, val_targets):
    
    train_dataloader, val_dataloader, train_total_DLbatches, val_total_DLbatches = batching(temp_train_data_max_count_class0,temp_train_data_min_count_class0,temp_train_targets_max_count_class0,temp_train_targets_min_count_class0, min_count, val_data, val_targets, sub_sample_samples = True)
    
    return train_dataloader, val_dataloader, train_total_DLbatches, val_total_DLbatches



# val_target_loo --> leave one out
def fit(val_target_loo):
    temp_train_data_max_count_class0,temp_train_data_min_count_class0,temp_train_targets_max_count_class0,temp_train_targets_min_count_class0, min_count, val_data, val_targets = call_splitter(val_target_loo)
    
    # Get model based on pretrained argument provided
    if model_params['model'].lower() == 'pct':
        model, file_str = create_model_pct(num_classes = num_classes, pretrained_path=pretrained_path, dataset=dataset)
    else:
        model, file_str = create_model_st(num_classes = num_classes, pretrained_path=pretrained_path, dataset=dataset)
    
    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Define the optimizer
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Define the number of training epochs
    num_epochs = 250

    # Training loop
    for epoch in range(num_epochs):
        train_loss_total = 0.0
        total_train_correct = 0
        total_train_samples = 0
        
        train_dataloader, val_dataloader, train_total_DLbatches, val_total_DLbatches = call_dataloader(temp_train_data_max_count_class0,temp_train_data_min_count_class0,temp_train_targets_max_count_class0,temp_train_targets_min_count_class0, min_count, val_data, val_targets)
        
        # Convert the training data to PyTorch tensors
        for i,(batch_data, batch_targets) in enumerate(train_dataloader):
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)

            # Set the model to training mode
            model.train()

            # Forward pass - Training
            train_outputs = model(batch_data)

            batch_targets = batch_targets.view(-1, 1)
        
            train_loss = criterion(train_outputs, batch_targets.float())
            train_probabilities = torch.sigmoid(train_outputs)

            # Compute accuracy
            train_predicted = (train_probabilities > 0.5).float()
            
            # Compute accuracy
            correct = (train_predicted == batch_targets).sum().item()
            
            # Accumulate the validation loss
            train_loss_total += train_loss.item()
        
        
            # Backward pass and optimization
            optimizer.zero_grad() #gradients are cleared before computing the gradients for the current batch
            train_loss.backward()
            optimizer.step()  #update the model parameters based on the computed gradients


            train_accuracy = correct / batch_data.size(0)
            total_train_correct += correct
            total_train_samples += batch_data.size(0)
            
            # Print the progress
            print(f"\rEpoch [{epoch+1}/{num_epochs}], Progress: {i+1}/{train_total_DLbatches}, Train Loss: {train_loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}", end="", flush=True)
            print(f"\r", end="", flush=True)

        # Compute the average training loss and accuracy
        avg_train_loss = train_loss_total / train_total_DLbatches
        avg_train_accuracy = total_train_correct / total_train_samples
    
        # keep track of loss and accuracy for Monte carlo simulation
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)
   
        # Print final results of epoch
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}", end="",flush=True) 
        print('')

    # ***** Evaluate on the left out class
    model.eval()
    val_true_labels = []
    predicted_labels = []
    for batch_data, batch_targets in val_dataloader:
        val_true_labels.extend(batch_targets.long().tolist())
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        # Forward pass - Validation
        with torch.no_grad():
            val_outputs = model(batch_data)
            batch_targets = batch_targets.view(-1, 1)
            val_probabilities = torch.sigmoid(val_outputs)
            val_predicted = (val_probabilities > 0.5).float()
            predicted_labels.extend(val_predicted.long().tolist())
    predicted_labels = [item for sublist in predicted_labels for item in sublist]
    return val_true_labels, predicted_labels


############# Main ###############
# 
predicted_labels = np.empty(0)
val_true_labels = np.empty(0)
for i in range(dataset.shape[0]):
    print('####### Left out ==> ', i, ' #######')
    val_true_label, predicted_label = fit(i)
    print(val_true_label, predicted_label)
    val_true_labels = np.concatenate((val_true_labels, val_true_label), axis=0)  # Concatenate along axis 0
    predicted_labels = np.concatenate((predicted_labels, predicted_label), axis=0)  # Concatenate along axis 0
    
# Compute the confusion matrix
confusion = confusion_matrix(val_true_labels, predicted_labels)
# Calculate accuracy
accuracy = accuracy_score(val_true_labels, predicted_labels)

print("Confusion Matrix:")
print(confusion)
print("Final Accuracy:", accuracy)
print('end')


# Stop the timer
end_time = time.time()

# Calculate the total time
total_time = end_time - start_time

# Print the total time
print("Total time:", total_time, "seconds")