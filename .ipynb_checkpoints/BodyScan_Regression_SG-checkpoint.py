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
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from itertools import islice
import numpy as np
import pandas as pd
from config.bodyscan_config import model_params
from model import create_model_st, create_model_pct

print('arg: ',sys.argv)

folder_path = "../../BodyScan_Data/3dbsf_txt/"

min_lines, max_lines = DL.get_min_max_lines(folder_path)
print(f"The minimum number of lines among all files: {min_lines}")
print(f"The maximum number of lines among all files: {max_lines}")

# Get device to run on available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define global variable to keep track of uniquestring fr outputfile
file_str = ''

def get_data(field):
    if field.lower() == 'age':
        field = 'Age'
        sheetName = 'filled_chamfer_dist_age1'
    elif field.lower() == 'weight':
        field = 'Weight #'
        sheetName = 'filled_chamfer_distance'
    elif field.lower() == 'height':
        field = 'Height'
        sheetName = 'processed_height'
        
    fields = ['File', field] 
    metaData_df = pd.read_excel('MetaData.xlsx', sheet_name=sheetName, usecols=fields)

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
    print(labels0)
    # Create a DataFrame from the zipped data
    zipped_df = pd.DataFrame(zipped_data, columns=['Dataset', 'File'])

    # Merge zipped_df with metaData_df on the 'File' column
    merged_df = pd.merge(zipped_df, metaData_df, on='File')
    
    # Separate the columns
    labels = merged_df['File'].to_numpy()
    dataset = np.stack(merged_df['Dataset'].to_numpy())
    targets = merged_df[field].to_numpy()
    
    return dataset, targets

# Funtion to custom data split 80:20
def call_splitter(val_target_loo):
    splitter = DL.DatasetSplitter(shuffle=True)
    train_data,train_targets0, val_data, val_targets0 = splitter.split_dataset_by_index(dataset, targets, val_target=val_target_loo, gender=True)

    # Create a MinMaxScaler
    scaler = MinMaxScaler()
    train_targets = scaler.fit_transform(train_targets0.reshape(-1, 1))
    val_targets = scaler.transform(val_targets0.reshape(-1, 1))

    
    return train_data,train_targets, val_data, val_targets

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

def call_dataloader(train_data,train_targets, val_data, val_targets):
    train_batch_size = 5
    val_batch_size = 1
    train_subsample_size = 8000
    val_subsample_size = 2000
    print('batch_size =', train_batch_size, 'subsample_size =', train_subsample_size)
    print(val_data.shape)
    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_targets))
    val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_targets))

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=custom_collate(subsample_size=train_subsample_size))
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, collate_fn=custom_collate(subsample_size=val_subsample_size))

    train_total_DLbatches = len(train_dataloader)
    val_total_DLbatches = len(val_dataloader)
    print('train num batches',train_total_DLbatches)
    print('val num batches',val_total_DLbatches)

    return train_dataloader, val_dataloader, train_total_DLbatches, val_total_DLbatches



# val_target_loo --> leave one out
def fit(val_target_loo):
    train_data,train_targets, val_data, val_targets = call_splitter(val_target_loo)
    train_dataloader, val_dataloader, train_total_DLbatches, val_total_DLbatches = call_dataloader(train_data,train_targets, val_data, val_targets)
    pretrained_path = model_params['pretrained_model_path']
    # Get model based on pretrained argument provided
    if model_params['model'].lower() == 'pct':
        model, file_str = create_model_pct(num_classes = num_classes, pretrained_path=pretrained_path, dataset=dataset)
    else:
        model, file_str = create_model_st(num_classes = num_classes, pretrained_path=pretrained_path, dataset=dataset)
    
    
    # Define the loss function
    criterion = nn.MSELoss()

    # Define the optimizer
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []

    # Define the number of training epochs
    num_epochs = 250

    # Training loop
    for epoch in range(num_epochs):
        train_loss_total = 0.0
        
        # Convert the training data to PyTorch tensors
        for i,(batch_data, batch_targets) in enumerate(train_dataloader):
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)

            # Set the model to training mode
            model.train()

            # Forward pass - Training
            train_outputs = model(batch_data)
            # batch_targets = batch_targets.view(-1, 1)
            train_loss = criterion(train_outputs, batch_targets.float())
        
            # Accumulate the validation loss
            train_loss_total += train_loss.item()

        
            # Backward pass and optimization
            optimizer.zero_grad() #gradients are cleared before computing the gradients for the current batch
            train_loss.backward()
            optimizer.step()  #update the model parameters based on the computed gradients

        # Compute the average training loss and accuracy
        avg_train_loss = train_loss_total / train_total_DLbatches
    
        # keep track of loss and accuracy for Monte carlo simulation
        train_losses.append(avg_train_loss)
   
        # Print final results of epoch
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}", end="")
        print('')

    # ***** Evaluate on the left out class
    model.eval()
    mean_val_loss = np.empty(0)
    for i in range(10):
        for batch_data, batch_targets in val_dataloader:
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)
            # batch_targets = batch_targets.view(-1, 1)
            # Forward pass - Validation
            with torch.no_grad():
                val_outputs = model(batch_data)
                val_loss = criterion(val_outputs, batch_targets.float())
                print('----',val_loss.cpu().detach().numpy())
                mean_val_loss = np.append(mean_val_loss, val_loss.cpu().detach().numpy())
            
            batch_targets = batch_targets.cpu()
            batch_data = batch_data.cpu()
        break
    avg_val_loss = np.mean(mean_val_loss)
    return avg_val_loss


############# Main ###############
# appended_probs to keep track of the num_files*num_files matrix
field = sys.argv[1]
dataset, targets = get_data(field)
avg_val_loss_lst = np.empty(0)
for i in range(dataset.shape[0]):
    print('####### Left out ==> ', i, ' #######')
    avg_val_loss = fit(i)
    print('Validation Loss', avg_val_loss)
    avg_val_loss_lst = np.append(avg_val_loss_lst, avg_val_loss)  # Concatenate along axis 0

    
# Calculate accuracy
avg_final_loss = np.mean(avg_val_loss_lst)

print("Final Loss:", avg_final_loss)
print('end')


# Stop the timer
end_time = time.time()

# Calculate the total time
total_time = end_time - start_time

# Print the total time
print("Total time:", total_time, "seconds")