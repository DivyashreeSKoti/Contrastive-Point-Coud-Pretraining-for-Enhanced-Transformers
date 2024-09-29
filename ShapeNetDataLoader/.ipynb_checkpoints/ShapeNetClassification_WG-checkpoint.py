#!/usr/bin/env python3
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
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader # pytorch dataloader
import random
import tqdm

import ShapeNetDataLoader as sndl
import SetTransformer_Extrapolating as ST

import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset

root_dir = '/home/shared/dsk2v/ShapeNetCore.v2.PC15k/'
sndl_train_obj = sndl.ShapeNetDataset(root_dir, split='train')
train_dataset0, train_targets0 = sndl_train_obj.load_data()
sndl_val_obj = sndl.ShapeNetDataset(root_dir, split='val')
val_dataset0, val_targets0 = sndl_val_obj.load_data()

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

def choose_random_files_per_label(point_clouds, labels):
    # Get unique labels and their counts
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    min_label_count = np.min(label_counts)
    chosen_point_clouds = []
    chosen_labels = []

    for label in unique_labels:
        # Get indices of samples with the current label
        label_indices = np.where(labels == label)[0]

        # Randomly pick num_files_per_label samples for the current label
        selected_indices = random.sample(list(label_indices), min(min_label_count, label_counts[label]))

        # Add selected point clouds and labels to the result
        chosen_point_clouds.extend(point_clouds[selected_indices])
        chosen_labels.extend(labels[selected_indices])

    return np.array(chosen_point_clouds), np.array(chosen_labels)

# Define the batch size, training subsample size and validation subsample size
def batching(data, targets, sub_sample_samples = False):
    # print(train_dataset1['point_cloud'])
    if sub_sample_samples:
        data,targets  = choose_random_files_per_label(data, targets)
        batch_size = 35
        subsample_size = 4000
    else:
        batch_size = len(data)//100
        subsample_size = 256
    dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(targets))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate(subsample_size=subsample_size))
    
    total_DLbatches = len(dataloader)
    
    return dataloader, total_DLbatches


print('Cuda available :- ',torch.cuda.is_available())

# Get device to run on available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\ndevice',device,'\n')
    
    
# Condition to check if pretrain modle is provided
if len(sys.argv) > 1:
    print('*************** Using pretrained ***************')
    pretrained_path = sys.argv[1]+'.pth' # Path to the pretrained model file
    pretrained_model = torch.load(pretrained_path)
    
    num_classes = bins  # Number of output classes
    projecttion_dim = 128
    
    # Additional layers for pretrained model
    additional_layers = nn.Sequential(
        nn.Linear(pretrained_model.embed_dim, projecttion_dim), 
        nn.LeakyReLU(),
        nn.Dropout(p=0.1)
    )
    
    # Final layer 
    final_layer = nn.Sequential(
        nn.Linear((projecttion_dim), num_classes)
    )
    
    # Stack all layers
    pytorch_model = FineTuneModel(
        pretrained_model,
        additional_layers,
        final_layer
    )
    file_str = sys.argv[1].split('/')[-1] # To keep unique for the output files
else:
    print('*************** New model ***************')

    # Define architecture for the model
    embed_dim = 64
    num_heads = 16
    num_induce = 128
    # feature_induce = 128
    stack=3
    ff_activation="gelu"
    dropout=0.05
    use_layernorm=False
    pre_layernorm=False
    is_final_block = False
    num_classes = sndl_train_obj.get_num_classes()

    # Create an instance of the PyTorch model
    pytorch_model = ST.PyTorchModel(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_induce=num_induce,
        stack=stack,
        ff_activation=ff_activation,
        dropout=dropout,
        use_layernorm=use_layernorm,
        is_final_block = is_final_block,
        num_classes = num_classes
    )

    print('Details:', 'embed_dim =', embed_dim,
    'num_heads =', num_heads,
    'num_induce =', num_induce,
    'stack =', stack,
    'dropout =', dropout, 
    # 'batch_size =', batch_size,
    # 'subsample_size =', train_subsample_size
         )
    # To keep unique for the output files
file_str = str(embed_dim)+'_'+str(num_heads)+'_'+str(num_induce)+'_'+str(stack)+'_'+str(dropout)

# Placing model onto GPU
pytorch_model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(pytorch_model.parameters(), lr=1e-4)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Define the number of training epochs
num_epochs = 2

# Training loop
for epoch in range(num_epochs):
    train_loss_total = 0.0
    total_train_correct = 0
    total_train_samples = 0
    
    train_dataloader, train_total_DLbatches = batching(train_dataset0, train_targets0, sub_sample_samples = True)
    
    
    # Convert the training data to PyTorch tensors
    for i,(batch_data, batch_targets) in enumerate(train_dataloader):
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        # Set the model to training mode
        pytorch_model.train()

        # Forward pass - Training
        train_outputs = pytorch_model(batch_data)

        train_loss = criterion(train_outputs, batch_targets.long())

        # Compute accuracy
        _, train_predicted = torch.max(train_outputs.data, 1)

        # Compute accuracy
        correct = (train_predicted == batch_targets).sum().item()
        # Accumulate the validation loss
        train_loss_total += train_loss.item()
        # Backward pass and optimization
        optimizer.zero_grad() #gradients are cleared before computing the gradients for the current batch
        train_loss.backward()
        optimizer.step()  #update the model parameters based on the computed gradients
        
        # Calculate training accuracy
        train_accuracy = correct / batch_data.size(0)
        total_train_correct += correct
        total_train_samples += batch_data.size(0)
        
        # Print progress
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Progress: {i+1}/{train_total_DLbatches}, Train Loss: {train_loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}", end="")
        batch_targets = batch_targets.cpu()
        batch_data = batch_data.cpu()
    # Compute the average training loss and accuracy
    avg_train_loss = train_loss_total / train_total_DLbatches
    avg_train_accuracy = total_train_correct / total_train_samples
    
    
    # Set the model to evaluation mode for validation
    pytorch_model.eval()
    val_loss_total = 0.0
    val_accuracy_total = 0.0
    total_val_correct = 0
    total_val_samples = 0
    val_dataloader, val_total_DLbatches = batching(val_dataset0, val_targets0, sub_sample_samples = True)
    for batch_data, batch_targets in val_dataloader:
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        # Forward pass - Validation
        with torch.no_grad():
            val_outputs = pytorch_model(batch_data)
            val_loss = criterion(val_outputs, batch_targets.long())
            # Accumulate the validation loss
            val_loss_total += val_loss.item()

            # Get predictions
            _, val_predicted = torch.max(val_outputs.data, 1)
            
            # Compute accuracy
            correct = (val_predicted == batch_targets).sum().item()
            total_val_correct += correct
            total_val_samples += batch_data.size(0)
            
            val_accuracy = correct / batch_data.size(0)
        batch_targets = batch_targets.cpu()
        batch_data = batch_data.cpu()
    # Compute the average validation loss and accuracy
    avg_val_loss = val_loss_total / val_total_DLbatches
    avg_val_accuracy = total_val_correct / total_val_samples

    # keep track of loss and accuracy for Monte carlo simulation
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)
    
    # Print final results of epoch
    print(f"\r Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f},Validation Accuracy: {avg_val_accuracy:.4f}" , end="")
    print('')
    
print("Validation accuracy:",*["%.8f"%(x) for x in val_accuracies])
print("Train accuracy:",*["%.8f"%(x) for x in train_accuracies])
print("Validation Loss:",*["%.8f"%(x) for x in val_losses])
print("Train Loss:",*["%.8f"%(x) for x in train_losses])


# Stop the timer
end_time = time.time()

# Calculate the total time
total_time = end_time - start_time

# Print the total time
print("Total time:", total_time, "seconds")