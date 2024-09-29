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
import numpy as np
import pandas as pd
import DataLoader as DL

#to plot data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer
import random

from PCT.modelnet_dataloader import ModelNet40
from PCT.model import Pct
from PCT.util import cal_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from itertools import islice
from config.bodyscan_config import model_params
from model import create_model_st, create_model_pct


print('arg: ',sys.argv)

temp_flag = True
modelNet40_train_obj = ModelNet40(partition='train')
train_dataset0, train_targets0 = modelNet40_train_obj.load_data()
modelNet40_val_obj = ModelNet40(partition='test')
val_dataset0, val_targets0 = modelNet40_val_obj.load_data()
train_unique_elements, counts = np.unique(train_targets0, return_counts=True)
val_unique_elements, counts = np.unique(val_targets0, return_counts=True)
print(train_dataset0.shape)
print(val_dataset0.shape)
num_classes = 40

# Get device to run on available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained_path = model_params['pretrained_model_path']

# Funtion to custom data split train:val by index
def call_splitter(val_target_loo):
    train_data,train_targets, train_last_class_index  = modelNet40_train_obj.load_dataset_by_index(leave_out_target=val_target_loo)
   
    val_data,val_targets  = modelNet40_val_obj.load_dataset_by_index(leave_out_target=val_target_loo, train_last_class_index=train_last_class_index)
    # print(unique_true_train_targets)
    return train_data,train_targets, val_data, val_targets



# Custom collate function to subssample the point cloud data
def custom_collate(subsample_size, augment_rotation = False):
    def collate_fn(batch):
        subsamples = []
        for data, target in batch:
            num_samples = data.shape[0]
            current_subsample_size = min(subsample_size, num_samples)
            indices = random.sample(range(num_samples), subsample_size)
            subsample = data[indices]
            if augment_rotation:
                # print('True')
                # Apply rotation augmentation to the subsample
                subsample = augmentation_rotation(temp_data=subsample)
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
def choose_random_files_per_label(point_clouds, labels, min_label_percent = 0.0):
    # Get unique labels and their counts
    # print(len(labels))
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    # print('label_counts',label_counts)
    if min_label_percent == 0.01:
        min_label_count = int(np.sum(label_counts) * 0.01 // len(unique_labels))
    elif min_label_percent == 0.1:
        min_label_count = int(np.sum(label_counts) * 0.1 // len(unique_labels))
    else:
        min_label_count = np.min(label_counts)
    chosen_point_clouds = []
    chosen_labels = []
    # print('min_label_percent', min_label_count)
    for label in unique_labels:
        # Get indices of samples with the current label
        label_indices = np.where(labels == label)[0]
        # Randomly pick num_files_per_label samples for the current label
        selected_indices = random.sample(list(label_indices), min(min_label_count, label_counts[label]))
        

        # Add selected point clouds and labels to the result
        chosen_point_clouds.extend(point_clouds[selected_indices])
        chosen_labels.extend(labels[selected_indices])

    return np.array(chosen_point_clouds), np.array(chosen_labels)

# Define the batch size, training subsample size and validation subsample sizefals
def batching(data, targets, undersampling = False, batch_size = 16, subsample_size=8000, min_label_percent = 0.0, augment_rotation = False):
    # print(train_dataset1['point_cloud'])
    global temp_flag 
    if undersampling:
        data,targets  = choose_random_files_per_label(data, targets, min_label_percent = min_label_percent)
        # print(len(data))
        # batch_size = 24
        # subsample_size = 5120
        # batch_size = 32
        # subsample_size = 8000
        batch_size = batch_size
        subsample_size = subsample_size
        if temp_flag:
            print('\nbs:',batch_size,'ss',subsample_size,'\n')
            temp_flag = False
    else:
        # batch_size = len(data)//100
        # subsample_size = 256
        batch_size = batch_size
        subsample_size = subsample_size
    dataset = TensorDataset(torch.from_numpy(data), torch.from_numpy(targets))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate(subsample_size=subsample_size, augment_rotation = augment_rotation))
    
    total_DLbatches = len(dataloader)
    
    return dataloader, total_DLbatches


# val_target_loo --> leave one out
def fit(val_target_loo):
    train_dataset,train_targets, val_dataset, val_targets = call_splitter(val_target_loo)
    
    global file_str
    # Get model based on pretrained argument provided
    if model_params['model'].lower() == 'pct':
        model, file_str = create_model_pct(num_classes = num_classes, pretrained_path=pretrained_path, dataset=dataset)
    else:
        model, file_str = create_model_st(num_classes = num_classes, pretrained_path=pretrained_path, dataset=dataset)
    
    # Define the loss function
    criterion = cal_loss

    # Define the optimizer
    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Define the number of training epochs
    num_epochs = 250
    batch_size = 32
    train_subsample_size = 256
    val_subsample_size = 128
    file_str += '_bs'+str(batch_size)+'_ss'+str(train_subsample_size)+'_vss'+str(val_subsample_size)
    # print(file_str)
    # Training loop
    for epoch in range(num_epochs):
        train_loss_total = 0.0
        total_train_correct = 0
        total_train_samples = 0
        
        train_dataloader, train_total_DLbatches = batching(train_dataset, train_targets, undersampling = True,  augment_rotation = False, batch_size = batch_size, subsample_size= train_subsample_size)
        
        # Convert the training data to PyTorch tensors
        for i,(batch_data, batch_targets) in enumerate(train_dataloader):
            temp_batch_size = batch_data.size(0)  # Assuming batch_data is a torch tensor

            if temp_batch_size <= 1:
                continue
                
            batch_data = batch_data.to(device)
            batch_targets = batch_targets.to(device)

            # Set the model to training mode
            model.train()

            # Forward pass - Training
            train_outputs = model(batch_data)

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


            train_accuracy = correct / batch_data.size(0)
            total_train_correct += correct
            total_train_samples += batch_data.size(0)
            
            # Print the progress
            # print(f"\rEpoch [{epoch+1}/{num_epochs}], Progress: {i+1}/{train_total_DLbatches}, Train Loss: {train_loss.item():.4f}, Train Accuracy: {train_accuracy:.4f}", end="", flush=True)
            # print(f"\r", end="", flush=True)
            batch_targets = batch_targets.cpu()
            batch_data = batch_data.cpu()
            
        # Compute the average training loss and accuracy
        avg_train_loss = train_loss_total / train_total_DLbatches
        avg_train_accuracy = total_train_correct / total_train_samples
        
        # Print final results of epoch
        print(f"\rEpoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}", end="",flush=True) 
        print('')

    # ***** Evaluate on the left out class
    model.eval()
    val_true_labels = []
    predicted_probs = []
    val_dataloader, val_total_DLbatches = batching(val_dataset, val_targets, undersampling = False,batch_size = batch_size, subsample_size=val_subsample_size)
    for batch_data, batch_targets in val_dataloader:
        # print(batch_targets)
        val_true_labels.extend(batch_targets.long().tolist())
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device)

        # Forward pass - Validation
        with torch.no_grad():
            val_outputs = model(batch_data)
            val_probs = torch.softmax(val_outputs, dim=1)
            _, val_predicted = torch.max(val_outputs.data, 1)
            predicted_probs.extend(val_probs.tolist())
        batch_targets = batch_targets.cpu()
        batch_data = batch_data.cpu()
    predicted_probs = np.mean(predicted_probs, axis = 0, keepdims=True)
    print('predicted_probs:',np.array(predicted_probs))
    # Move the model back to CPU
    model.to('cpu')
    
    # Delete the model object
    del model
    
    # Clear cached memory
    torch.cuda.empty_cache()
    
    return predicted_probs, val_true_labels

def write_to_csv(data):
    # Construct the absolute path
    fp = os.path.join(script_dir)
    # Resolve the absolute path
    fp = os.path.abspath(fp)
    temp_folder_path = fp + '/probability_output_files/ModelNet40/'+model_params['model'].lower()+'/Strong_Generalization/two/'+ file_str
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
        print(f"Folder created: {temp_folder_path}")
    else:
        print(f"Folder already exists: {temp_folder_path}")

    output_file = temp_folder_path +'/output_file_'+ str(os.getpid())+'.csv'

    df = pd.DataFrame(data)
    
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # appended_probs to keep track of the num_files*num_files matrix
    appended_probs = np.empty((0, num_classes-1))
    # appended_targets = np.empty(0)
    for i in range(18,19):
        print('####### Left out ==> ', i, ' #######')
        predicted_probs, val_true_labels = fit(i)

        appended_probs = np.concatenate((appended_probs, predicted_probs), axis=0)  # Concatenate along axis 0
        # appended_targets = np.concatenate((appended_targets, val_original_targets), axis=0)  # Concatenate along axis 0
    print('appended_probs',appended_probs.shape)
    all_probs = np.zeros((num_classes, num_classes))
    j = 0

    # Iterate over each row in the original (n-1) x n_minus_1 matrix
    for i in range(18,19):
        # Copy elements to the left of the diagonal
        all_probs[i, :i] = appended_probs[j, :i]

        # Shift elements to the right by one position
        all_probs[i, i+1:] = appended_probs[j, i:]
        j+=1

    # Insert the last row from the (n-1) x n_minus_1 matrix to the last row of the new matrix
    all_probs[-1, :-1] = appended_probs[-1]
    
    print("\nTransformed n x n matrix:")
    print(all_probs)
    # print("Sorted_predicted_probs:\n",appended_probs)
    print('end')
        
    # output_file to store the data
    write_to_csv(all_probs)



# Stop the timer
end_time = time.time()

# Calculate the total time
total_time = end_time - start_time

# Print the total time
print("Total time:", total_time, "seconds")