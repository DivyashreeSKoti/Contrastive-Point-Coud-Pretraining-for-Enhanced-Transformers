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
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader # pytorch dataloader
import random
import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import ShapeNetDataLoader as sndl
from PCT.model import Pct
from PCT.util import cal_loss
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from config.bodyscan_config import model_params
from model import create_model_st, create_model_pct

temp_flag = True
normalization = 0
root_dir = '/scratch/user/u.ds162927/ShapeNetCore.v2.PC15k/'
# file_list = './partitioned/train_99_balanced_percent_equal.txt'
# class_lst_file = './partitioned/Classified/Tail10Classes.txt'
# file_list1 = './partitioned/train_1_balanced_percent_equal.txt'
print('normalization:', normalization)

sndl_train_obj = sndl.ShapeNetDataset(root_dir, split='train', normalization = normalization)
train_dataset0, train_targets0 = sndl_train_obj.load_data()
sndl_val_obj = sndl.ShapeNetDataset(root_dir, split='val', normalization = normalization)
val_dataset0, val_targets0 = sndl_val_obj.load_data()
# modelNet40_obj = ModelNet40()
# train_dataset0, train_targets0 = modelNet40_obj.load_data(partition='train', num_points=1024)
# val_dataset0, val_targets0 = modelNet40_obj.load_data(partition='test', num_points=1024)

print(train_dataset0.shape)
print(val_dataset0.shape)

print('Cuda available :- ',torch.cuda.is_available())

# Get device to run on available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\ndevice',device,'\n')


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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate(subsample_size=subsample_size, augment_rotation = augment_rotation), drop_last=True)
    
    total_DLbatches = len(dataloader)
    
    return dataloader, total_DLbatches

num_classes = sndl_train_obj.get_num_classes()  # Number of output classes

pretrained_path = model_params['pretrained_model_path']
# Get model based on pretrained argument provided
if model_params['model'].lower() == 'pct':
    model, file_str = create_model_pct(num_classes = num_classes, pretrained_path=pretrained_path, dataset=dataset)
else:
    model, file_str = create_model_st(num_classes = num_classes, pretrained_path=pretrained_path, dataset=dataset)


    # model = Pct(dropout, output_channels = 40).to(device)
print('********************\n',str(model),'\n****************************')
model.to(device)
model = nn.DataParallel(model)
momentum = 0.9
learning_rate = 1e-4
lambda_value = 0.001
print('param',model.parameters())
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5e-4)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
criterion = cal_loss
num_epochs = 250

file_str += '_lr'+str(learning_rate)+'_ep'+str(num_epochs)+'_loss'

train_losses = []
train_accuracies = []
train_f1 = []
val_losses = []
val_accuracies = []
val_f1 = []

# Training loop
for epoch in range(num_epochs):
    train_loss_total = 0.0
    total_train_correct = 0
    total_train_samples = 0
    # print(train_targets0.shape)
    train_dataloader, train_total_DLbatches = batching(train_dataset0, train_targets0, undersampling = True,  augment_rotation = False, batch_size = 32, subsample_size= 2048)
    all_predictions = []
    all_targets = []
    
    confusion_matrices = []
    # Convert the training data to PyTorch tensors
    for i,(batch_data, batch_targets) in enumerate(train_dataloader):
        # batch_data = batch_data.permute(0, 2, 1)
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device).squeeze()
        # print(batch_data.shape)
        # Set the model to training mode
        model.train()
        
#         regularization_loss = 0
#         for param in model.parameters():
#             regularization_loss += torch.norm(param, p=2) 
        
        # Forward pass - Training
        train_outputs = model(batch_data)
        train_loss = criterion(train_outputs, batch_targets.long()) #+ lambda_value * regularization_loss
        # Compute accuracy
        _, train_predicted = torch.max(train_outputs.data, 1)
        all_predictions.extend(train_predicted.cpu().numpy())
        all_targets.extend(batch_targets.cpu().numpy())
       
        # Compute accuracy
        correct = (train_predicted == batch_targets).sum().item()
        # Accumulate the validation loss
        train_loss_total += train_loss.item()
        # Backward pass and optimization
        optimizer.zero_grad() #gradients are cleared before computing the gradients for the current batch
        train_loss.backward()
        optimizer.step()  #update the model parameters based on the computed gradients
        
        # print(correct)
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
    # Calculate F1 score
    train_f1_temp = f1_score(all_targets, all_predictions, average='macro')

    
    
    # Set the model to evaluation mode for validation
    model.eval()
    val_loss_total = 0.0
    val_accuracy_total = 0.0
    total_val_correct = 0
    total_val_samples = 0
    val_dataloader, val_total_DLbatches = batching(val_dataset0, val_targets0, undersampling = False,batch_size = 64, subsample_size=2048)
    # print('val_total_DLbatches',val_total_DLbatches)
    all_val_predictions = []
    all_val_targets = []
    for batch_data, batch_targets in val_dataloader:
        # batch_data = batch_data.permute(0, 2, 1)
        batch_data = batch_data.to(device)
        batch_targets = batch_targets.to(device).squeeze() 

        # Forward pass - Validation
        with torch.no_grad():
            val_outputs = model(batch_data)
            val_loss = criterion(val_outputs, batch_targets.long())
            # Accumulate the validation loss
            val_loss_total += val_loss.item()

            # Get predictions
            _, val_predicted = torch.max(val_outputs.data, 1)
            all_val_predictions.extend(val_predicted.cpu().numpy())
            all_val_targets.extend(batch_targets.cpu().numpy())

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
    val_f1_temp = f1_score(all_val_targets, all_val_predictions, average='macro')
    # conf_matrix = confusion_matrix(all_val_targets, all_val_predictions)
    # confusion_matrices.append(conf_matrix)
    
    
    # keep track of loss and accuracy for Monte carlo simulation
    train_losses.append(avg_train_loss)
    train_accuracies.append(avg_train_accuracy)
    train_f1.append(train_f1_temp)
    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)
    val_f1.append(val_f1_temp)
    
    # Print final results of epoch
    print(f"\r Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Train F1: {train_f1_temp:.4f}, Validation Loss: {avg_val_loss:.4f},Validation Accuracy: {avg_val_accuracy:.4f}, Val F1: {val_f1_temp:.4f}" , end="")
    print('')
    
    
# **************************************************

true_labels = []
predicted_probs = []
predicted_labels = []
all_probs = np.zeros((num_classes, num_classes))
val_dataloader, val_total_DLbatches = batching(val_dataset0, val_targets0, undersampling = False, batch_size = 64, subsample_size = 2048)

def write_to_csv(data):
    # Construct the absolute path
    fp = os.path.join(script_dir)
    # Resolve the absolute path
    fp = os.path.abspath(fp)
    temp_folder_path = fp + '/probability_output_files/'+model_params['model'].lower()+'/ShapeNetCore55/Weak_Generalization/'+ file_str
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
        print(f"Folder created: {temp_folder_path}")
    else:
        print(f"Folder already exists: {temp_folder_path}")

    output_file = temp_folder_path +'/output_file_'+ str(os.getpid())+'.csv'

    df = pd.DataFrame(data)
    
    df.to_csv(output_file, index=False)

for batch_data, batch_targets in val_dataloader:
    true_labels.extend(batch_targets.long().tolist())
    batch_data = batch_data.to(device)
    batch_targets = batch_targets.to(device)
    
    # Forward pass - Validation
    with torch.no_grad():
        val_outputs = model(batch_data)
        val_probs = torch.softmax(val_outputs, dim=1)
        _, val_predicted = torch.max(val_outputs.data, 1)
        
        # Convert tensor to numpy array
        val_probs_np = val_probs.cpu().numpy()
        val_predicted_np = val_predicted.cpu().numpy()
        batch_targets_np = batch_targets.cpu().numpy()
        
        # Update confusion matrix
        for i in range(len(batch_targets_np)):
            true_class = batch_targets_np[i]
            predicted_class = val_predicted_np[i]
            all_probs[true_class][predicted_class] += val_probs_np[i][predicted_class]

# Normalize confusion matrix to get probabilities
all_probs /= np.sum(all_probs, axis=1, keepdims=True)

print('predicted_probs',all_probs)

# output_file to store the data
write_to_csv(all_probs)
    
# **************************************************

    
# average_conf_matrix = np.mean(confusion_matrices, axis=0)

print("Validation accuracy:",*["%.8f"%(x) for x in val_accuracies])
print("Train accuracy:",*["%.8f"%(x) for x in train_accuracies])
print("Validation Loss:",*["%.8f"%(x) for x in val_losses])
print("Train Loss:",*["%.8f"%(x) for x in train_losses])
print("Validation F1:",*["%.8f"%(x) for x in val_f1])
print("Train F1:",*["%.8f"%(x) for x in train_f1])


# Stop the timer
end_time = time.time()

# Calculate the total time
total_time = end_time - start_time

# Print the total time
print("Total time:", total_time, "seconds")