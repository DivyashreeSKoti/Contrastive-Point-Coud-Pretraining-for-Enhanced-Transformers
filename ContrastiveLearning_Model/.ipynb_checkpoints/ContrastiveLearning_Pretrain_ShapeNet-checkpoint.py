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
parent_dir = os.path.abspath('./../')
sys.path.append(current_dir)
sys.path.append(parent_dir)

# Set the working directory to the directory containing the script
custom_path = current_dir

# Get the absolute path of the current script
script_dir = os.path.abspath(custom_path)

# ********* END ************
import ShapeNetDataLoader as sndl
import SetTransformer_Extrapolating as ST
import ContrastiveModel as CM
import DataLoader as dl
from PCT.model import Pct
from PCT.util import cal_loss

#to plot data
import seaborn as sns

from torch.utils.data import DataLoader
import argparse

import torch
import torch.nn as nn

import numpy as np
import copy

# Check if the script is called with an argument
if len(sys.argv) < 2:
    print("Please provide an argument.")
    sys.exit(1)

# To track the total number of trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Parsing the command line arguments
parser = argparse.ArgumentParser(description='argument parser')
parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
parser.add_argument('--model_name', type=str, default='ST', help='Model Name')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
parser.add_argument('--subsample_size', type=int, default=2048, help='Batch Size')


args = parser.parse_args()

# Defining the path for input folder
root_dir = "/scratch/user/u.ds162927/ShapeNetCore.v2.PC15k/"
# file_list = '../partitioned/train_99_balanced_percent_equal.txt'
# standard scalar = 0, min-max = 1
normalization = 0

# Loading the data
sndl_train_obj = sndl.ShapeNetDataset(root_dir, split='train', normalization = normalization)
train_dataset0, train_targets0 = sndl_train_obj.load_data()
sndl_val_obj = sndl.ShapeNetDataset(root_dir, split='val', normalization = normalization)
val_dataset0, val_targets0 = sndl_val_obj.load_data()
print(train_dataset0.shape)

# Get device to run on available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('cuda: ', torch.cuda.is_available())
# Define the batch size, training subsample size and validation subsample size
batch_size = args.batch_size
subsample_size = args.subsample_size
num_train_batches=200

# Load the batches using CustomDataLoader as to create more number of batches with replacement
train_subsampled_dataloader = sndl.CustomDataLoader(train_dataset0, train_targets0, batch_size=batch_size, num_batches=num_train_batches, subsample_size=subsample_size, shuffle=True, augment_rotation = True)
val_subsampled_dataloader = sndl.CustomDataLoader(val_dataset0, val_targets0, batch_size=batch_size, num_batches=50, subsample_size=subsample_size, shuffle=True, is_training = False)

# Redefining learning rate
learning_rate = 1e-3
projection_dim = 1024
model_name = args.model_name.upper()

print('**********************************',model_name,'**********************************')

if(model_name == 'ST'):
    # Define architecture for the model
    embed_dim = 128
    num_heads = 32
    num_induce = 256
    stack=3
    ff_activation="gelu"
    dropout=0.05
    use_layernorm=True
    pre_layernorm=False
    is_final_block = False
    num_classes = 0 #this is dummy value
    loss = nn.CrossEntropyLoss()
    # Create an instances of the PyTorch model for contrastive learning base model
    masked_encoder = ST.PyTorchModel(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_induce=num_induce,
        stack=stack,
        ff_activation=ff_activation,
        dropout=dropout,
        use_layernorm=use_layernorm,
        pre_layernorm=pre_layernorm,
        is_final_block = is_final_block,
        num_classes = num_classes
    )
    unmasked_encoder = copy.deepcopy(masked_encoder)
    
elif(model_name == 'PCT'):
    dropout = 0.1
    print(sndl_train_obj.get_num_classes())
    embed_dim = 256
    learning_rate = 1e-4
    projection_dim = 1024
    loss = cal_loss
    masked_encoder = Pct(dropout, output_channels = sndl_train_obj.get_num_classes(), embed_dim=embed_dim)
    unmasked_encoder = Pct(dropout, output_channels = sndl_train_obj.get_num_classes(), embed_dim=embed_dim)
    
    # model = Pct(dropout, output_channels = 40).to(device)
    # print('********************\n',str(masked_encoder),'\n****************************')
    # model = nn.DataParallel(model)
    

# Placing model onto GPU
masked_encoder.to(device)
unmasked_encoder.to(device)
print(loss)
# Define Contrastive model
cm = CM.ContrastiveModel(device, masked_encoder=masked_encoder, unmasked_encoder=unmasked_encoder, embed_dim = embed_dim, projection_dim = projection_dim, lr = learning_rate, loss=loss)
cm.to(device)

num_epochs = args.epochs
val_accuracies, train_accuracies, val_losses, train_losses = cm.fit(train_subsampled_dataloader, val_subsampled_dataloader, epochs = num_epochs)
    
print("Validation accuracy:",*["%.8f"%(x) for x in val_accuracies])
print("Train accuracy:",*["%.8f"%(x) for x in train_accuracies])
print("Validation Loss:",*["%.8f"%(x) for x in val_losses])
print("Train Loss:",*["%.8f"%(x) for x in train_losses])


fp = os.path.join(script_dir)
fp = os.path.abspath(fp)
if(model_name == 'ST'):
    print('Details:', 'embed_dim =', embed_dim,
      'num_heads =', num_heads,
      'num_induce =', num_induce,
      'stack =', stack,
      'dropout =', dropout, 
      'batch_size =', batch_size,
      'subsample_size =', subsample_size, 
      'num_train_batches', num_train_batches,
      'num_epochs', num_epochs,
      'normalization', normalization,
      'use_layernorm', use_layernorm,
      'pre_layernorm', pre_layernorm,
      'embed_dim =', embed_dim,
      'learning_rate', learning_rate,
      'projection_dim', projection_dim)
    
    masked_encoder_model = fp+'/saved_models/ShapeNetPretrained55'+'/masked_encoder_'+str(batch_size)+'_'+str(subsample_size)+'_'+str(num_train_batches)+'_'+str(num_epochs)+'_'+str(normalization)+'_'+ str(embed_dim)+'_'+ str(num_heads)+'_'+ str(num_induce)+'_'+str(stack)+'_'+str(dropout)+'_'+str(use_layernorm)+'_'+str(pre_layernorm)+'_pd'+str(projection_dim)+'_ed'+str(embed_dim)+'_lr'+str(learning_rate)+'_sndl_'+model_name+'.pth'
    
    unmasked_encoder_model = fp+'/saved_models/ShapeNetPretrained55'+'/unmasked_encoder_'+str(batch_size)+'_'+str(subsample_size)+'_'+str(num_train_batches)+'_'+str(num_epochs)+'_'+str(normalization)+'_'+ str(embed_dim)+'_'+ str(num_heads)+'_'+ str(num_induce)+'_'+str(stack)+'_'+str(dropout)+'_'+str(use_layernorm)+'_'+str(pre_layernorm)+'_pd'+str(projection_dim)+'_ed'+str(embed_dim)+'_lr'+str(learning_rate)+'_sndl_'+model_name+'.pth'

elif(model_name == 'PCT'):
    print('Details:',
      'batch_size =', batch_size,
      'subsample_size =', subsample_size, 
      'num_train_batches', num_train_batches,
      'num_epochs', num_epochs,
      'normalization', normalization,
      'embed_dim =', embed_dim,
      'learning_rate', learning_rate,
      'projection_dim', projection_dim)
    
    masked_encoder_model = fp+'/saved_models/ShapeNetPretrained55/pct/masked_encoder_'+str(batch_size)+'_'+str(subsample_size)+'_'+str(num_train_batches)+'_'+str(num_epochs)+'_'+str(normalization)+'_pd'+str(projection_dim)+'_ed'+str(embed_dim)+'_lr'+str(learning_rate)+'_sndl_'+model_name+'_loss.pth'
    
    unmasked_encoder_model = fp+'/saved_models/ShapeNetPretrained55/pct/unmasked_encoder_'+str(batch_size)+'_'+str(subsample_size)+'_'+str(num_train_batches)+'_'+str(num_epochs)+'_'+str(normalization)+'_pd'+str(projection_dim)+'_ed'+str(embed_dim)+'_lr'+str(learning_rate)+'_sndl_'+model_name+'_loss.pth'
# Save the ContrastiveModel object
torch.save(masked_encoder, masked_encoder_model)
torch.save(unmasked_encoder, unmasked_encoder_model)


# Stop the timer
end_time = time.time()

# Calculate the total time
total_time = end_time - start_time

# Print the total time
print("Total time:", total_time, "seconds")
