#!/usr/bin/env python3
# coding: utf-8
import sys
import os
import numpy as np
import torch
from DataLoader import LoadDataBatch
from DataLoader import DatasetSplitter
from model import create_model_st, create_model_pct
from train.train_validate import train_model, get_probabilities
import SetTransformer_Extrapolating as ST
from utils import get_current_dir, write_to_csv
from config.bodyscan_config import model_params
import time

def get_dataset_targets(ld, folder_path, num_classes):
    field = 'Age'
    sheet_name = 'filled_chamfer_dist_age1'
    dataset, targets = ld.get_meta_data(field, sheet_name, folder_path, bins = num_classes)
    
    return dataset, targets

# Funtion to custom data split 80:20
def call_splitter(val_target_loo):
    splitter = DatasetSplitter(validation_ratio=0.2, shuffle=True)
    train_data,train_targets, val_data, val_targets, train_class_mapping, val_class_mapping = splitter.split_dataset_by_index(dataset, targets, val_target=val_target_loo)
    return train_data,train_targets, val_data, val_targets, train_class_mapping, val_class_mapping

def call_dataloader(train_data,train_targets, val_data, val_targets):
    train_batch_size = 4
    val_batch_size = 1
    train_subsample_size = 8000
    val_subsample_size = 2000
    
    train_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_targets)),
        batch_size=train_batch_size, shuffle=True, collate_fn=ld.custom_collate(subsample_size=train_subsample_size)
    )

    val_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_targets)),
        batch_size=val_batch_size, shuffle=True, collate_fn=ld.custom_collate(subsample_size=val_subsample_size)
    )

    return train_dataloader, val_dataloader



# val_target_loo --> leave one out
def fit(val_target_loo):
    global file_name
    train_data,train_targets, val_data, val_targets, train_class_mapping, val_class_mapping = call_splitter(val_target_loo)
    train_dataloader, val_dataloader,  = call_dataloader(train_data,train_targets, val_data, val_targets)

    pretrained_path = model_params['pretrained_model_path']

    if model_params['model'].lower() == 'pct':
        model, file_name = create_model_pct(num_classes = (num_classes-1), pretrained_path=pretrained_path)
    else:
        model, file_name = create_model_st(num_classes = (num_classes-1), pretrained_path=pretrained_path)

    # Train the model
    predicted_probs, true_labels = train_model(
        model=model, train_dataloader = train_dataloader, val_dataloader=val_dataloader, device = device, num_epochs = 2, wg_flag = False, age_flag = True
    )
    return predicted_probs

if __name__ == "__main__": 
    
    # Keep track of run time
    start_time = time.time()

    # Define global variable to keep track of uniquestring fr outputfile
    file_name = ''

    folder_path = model_params['folder_path']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('\ndevice', device, '\n')

    ld = LoadDataBatch()
    
    # Load data
    num_classes = 4
    dataset, targets = get_dataset_targets(ld, folder_path, num_classes)
    

    # appended_probs to track of confusion probabilties of all other classes
    appended_probs = np.empty((0, num_classes-1))

    for i in range(num_classes):
        print('####### Left out ==> ', i, ' #######')
        predicted_probs = fit(i)

        predicted_probs = np.array(predicted_probs)

        appended_probs = np.concatenate((appended_probs, predicted_probs), axis=0)  # Concatenate along axis 0

    # all_probs to keep track of the num_files*num_files matrix
    all_probs = np.zeros((num_classes, num_classes))

    # Iterate over each row in the original (n-1) x n_minus_1 matrix
    for i in range(num_classes-1):
        # Copy elements to the left of the diagonal
        all_probs[i, :i] = appended_probs[i, :i]

        # Shift elements to the right by one position
        all_probs[i, i+1:] = appended_probs[i, i:]

    # Insert the last row from the (n-1) x n_minus_1 matrix to the last row of the new matrix
    all_probs[-1, :-1] = appended_probs[-1]


    print("\nTransformed n x n matrix:")
    print(all_probs)
    print('end')

    # Save results
    result_path = '/probability_output_files/binned_age/'+model_params['model'].lower()+'/strong_generalization/'
    script_dir = get_current_dir()

    write_to_csv(all_probs, script_dir, result_path, file_name)

    # Stop the timer
    end_time = time.time()

    # Calculate the total time
    total_time = end_time - start_time

    # Print the total time
    print("Total time:", total_time, "seconds")