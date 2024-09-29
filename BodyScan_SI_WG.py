#!/usr/bin/env python3
# coding: utf-8
import sys
import os
import torch
from DataLoader import LoadDataBatch as ld
from model import create_model_st, create_model_pct
from train.train_validate import train_model, get_probabilities
import SetTransformer_Extrapolating as ST
from utils import get_current_dir, write_to_csv
from config.bodyscan_config import model_params
import time

# Keep track of run time
start_time = time.time()

print(model_params)

folder_path = model_params['folder_path']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('\ndevice', device, '\n')

# Load data
dataset, targets, _ = ld.load_data(folder_path)
num_classes = dataset.shape[0]
# Split data
train_data, train_targets, val_data, val_targets = ld.split_data(dataset, targets)

# Prepare DataLoader
batch_size = 4
train_subsample_size = 8000
val_subsample_size = 2048
train_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_targets)),
    batch_size=batch_size, shuffle=True, collate_fn=ld.custom_collate(subsample_size=train_subsample_size)
)

val_dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_targets)),
    batch_size=batch_size, shuffle=True, collate_fn=ld.custom_collate(subsample_size=val_subsample_size)
)

pretrained_path = model_params['pretrained_model_path']

if model_params['model'].lower() == 'pct':
    model, file_name = create_model_pct(num_classes = num_classes, pretrained_path=pretrained_path, dataset=dataset)
else:
    model, file_name = create_model_st(num_classes = num_classes, pretrained_path=pretrained_path, dataset=dataset)

# Train the model
train_losses, train_accuracies, val_losses, val_accuracies = train_model(
    model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=device, num_epochs = 2
)


# Save results
result_path = '/probability_output_files/body_scan/'+model_params['model'].lower()+'/weak_generalization/'
script_dir = get_current_dir()
sorted_predicted_probs = get_probabilities(model=model,val_dataloader=val_dataloader, device=device)

write_to_csv(sorted_predicted_probs, script_dir, result_path, file_name, model)

# Stop the timer
end_time = time.time()

# Calculate the total time
total_time = end_time - start_time

# Print the total time
print("Total time:", total_time, "seconds")