#!/usr/bin/env python3
#to load data

import os

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

import os
import random
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
import itertools    

def get_min_max_lines(folder_path):
    min_lines = float('inf')
    max_lines = 0

    # Construct the absolute path: had to do this as with jobs it code was unable to find the files
    folder_path = os.path.join(script_dir, folder_path)
    print(folder_path)
    # Resolve the absolute path
    folder_path = os.path.abspath(folder_path)
    print(folder_path)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r") as file:
                num_lines = sum(1 for line in file)
                min_lines = min(min_lines, num_lines)
                if min_lines == 0:
                    print(file_name)
                max_lines = max(max_lines, num_lines)

    return min_lines, max_lines

def load_dataset(folder_path, num_lines=None):
    dataset = []
    labels = []
    len_lines = 0
    folder_path = os.path.join(script_dir, folder_path)
    print(folder_path)
    # Resolve the absolute path
    folder_path = os.path.abspath(folder_path)
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            with open(file_path, "r") as file:
                lines = file.readlines()
                if num_lines is not None:
                    if len(lines) >= num_lines:
                        selected_indices = sorted(random.sample(range(len(lines)), num_lines))
                        selected_lines = [lines[i] for i in selected_indices]
                        lines = selected_lines
                    else:
                        lines = lines + ["0 0 0\n"] * (num_lines - len(lines))
                points = []
                for line in lines:
                    point = list(map(float, line.strip().split()))  # Convert each line to a list of floats
                    
                    # Apply normalization
                    point[0] /= 200.0  # Divide x by 200
                    point[1] /= 200.0  # Divide y by 200
                    point[2] /= 1500.0  # Divide z by 1500
                    
                    points.append(point)
                dataset.append(points)
                labels.append(file_name.split(".")[0])  # Assuming the file name represents the label

    dataset = np.array(dataset)
    # Convert target labels to numerical values
    label_encoder = LabelEncoder()
    targets = label_encoder.fit_transform(labels)
    return dataset, np.array(targets), labels



class DatasetSplitter:
    def __init__(self, validation_ratio=0.2, shuffle=True):
        self.validation_ratio = validation_ratio
        self.shuffle = shuffle
        
    def reset_targets(self, targets, last_class_index=0):
         # Determine the actual classes present in the training data
        true_targets = np.unique(targets)
        # Map the target labels to class indices for training data
        class_mapping = {class_label: class_index for class_index, class_label in enumerate(true_targets, start=last_class_index)}
        mapped_targets = np.array([class_mapping[label] for label in targets])
        
        return mapped_targets, true_targets, class_mapping

    # Function for strong generalization to leave one out
    def split_dataset_by_index(self, dataset, targets, val_target = 0, binary = False, multi_sample = False):
        num_files, num_lines, num_dimensions = dataset.shape

        # get indices of samples
        indices = np.arange(num_files)
        
        # get the actual index by validation value
        if binary:
            # leave one out target is index
            val_index = np.array([val_target])
        else:
            # assigned file value is matched to leave one out index
            val_index = np.where(targets == val_target)[0]
        # Split the dataset and targets into training and validation sets
        train_data = dataset[np.setdiff1d(indices, val_index), :, :]
        train_targets = targets[np.setdiff1d(indices, val_index)]
        val_data = dataset[val_index,:, :]
        val_targets = targets[val_index]
        if binary:
            return train_data,train_targets, val_data, np.array(val_targets).reshape(-1)
        elif multi_sample: 
            
            return train_data,train_targets, val_data, np.array(val_targets).reshape(-1)
            
        train_mapped_targets, true_train_targets, train_class_mapping = self.reset_targets(train_targets)
        # Getlast class index of the training data
        train_last_class_index = len(true_train_targets)

        val_mapped_targets, _, val_class_mapping = self.reset_targets(val_targets,train_last_class_index)
        return train_data,train_mapped_targets, val_data, val_mapped_targets, train_class_mapping, val_class_mapping

    # Function to split data based on dimension.
    # Dimension 1 is to split point clouds for train and validation
    def split_dataset(self, dataset, targets,dimension=1):
        num_files, num_lines, num_dimensions = dataset.shape

        # Shuffle the indices along the num_lines axis if shuffle is enabled
        if dimension == 0:
            indices = np.arange(num_files)
            if self.shuffle:
                np.random.shuffle(indices)
            # Calculate the number of sample to include in the validation set
            validation_data = int(num_files * self.validation_ratio)
        elif dimension == 1:
            indices = np.arange(num_lines)
            if self.shuffle:
                np.random.shuffle(indices)
            else: 
                print('No shuffle')
            # Calculate the number of lines to include in the validation set
            validation_data = int(num_lines * self.validation_ratio)
        
        if self.shuffle:
            val_indices = np.random.choice(indices, size=validation_data, replace=False)
        else:
            start_of_val_indices = np.random.choice(indices)
            val_indices = indices[start_of_val_indices:start_of_val_indices + validation_data]
        if dimension == 0:
            # Split the dataset and targets into training and validation sets
            train_data = dataset[np.setdiff1d(indices, val_indices), :, :]
            train_targets = targets[np.setdiff1d(indices, val_indices)]
            val_data = dataset[val_indices,:, :]
            val_targets = targets[val_indices]
            
            train_mapped_targets, true_train_targets,_ = self.reset_targets(self, train_targets)
            # Determine the last class index used in the training data
            train_last_class_index = len(true_train_targets)

            val_mapped_targets,_,_ = self.reset_targets(self, val_targets,train_last_class_index)
            
            train_targets = train_mapped_targets
            val_targets = val_mapped_targets
        elif dimension == 1:
            # Split the dataset and targets into training and validation sets
            train_data = dataset[:, np.setdiff1d(indices, val_indices), :]
            train_targets = targets
            val_data = dataset[:, val_indices, :]
            val_targets = targets
        
        return train_data,train_targets, val_data, val_targets

# Custom Dataloader for the contrastive learning
class CustomDataLoader:
    def __init__(self, data, targets, batch_size=32, num_batches=100, subsample_size = 800, shuffle=True, augmentation_by_random_bodypart=False, augmentation_by_cube=False, augment_rotation=False, visualize_cube = False, is_training = True):
        self.data = data
        self.targets = self.targets = torch.tensor(targets.astype(np.int32))
        self.batch_size = batch_size
        self.subsample_size = subsample_size
        # self.dimension = dimension
        self.shuffle = shuffle
        self.num_samples = len(np.unique(targets))
        self.num_batches = num_batches
        self.curr_batch = 0
        # self.sub_sample_shuffle = sub_sample_shuffle
        self.augmentation_by_random_bodypart = augmentation_by_random_bodypart
        self.augmentation_by_cube = augmentation_by_cube
        self.visualize_cube = visualize_cube
        self.epoch = 0
        self.epoch_lst = [15, 30, 60, 80, 90, 100, 110, 130]
        self.rho = 0.0
        self.augment_rotation = augment_rotation
        self.training = is_training
        self.shufflecall()
        
    # Randomly choose the labels to form batches by batch size
    def shufflecall(self):
        if self.shuffle:
            self.labels = torch.randint(self.num_samples, size=(self.num_batches, self.batch_size))
        else:
            self.labels = torch.arange(self.num_samples).repeat(self.num_batches, self.batch_size)
    
    def __num_batches__(self):
        return int(np.ceil(self.num_samples / self.batch_size))
    
    # Function to generate a random rotation matrix
    def random_rotation_matrix(self, choice, fixed_angle=None, x_range = 45, y_range = 45, z_range = 180):
        rotation_matrix_x = np.eye(3)  # Initialize rotation_matrix_x as identity matrix
        rotation_matrix_y = np.eye(3)  # Initialize rotation_matrix_y as identity matrix
        rotation_matrix_z = np.eye(3)  # Initialize rotation_matrix_z as identity matrix
        if 'x' in choice:
            if fixed_angle is None:
                angle_x = np.radians(np.random.uniform(-x_range, x_range))
            else:
                angle_x = np.radians(fixed_angle)
            rotation_matrix_x = np.array([
                [1, 0, 0],
                [0, np.cos(angle_x), -np.sin(angle_x)],
                [0, np.sin(angle_x), np.cos(angle_x)]
            ])
        if 'y' in choice:
            if fixed_angle is None:
                angle_y = np.radians(np.random.uniform(-y_range, y_range))
            else:
                angle_y = np.radians(fixed_angle)
            rotation_matrix_y = np.array([
                [np.cos(angle_y), 0, np.sin(angle_y)],
                [0, 1, 0],
                [-np.sin(angle_y), 0, np.cos(angle_y)]
            ])
        if 'z' in choice: 
            if fixed_angle is None:
                angle_z = np.radians(np.random.uniform(-z_range, z_range))
            else:
                angle_z = np.radians(fixed_angle)
            rotation_matrix_z = np.array([
                [np.cos(angle_z), -np.sin(angle_z), 0],
                [np.sin(angle_z), np.cos(angle_z), 0],
                [0, 0, 1]
            ])

        # # Combine the rotation matrices
        rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

        return rotation_matrix

    def get_upper_limit_subsample_size(self, num_subsamples):
        # randomly subsample the point clouds by random size
        # subsample_size =  np.random.choice(range(self.subsample_size + 1, num_subsamples + 1), size=1)[0]
        # subsample_size = 10000
        if self.rho == 0.0:
            ss = self.subsample_size + 1
        elif self.rho == 1.0:
            ss = num_subsamples
        else:
            ss = int((num_subsamples - (num_subsamples // 9)) * self.rho)
            
        ss = max(ss, self.subsample_size + 1)
        return ss
    
    def get_labels(self, batch_index):
        return self.labels[batch_index]
        
    #for contrastive learning getting augemented data and form matrices
    def __getitem__(self, batch_index):
        batch = ([], [])
        # print(self.data[0].shape[0])
        if self.augmentation_by_random_bodypart:
            augment_lenght = np.random.choice(range(self.data[0].shape[0]//5))
        elif self.augmentation_by_cube:
            augment_length = 1024 # Keep it constant
        if self.epoch in self.epoch_lst:
            self.update_rho()
        if self.training:
            self.epoch += 1
        # print(self.data[0])
        # print(self.targets)
        # print(self.num_samples)
        # print(self.labels)
        # Get label for each batch
        for label in self.labels[batch_index]:
            # print(label)
            indices = torch.nonzero(self.targets == label).squeeze()
            # print('---',indices)
            # print(indices.numel())
            if indices.numel()>1:
                indice = random.choice(indices)
                # print('***',indice)
            else:
                indice = indices
            temp_data = self.data[indice]
            # print('===',temp_data.shape)
            # return temp_data
            num_subsamples = temp_data.shape[0]
            # print(num_subsamples)
            subsample_indices = np.arange(num_subsamples)
            if self.augmentation_by_random_bodypart:
                variant_a_indices = self.augmentation_by_random_bodypart_subsample(num_subsamples,subsample_indices,augment_lenght)
                variant_b_indices = self.augmentation_by_random_bodypart_subsample(num_subsamples,subsample_indices,augment_lenght)
            elif self.augmentation_by_cube and self.visualize_cube:
                variant, random_point, small_cube_min, small_cube_max = self.augmentation_by_cube_subsample(temp_data,num_subsamples,subsample_indices,augment_length)
                return variant, random_point, small_cube_min, small_cube_max, temp_data
            elif self.augmentation_by_cube and not self.visualize_cube:
                variant_a = self.augmentation_by_cube_subsample(temp_data,num_subsamples,subsample_indices,augment_length)
                variant_b = self.augmentation_by_cube_subsample(temp_data,num_subsamples,temp_data,augment_length)
            # elif self.augment_rotation:
            #     print('rotation')
            #     variant_a = self.augmentation_rotation(temp_data)
            #     variant_b = self.augmentation_rotation(temp_data)
            #     print('variant_a',variant_a.shape)
            #     print('variant_b',variant_b.shape)
            else:
                # print(range(num_subsamples))
                # print(self.subsample_size)
                variant_a_indices = random.sample(range(num_subsamples), self.subsample_size)
                variant_b_indices = random.sample(range(num_subsamples), self.subsample_size)
        
            if not self.augmentation_by_cube:
                variant_a = temp_data[variant_a_indices]
                variant_b = temp_data[variant_b_indices]
                if self.augment_rotation:
                    variant_a = self.augmentation_rotation(variant_a)
                    variant_b = self.augmentation_rotation(variant_b)
                    # print('variant_a',variant_a.shape)
            
            batch[0].append(variant_a)
            batch[1].append(variant_b)
        return tuple(np.array(batch)) # batch
    
    def __len__(self):
        return self.num_batches
    
    def on_epoch_end(self):
        self.shufflecall()

class LoadDataBatch:
    # Load dataset
    def load_data(self, folder_path):
        min_lines, max_lines = get_min_max_lines(folder_path)
        print(f"The minimum number of lines among all files: {min_lines}")
        print(f"The maximum number of lines among all files: {max_lines}")
        
        dataset, targets, labels = load_dataset(folder_path, num_lines=min_lines)
        return dataset, targets, labels
    
    # Custom data split 80:20
    def split_data(self, dataset, targets, validation_ratio=0.2):
        splitter = DatasetSplitter(validation_ratio=validation_ratio, shuffle=True)
        train_data, train_targets, val_data, val_targets = splitter.split_dataset(dataset, targets)
        return train_data, train_targets, val_data, val_targets
    
    # Custom collate function to subsample point cloud data
    def custom_collate(self, subsample_size):
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
    
    # Load Metadata as targets
    def get_meta_data(self, field, sheet_name, folder_path, bins = 0):
        fields = ['File', field] 
        metaData_df = pd.read_excel('MetaData.xlsx', sheet_name=sheet_name, usecols=fields)

        nan_flag = metaData_df.isna()
        if nan_flag.values.any():
            print(nan_flag)
        file_lst = metaData_df['File'].to_numpy()
        file_lst_without_obj = np.array([s.replace(".obj", "") for s in file_lst])
        file_lst_without_obj = file_lst_without_obj.tolist()
        metaData_df['File'] = file_lst_without_obj
        
        # Load data and corresponding targets
        dataset, _, labels = self.load_data(folder_path)

        # Zip labels0 and dataset0
        zipped_data = list(zip(dataset, labels))
  
        # Create a DataFrame from the zipped data
        zipped_df = pd.DataFrame(zipped_data, columns=['Dataset', 'File'])

        # Merge zipped_df with metaData_df on the 'File' column
        merged_df = pd.merge(zipped_df, metaData_df, on='File')
        
        # Separate the columns
        labels = merged_df['File'].to_numpy()
        dataset = np.stack(merged_df['Dataset'].to_numpy())
        targets = merged_df[field].to_numpy()
        
        if bins > 0:
            # Specify the number of bins
            kb_discretizer = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='quantile')
            targets = kb_discretizer.fit_transform(targets.reshape(-1, 1))
            targets = targets.flatten()
        
        return dataset, targets

