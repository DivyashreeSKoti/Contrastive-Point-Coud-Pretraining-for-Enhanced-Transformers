import os
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import TensorDataset, DataLoader
from multiprocessing import Pool, cpu_count

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_files_wrapper(args):
    return load_files(*args)

def load_files(cls, split, class_to_idx, root_dir,normalization):
    label_target_dict = {}
    class_path = os.path.join(root_dir, cls, split)
    if not os.path.isdir(class_path):
        return []
    files = os.listdir(class_path)
    files = [file for file in files if os.path.isfile(os.path.join(class_path, file)) and not file.startswith('._')]
    target = class_to_idx[cls]
    label_target_dict[cls] = target

    point_clouds_cls = []
    targets_cls = []
    labels_cls = []
    for file in files:
        file_path = os.path.join(class_path, file)
        point_cloud = np.load(file_path)
        if normalization == 0:
            point_cloud = (point_cloud - np.mean(point_cloud, axis=0)) / np.std(point_cloud, axis=0)
        elif normalization == 1:
            min_vals = np.min(point_cloud, axis=0)
            max_vals = np.max(point_cloud, axis=0)
            point_cloud = (point_cloud - min_vals) / (max_vals - min_vals)

        point_clouds_cls.append(point_cloud)
        targets_cls.append(target)
        labels_cls.append(cls)
    return point_clouds_cls, targets_cls, labels_cls

class ShapeNetDataset():
    def __init__(self, folder_path, split='train', file_list=None, normalization = None, class_lst = None):
        folder_path = os.path.join(script_dir, folder_path)
        self.root_dir = os.path.abspath(folder_path)
        # print(self.root_dir)
        self.split = split
        self.classes = sorted(os.listdir(os.path.join(self.root_dir)))
        # print(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        # self.point_clouds, self.labels = self.load_data()
        self.file_list = file_list
        self.normalization = normalization
        self.point_clouds = None
        self.targets = None
        self.labels = None
        self.label_target_dict = None
        self.class_lst = class_lst
        
    def load_data(self):
        point_clouds = []
        targets = []
        labels = []
        if self.file_list is None:
            if self.class_lst is not None:
                with open(self.class_lst, 'r') as cls_file:
                    self.classes = sorted(line.strip() for line in cls_file.readlines())
                self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

            with Pool(cpu_count()) as p:
                results = list(tqdm(p.imap(load_files_wrapper, [(cls, self.split, self.class_to_idx,self.root_dir, self.normalization) for cls in self.classes]), total=len(self.classes), desc=f'Loading {self.split} data'))

            for point_clouds_cls, targets_cls, labels_cls in results:
                point_clouds.extend(point_clouds_cls)
                targets.extend(targets_cls)
                labels.extend(labels_cls)

        else:
            with open(self.file_list, 'r') as file:
                lines = file.readlines()
                if self.class_lst is not None:
                    with open(self.class_lst, 'r') as cls_file:
                        self.classes = sorted(line.strip() for line in cls_file.readlines())
                    self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

                for line in lines:
                    line = line.strip()
                    class_folder, _, file_name = line.split('/')
                    if self.class_lst is not None:
                        if class_folder not in self.classes:
                            continue
                    file_path = os.path.join(self.root_dir, line)
                    point_cloud = np.load(file_path)
                    if self.normalization == 0:
                        point_cloud = (point_cloud - np.mean(point_cloud, axis=0)) / np.std(point_cloud, axis=0)
                    elif self.normalization == 1:
                        min_vals = np.min(point_cloud, axis=0)
                        max_vals = np.max(point_cloud, axis=0)
                        point_cloud = (point_cloud - min_vals) / (max_vals - min_vals)
                    target = self.class_to_idx[class_folder]
                    self.point_clouds.append(point_cloud)
                    self.targets.append(target)
                    self.labels.append(class_folder)
                    if class_folder not in self.label_target_dict:
                        self.label_target_dict[class_folder] = target

        self.point_clouds = np.array(point_clouds)
        self.targets = np.array(targets)
        self.labels = np.array(labels)
        return self.point_clouds, self.targets
    
    def reset_targets(self, targets, last_class_index=0):
         # Determine the actual classes present in the training data
        true_targets = np.unique(targets)
        # Map the target labels to class indices for training data
        class_mapping = {class_label: class_index for class_index, class_label in enumerate(true_targets, start=last_class_index)}
        mapped_targets = np.array([class_mapping[label] for label in targets])
        
        return mapped_targets, true_targets
    
    # Function for strong generalization to leave one out
    def load_dataset_by_index(self, leave_out_target = 0, train_last_class_index = None):
        num_samples, num_points, num_dimensions = self.point_clouds.shape
        # get indices of samples
        indices = np.arange(num_samples)
        
        # assigned file value is matched to leave one out index
        leave_out_index = np.where(self.targets == leave_out_target)[0]
        if train_last_class_index:
            val_data = self.point_clouds[leave_out_index,:, :]
            val_targets = self.targets[leave_out_index]
            val_mapped_targets, _ = self.reset_targets(val_targets,train_last_class_index)
            return val_data, val_mapped_targets
        else:
             # Split the dataset and targets into training and validation sets
            train_data = self.point_clouds[np.setdiff1d(indices, leave_out_index), :, :]
            train_targets = self.targets[np.setdiff1d(indices, leave_out_index)]
            train_mapped_targets, true_train_targets = self.reset_targets(train_targets)
             # Getlast class index of the training data
            train_last_class_index = len(true_train_targets)
            return train_data, train_mapped_targets, train_last_class_index
        
        

        
        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        target = self.targets[idx]

        return {'point_cloud': point_cloud, 'target': target}
    
    def get_num_classes(self):
        return len(np.unique(self.targets))
    
    def get_label_target(self):
        return self.label_target_dict
    
    
# Custom Dataloader for the contrastive learning
class CustomDataLoader:
    def __init__(self, data, targets, batch_size=32, num_batches=100, subsample_size = 800, dimension=0, shuffle=True, augmentation_by_random_bodypart=False, augmentation_by_cube=False, augment_rotation=False, visualize_cube = False, is_training = True):
        self.data = data
        # self.targets = torch.tensor(targets.astype(np.int32))
        self.batch_size = batch_size
        self.subsample_size = subsample_size
        self.dimension = dimension
        self.shuffle = shuffle
        self.num_samples = data.shape[dimension]
        self.num_batches = num_batches
        self.curr_batch = 0
        # self.sub_sample_shuffle = sub_sample_shuffle
        self.augmentation_by_random_bodypart = augmentation_by_random_bodypart
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
    
    def __getitem_bybatchindex__(self,batch_index):
        indices = self.indices[batch_index]
        if self.dimension == 0:
            batch_data = self.data[indices]
            batch_targets = self.targets[indices]
        elif self.dimension == 1:
            batch_data = self.data[:, indices, :]
            batch_targets = self.targets
        elif self.dimension == 2:
            batch_data = self.data[:, :, indices]
            batch_targets = self.targets
        else:
            raise ValueError("Invalid dimension value. Must be 0, 1, or 2.")
       
        return batch_data, batch_targets

    
    #for contrastive learning getting augemented data and form matrices
    def __getitem__(self, batch_index):
        batch = ([], [])
        if self.training:
            self.epoch += 1
        # print('----',self.labels)
        # Get label for each batch
        for label in self.labels[batch_index]:
            # print(label)
            temp_data = self.data[label]
            # return temp_data
            num_subsamples = temp_data.shape[0]
            # print(num_subsamples)
            subsample_indices = np.arange(num_subsamples)
            
            variant_a_indices = random.sample(range(num_subsamples), self.subsample_size)
            variant_b_indices = random.sample(range(num_subsamples), self.subsample_size)
            
            variant_a = temp_data[variant_a_indices]
            variant_b = temp_data[variant_b_indices]
            
            batch[0].append(variant_a)
            batch[1].append(variant_b)
        return tuple(np.array(batch)) # batch
    
    def __len__(self):
        return self.num_batches
    
    def on_epoch_end(self):
        self.shufflecall()
        
    def get_labels(self, batch_index):
        return self.labels[batch_index]

        