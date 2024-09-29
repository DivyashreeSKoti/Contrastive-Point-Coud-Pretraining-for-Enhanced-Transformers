import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

script_dir = os.path.dirname(os.path.abspath(__file__))


class ModelNet40(Dataset):
    def __init__(self, partition):
        print('ModelNet Loading',partition)
        self.partition = partition
        self.point_clouds = None
        self.targets = None
    
    def load_h5_file(self,h5_name):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
        return data, label

    def load_data(self, num_points=8000):
        folder_path = '/home/shared/dsk2v/modelnet40_ply_hdf5_2048/'
        folder_path = os.path.join(script_dir, folder_path)
        root_dir = os.path.abspath(folder_path)
        all_data = []
        all_label = []

        h5_files = glob.glob(os.path.join(root_dir, 'ply_data_%s*.h5' % self.partition))

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.load_h5_file, h5_files))

        for data, label in results:
            all_data.append(data)
            all_label.append(label)

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        # print('------', all_data.shape, all_label.shape)
        self.point_clouds = all_data
        self.targets = all_label.squeeze()
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
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
