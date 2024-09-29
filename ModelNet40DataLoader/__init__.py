import os
import numpy as np
from tqdm import tqdm

# Get the absolute path of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

class ModelNetDataset():
    def __init__(self, folder_path, split='train', file_list=None, normalization = None):
        folder_path = os.path.join(script_dir, folder_path)
        self.root_dir = os.path.abspath(folder_path)
        # print(self.root_dir)
        self.split = split
        self.classes = sorted(os.listdir(os.path.join(self.root_dir)))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.file_list = file_list
        self.normalization = normalization
        
    def load_data(partition):
        # download()
        BASE_DIR = os.getcwd()
        DATA_DIR = os.path.join(BASE_DIR, 'data')  # you can modify here to assign the path where dataset's root located at
        print(DATA_DIR)
        point_clouds = []
        labels = []
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
            # print(h5_name)
            f = h5py.File(h5_name)
            data = f['data'][:].astype('float32')
            # print(data)
            label = f['label'][:].astype('int64')
            f.close()
            point_clouds.append(data)
            labels.append(label)
        point_clouds = np.concatenate(point_clouds, axis=0)
        labels = np.concatenate(labels, axis=0)
        self.point_clouds, self.labels = np.array(point_clouds), np.array(labels)
        return self.point_clouds, self.labels
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        point_cloud = self.point_clouds[idx]
        label = self.labels[idx]

        return {'point_cloud': point_cloud, 'label': label}
    
    def get_num_classes(self):
        return len(np.unique(self.labels))