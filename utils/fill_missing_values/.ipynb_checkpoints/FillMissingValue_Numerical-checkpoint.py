#!/usr/bin/env python3
# coding: utf-8

from scipy.spatial.distance import cdist
import numpy as np
import Chamfer_Distance as cd

def fill_missing_value_numerical_cdist(pc_data, labels, targets):
    # Iterate through each set of point clouds
    for index, (mlabel,mtarget) in targets.iterrows():
        min_distance = float('inf')
        # Find the index of missing target in this set
        missing_index = np.isnan(mtarget)
        # print(mlabel)
        if missing_index:
            file_indices = [i for i, item1 in enumerate(labels) if item1 == mlabel]
            
            missing_point_cloud = pc_data[file_indices]
            # print(missing_point_cloud.shape)
            
            # Flatten the point clouds to 2D arrays
            missing_point_cloud_flat = missing_point_cloud.reshape(-1, missing_point_cloud.shape[-1])
            # print(missing_point_cloud_flat.shape)
            # Iterate through the point cloud
            for i, data in enumerate(pc_data):
                data = data.reshape(-1, data.shape[-1])
                # Skip the point with the missing target
                if labels[i]==mlabel:
                    continue
                # print(data.shape)
                # Calculate Chamfer distance between the current point and the missing point
                distance = np.mean(cdist(missing_point_cloud_flat, data, 'euclidean'))
                # print(targets.shape)
                # If this point is closer than the previous nearest neighbor, update the target
                if distance < min_distance:
                    min_distance = distance
                    # Assuming 'targets' is a pandas DataFrame
                    nearest_target = targets.loc[targets.iloc[:, 0] == labels[i]].values[0][1]
                    # print('nearest_target',nearest_target)
                    # print(targets.shape)
                    targets.iloc[index,1] = nearest_target
    return targets

def fill_missing_value_numerical_chamfer_distance(pc_data, labels, targets):
    # Iterate through each set of point clouds
    for index, (mlabel,mtarget) in targets.iterrows():
       
        # Find the index of missing target in this set
        missing_index = np.isnan(mtarget)
        print(missing_index)
        if missing_index:
            min_distance = float('inf')
            file_indices = [i for i, item1 in enumerate(labels) if item1 == mlabel]
            
            missing_point_cloud = pc_data[file_indices]
            # print(missing_point_cloud.shape)
            
            # Flatten the point clouds to 2D arrays
            missing_point_cloud_flat = missing_point_cloud.reshape(-1, missing_point_cloud.shape[-1])
            # print(missing_point_cloud_flat.shape)
            # Iterate through the point cloud
            for i, data in enumerate(pc_data):
                data = data.reshape(-1, data.shape[-1])
                # Skip the point with the missing target
                if labels[i]==mlabel:
                    continue
                # print(data.shape)
                # Calculate Chamfer distance between the current point and the missing point
                distance = cd.chamfer_distance(missing_point_cloud_flat, data)
                
                # If this point is closer than the previous nearest neighbor, update the target
                if distance < min_distance:
                    print(distance)
                    min_distance = distance
                    # Assuming 'targets' is a pandas DataFrame
                    nearest_target = targets.loc[targets.iloc[:, 0] == labels[i]].values[0][1]
                    # print('nearest_target',nearest_target)
                    # print(targets.shape)
                    targets.iloc[index,1] = nearest_target
    return targets