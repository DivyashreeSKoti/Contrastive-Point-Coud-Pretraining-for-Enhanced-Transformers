import os
import sys
import pandas as pd
import torch

def get_current_dir():
    current_dir = os.path.abspath('./')
    sys.path.append(current_dir)

    # Set the working directory to the directory containing the script
    custom_path = current_dir

    # Get the absolute path of the current script
    script_dir = os.path.abspath(custom_path)
    
    return script_dir

def write_to_csv(data, script_dir, result_path, file_name, model=None):
    fp = os.path.abspath(script_dir)
    temp_folder_path = fp + result_path + file_name
    if not os.path.exists(temp_folder_path):
        os.makedirs(temp_folder_path)
        print(f"Folder created: {temp_folder_path}")
    else:
        print(f"Folder already exists: {temp_folder_path}")

    output_file = temp_folder_path + '/output_file_' + str(os.getpid()) + '_'
    df = pd.DataFrame(data)
    df.to_csv(output_file + '.csv', index=False)
    torch.save(model, output_file + '.pth')