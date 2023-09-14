import argparse
from pathlib import Path
import json
import numpy as np
import re
import math

def arg_parser():
    parser = argparse.ArgumentParser(description='Determine parameter values')
    parser.add_argument('--data_folder', type=Path, help='Path to data folder')
    parser.add_argument('--deter-eps-value', action='store_true', help='Determine best eps value for DBSCAN')
    parser.add_argument('--deter-temp-value', action='store_true', help='Determine best number of tracker value')
    return parser.parse_args()

def read_data_file(data_folder):
    with open(data_folder/"stats.json") as f:
        d = json.load(f)
        print(d['rmse'])
    
    timestamps = np.load(data_folder/"timestamps.npy")
    print(timestamps.shape[0])    

def main():
    args = arg_parser()

    # Find all the folders inside the data folder
    folders = [f for f in args.data_folder.iterdir() if f.is_dir()]
    choose_pattern = ['walking', 'sitting']

    for pattern in choose_pattern:
        for folder in folders:
            if pattern in str(folder):
                folder_name_list = str(folder).split('_')
                if folder_name_list[-1].isdigit() and folder_name_list[-2].isdigit():
                    read_data_file(folder)

if __name__ == "__main__":
    main()

