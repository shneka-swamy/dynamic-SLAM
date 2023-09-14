import argparse
from pathlib import Path
import numpy as np

def arg_parser():
    parser = argparse.ArgumentParser(description='Determine parameter values')
    parser.add_value('--data_folder', type=Path, help='Path to data folder')
    parser.add_value('--deter-eps-value', action='store_true', help='Determine best eps value for DBSCAN')
    parser.add_value('--deter-temp-value', action='store_true', help='Determine best number of tracker value')
    return parser.parse_args()

def read_data_file(data_folder):
    # Read numpy data
    data = np.load(data_folder)
    print(data)

def main():
    args = arg_parser()
    read_data_file(args.data_folder)


if __name__ == "__main__":
    main()

