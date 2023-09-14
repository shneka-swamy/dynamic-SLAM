import argparse
from pathlib import Path
import json
import numpy as np
import re
import math
import zipfile
import pandas as pd

def arg_parser():
    parser = argparse.ArgumentParser(description='Determine parameter values')
    parser.add_argument('--data_folder', type=Path, help='Path to data folder')
    parser.add_argument('--deter-eps-value', action='store_true', help='Determine best eps value for DBSCAN')
    parser.add_argument('--deter-temp-value', action='store_true', help='Determine best number of tracker value')
    return parser.parse_args()

def read_data_file(data_folder):
    # Read stat inside zip file and print rmse, without extracting the zip file

    with zipfile.ZipFile(data_folder, 'r') as zip_ref:
        with zip_ref.open('stats.json') as f:
            d = json.load(f)
        
        with zip_ref.open('timestamps.npy') as f:
            timestamps = np.load(f)

        return d['rmse'], timestamps.shape[0]

def find_best_eps(df, df_store, pattern):
    # Group the dataframe by eps value
    grouped = df.groupby('eps')
    # Find the maximum no_kf value for each eps value
    # print all 'rmses' for each eps value, eps value are columns and rows are runNo
    table = grouped['rmse'].apply(list)

    def print_row(row):
        print(' '.join(f'{x:.5f}' for x in row), end='')
        mean, std = np.mean(row), np.std(row)
        print(f'    {mean:.5f} ± {std:.5f}')

    def print_header():
        # print numbers 10, 20 ... 70 with enough spaces
        print('    ', end='')
        for eps in table.index:
            print(f'{eps:8}', end='')
        print()

    # print table with 5 decimal places
    r1, r2, r3, r4, r5, r6, r7 = table
    print_header()
    print_row(r1)
    print_row(r2)
    print_row(r3)
    print_row(r4)
    print_row(r5)
    print_row(r6)
    print_row(r7)

    for name, group in grouped:
        min_rmse = group['rmse'].min()
        df_store = df_store.append({'pattern': pattern, 'eps': name, 'rmse': min_rmse}, ignore_index=True)

    return df_store

def find_best_temp(df, df_store, pattern):
    rmses = df['rmse'].to_numpy()
    print(rmses)
    no_kfs = df['no_kf'].to_numpy()

    # If rmses is empty, return
    if rmses.size == 0:
        mean_rmse = 0
        no_kf = 0   
    
    else:
        mean = np.mean(rmses)
        no_kf = np.mean(no_kfs)
    
    return df_store.append({'pattern': pattern, 'rmse': mean, 'no_kf': no_kf}, ignore_index=True)

    # print("Pattern: ", pattern)
    # print("Average RMSE: ", rmses.mean())
    # print("Average no_kf: ", np.mean(no_kfs))



# Plot a graph of eps vs rmse for each pattern
# Not used for other dataset because of the way data is stored
def plot_graph_rmse(df_store):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context("paper")
    sns.set(font_scale=1)
    sns.lineplot(x='eps', y='rmse', hue='pattern', data=df_store)
    plt.xlabel('eps')
    plt.ylabel('ATE_RMSE')
    plt.tight_layout()
    plt.savefig('temp_vs_rmse.pdf')


def main():
    args = arg_parser()

    # Find all the folders inside the data folder
    folders = [f for f in args.data_folder.iterdir() if f.is_file()]
    choose_pattern = ['walking_xyz', 'sitting_rpy']

    #df_store = pd.DataFrame(columns=['pattern', 'eps', 'rmse'])
    df_store = pd.DataFrame(columns=['pattern', 'rmse', 'no_kf'])

    for pattern in choose_pattern:
        df = pd.DataFrame(columns=['eps','runNo', 'rmse', 'no_kf'])
        for i, folder in enumerate(folders):
            # make sure pattern in folder and extension is zip
            if pattern in str(folder) and str(folder).endswith('.zip'):
                folder_name_list = str(folder.stem).split('_')
                if folder_name_list[-1].isdigit() and folder_name_list[-2].isdigit():
                    rmse, no_kf = read_data_file(folder)
                    df = df.append({'eps': folder_name_list[-2], 'runNo': folder_name_list[-1], 'rmse': rmse, 'no_kf': no_kf}, ignore_index=True)
        

        # Find the best eps value
        #df_store = find_best_eps(df, df_store, pattern)

        # Print for the best template
        df_store = find_best_temp(df, df_store, pattern)

    plot_graph(df_store)
    # Plot a graph of eps vs rmse for each pattern
    #plot_graph(df_store)

if __name__ == "__main__":
    main()

