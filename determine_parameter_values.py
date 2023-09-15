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

    print(f'Processing {data_folder}')
    with zipfile.ZipFile(data_folder, 'r') as zip_ref:
        with zip_ref.open('stats.json') as f:
            d = json.load(f)
        
        with zip_ref.open('timestamps.npy') as f:
            timestamps = np.load(f)

        print(f'rmse: {d["rmse"]:.5f}')

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

def find_best_temp(df, pattern):
    rmses = df['rmse'].to_numpy()
    no_kfs = df['no_kf'].to_numpy()

    # If rmses is empty, return
    if rmses.size == 0:
        mean_rmse = 0
        no_kf = 0   
    
    else:
        mean_rmse = np.mean(rmses)
        no_kf = np.mean(no_kfs)
    
    #return df_store.append({'pattern': pattern, 'rmse': mean, 'no_kf': no_kf}, ignore_index=True)

    return (pattern, mean_rmse, no_kf)



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

def get_pattern(df_store):
    # print rsme and no_kf as list
    patternList = df_store['pattern'].to_numpy()
    rsmeList = df_store['rmse'].to_numpy()
    no_kfList = df_store['no_kf'].to_numpy()
    

    # print list with brackets and comma separated
    def printList(list):
        print('[' + ', '.join(map(str, list)) + ']')
    printList(patternList)
    printList(rsmeList)
    printList(no_kfList)

    print(df_store)

def main():
    args = arg_parser()

    # Find all the folders inside the data folder
    folders = [f for f in args.data_folder.iterdir() if f.is_file()]
    choose_pattern = ['walking_xyz', 'sitting_rpy']
    choose_pattern=["walking_xyz", "sitting_xyz", "sitting_halfsphere" ,"walking_rpy", "walking_static", "walking_halfsphere"]

    #df_store = pd.DataFrame(columns=['pattern', 'eps', 'rmse'])
    df_store_ate = pd.DataFrame(columns=['pattern', 'rmse', 'no_kf'])
    df_store_tr = pd.DataFrame(columns=['pattern', 'rmse', 'no_kf'])
    df_store_rr = pd.DataFrame(columns=['pattern', 'rmse', 'no_kf'])

    for pattern in choose_pattern:
        df_ape = pd.DataFrame(columns=['eps','runNo', 'rmse', 'no_kf'])
        df_rr = pd.DataFrame(columns=['eps','runNo', 'rmse', 'no_kf'])
        df_tr = pd.DataFrame(columns=['eps','runNo', 'rmse', 'no_kf'])

        for i, folder in enumerate(folders):
            # make sure pattern in folder and extension is zip
            if pattern in str(folder) and str(folder).endswith('.zip'):
                folder_name_list = str(folder.stem).split('_')
                typeStr = folder_name_list[-1]
                seqNo = folder_name_list[-2]
                epsNo = folder_name_list[-3]
                if epsNo and seqNo:
                    rmse, no_kf = read_data_file(folder)
                    
                    if 'rr' in typeStr:
                        df_rr = df_rr.append({'eps': folder_name_list[-2], 'runNo': folder_name_list[-1], 'rmse': rmse, 'no_kf': no_kf}, ignore_index=True)
                    elif 'tr' in typeStr:
                        df_tr = df_tr.append({'eps': folder_name_list[-2], 'runNo': folder_name_list[-1], 'rmse': rmse, 'no_kf': no_kf}, ignore_index=True)
                    elif 'ape' in typeStr:
                        df_ape = df_ape.append({'eps': folder_name_list[-2], 'runNo': folder_name_list[-1], 'rmse': rmse, 'no_kf': no_kf}, ignore_index=True)
                    else:
                        print("Error: typeStr not found")
                        assert False, f"{typeStr} not found"        

        # Find the best eps value
        #df_store = find_best_eps(df, df_store, pattern)

        # Print for the best template
        pattern, rmse, no_kf = find_best_temp(df_ape, pattern)
        df_store_ate = df_store_ate.append({'pattern': pattern, 'rmse': rmse, 'no_kf': no_kf}, ignore_index=True)

        pattern, rmse, no_kf = find_best_temp(df_tr, pattern)
        df_store_tr = df_store_tr.append({'pattern': pattern, 'rmse': rmse, 'no_kf': no_kf}, ignore_index=True)

        pattern, rmse, no_kf = find_best_temp(df_rr, pattern)
        df_store_rr = df_store_rr.append({'pattern': pattern, 'rmse': rmse, 'no_kf': no_kf}, ignore_index=True)

    
    # Print the pattern, rmse and no_kf as list
    print("ATE")
    get_pattern(df_store_ate)
    print("TR")
    get_pattern(df_store_tr)
    print("RR")
    get_pattern(df_store_rr)

    #plot_graph(df_store)
    # Plot a graph of eps vs rmse for each pattern
    #plot_graph(df_store)

if __name__ == "__main__":
    main()

