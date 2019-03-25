
import re
import sys
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

def analyze_file(env_file_dict):
    for env_name, file_name in env_file_dict.items():
        form_value_dataframe(env_name, file_name)
    return

def form_value_dataframe(env_name, file_name):
    advmean_col = get_stat_col(file_name, 'eplen')
    advstd_col = get_stat_col(file_name, 'advstd')
    ev_col = get_stat_col(file_name, 'ev')
    opt_dyn_loss_col = get_stat_col(file_name, 'opt_dyn_loss')
    recent_best_ext_ret_col = get_stat_col(file_name, 'recent_best_ext_ret')

    ax = advmean_col.plot(title= env_name + ': Mean of Agent\'s advantage')
    plt.show()
    ax_1 = advstd_col.plot(title=env_name + ': Standard Deviation of Agent\'s advantage')
    plt.show()
    ax_2 = ev_col.plot(title=env_name + ': Explained variance of Agent')
    plt.show()
    ax_3 = opt_dyn_loss_col.plot(title=env_name + ': Optimum Dynamics Loss for Agent')
    plt.show()
    ax_4 = recent_best_ext_ret_col.plot(title=env_name + ': Best recent Extrinsic Reward Agent received')
    plt.show()
    return


def get_stat_col(file_name, stat):
    stat_pattern = '\|\s*' + stat + '\s*\|\s*(([-]?\d*\.?\d[eE]?[-+]?[0-9]+)*)?\s*\|'
    stat_val_list = []
    with open(file_name, 'r') as fp:
        for line in fp:
            match = re.search(stat_pattern, line)
            if match:
                f_read_value = float(match.groups()[0])
                stat_val_list.append(f_read_value)
    stat_col = pd.Series(stat_val_list, name=stat)
    return stat_col


if __name__ == '__main__':
    env_file_dict = {'Space Invaders': 'space_invaders.txt', 'Breakout': 'breakout.txt'}
    analyze_file(env_file_dict)
