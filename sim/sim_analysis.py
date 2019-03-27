
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
    best_ext_ret_col = get_stat_col(file_name, 'best_ext_ret')

    plt.cla()
    ax = advmean_col.plot(title= env_name + ': Mean of Agent\'s advantage')
    ax.set_xlabel('Number of Updates')
    ax.set_ylabel('Mean Advantage')
    mat_fig = ax.get_figure()
    mat_fig.savefig('plots/' + env_name + '_advmean.png', bbox_inches='tight')

    plt.cla()  # Clear axis
    ax_1 = advstd_col.plot(title=env_name + ': Standard Deviation of Agent\'s advantage')
    ax_1.set_xlabel('Number of Updates')
    ax_1.set_ylabel('S.D. of Advantage')
    mat_fig = ax_1.get_figure()
    mat_fig.savefig('plots/' + env_name + '_advstd.png', bbox_inches='tight')

    plt.cla()  # Clear axis
    ax_2 = ev_col.plot(title=env_name + ': Explained variance of Agent')
    ax_2.set_xlabel('Number of Updates')
    ax_2.set_ylabel('EV')
    mat_fig = ax_2.get_figure()
    mat_fig.savefig('plots/' + env_name + '_ev.png', bbox_inches='tight')

    plt.cla()  # Clear axis
    ax_3 = opt_dyn_loss_col.plot(title=env_name + ': Optimum Dynamics Loss for Agent')
    ax_3.set_xlabel('Number of Updates')
    ax_3.set_ylabel('ODL')
    mat_fig = ax_3.get_figure()
    mat_fig.savefig('plots/' + env_name + '_opt_dyn_loss.png', bbox_inches='tight')

    plt.cla()  # Clear axis
    ax_4 = recent_best_ext_ret_col.plot(title=env_name + ': Best recent Extrinsic Reward Agent received')
    ax_4.set_xlabel('Best Recent ER')
    ax_4.set_ylabel('')
    mat_fig = ax_4.get_figure()
    mat_fig.savefig('plots/' + env_name + '_recent_best_ext_ret.png', bbox_inches='tight')

    plt.cla()  # Clear axis
    ax_5 = best_ext_ret_col.plot(title=env_name + ': Best Extrinsic Reward Agent received')
    ax_5.set_xlabel('Number of Updates')
    ax_5.set_ylabel('Best ER')
    mat_fig = ax_5.get_figure()
    mat_fig.savefig('plots/' + env_name + '_best_ext_ret.png', bbox_inches='tight')
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
