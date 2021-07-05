import os
import adjustText
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_manager import RESULTS_PATH


def grouped_rmeans(model=0):
    df = pd.read_csv(os.path.join(RESULTS_PATH, 'stats.csv'))
    df = df.loc[(df['model'] == model)]

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), dpi=200, sharex='all', sharey='all')

    plt.subplots_adjust(wspace=0, hspace=0)

    cmap = plt.cm.get_cmap('tab20')
    gr_rmeans = [list(range(group, group + 5)) for group in range(1, 31, 5)]

    for row in range(2):
        for column in range(3):
            axis = ax[row, column]
            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            if row == 1:
                axis.set_xlabel('# beacons')
                axis.set_xticks(range(1, 10))
                axis.spines['bottom'].set_color('0.5')
            else:
                axis.spines['bottom'].set_visible(False)
                axis.set_xticks([])
                axis.tick_params(axis='x', colors='white')

            if column == 0:
                axis.set_ylabel('error m/s')
                axis.set_yticks(np.arange(0, 0.305, 0.1))
                axis.spines['left'].set_color('0.5')
            else:
                axis.grid(True)
                axis.spines['left'].set_visible(False)
                axis.tick_params(axis='y', colors='white')

            rmeans = gr_rmeans[row * 3 + column]
            axis.axhline(y=0.10, c='k', linestyle='--', linewidth=1.5, alpha=0.70)
            for i, rmean in enumerate(rmeans):
                dfp = df.loc[(df['rmean'] == rmean) & (df['device'] == 'all')]
                if not dfp.empty:
                    axis.plot(dfp['length'], dfp['error'], label=rmean, marker='o', color=cmap(i * 2))

            axis.set_xlim(0.5, 9.5)
            axis.set_ylim(0.00, 0.32)
            axis.xaxis.grid(False)
            axis.yaxis.grid(True)
            axis.legend(title='window size', fontsize='small', loc='upper center')
            str_model = 'iBKS 105' if model == 0 else 'iBKS plus'
            plt.suptitle(f'Error in gait speed estimation for beacons of model {str_model}')
            plt.tight_layout()

    filename = f'all_results_beacons_combined_model{model}_grouped_rmeans.png'
    plt.savefig(os.path.join('..', 'plots', filename))


def all_devices_rmean_backtracking(model=0, rmean=13, beacons_range=None, error_range=None):
    df = pd.read_csv(os.path.join(RESULTS_PATH, 'stats_with_backtracking.csv'))
    df = df.loc[(df['model'] == model) & (df['rmean'] == rmean) & (df['device'] != 'all')]
    if beacons_range is not None:
        df = df.loc[(df['length'] >= beacons_range[0]) & (df['length'] <= beacons_range[1])]
    df = df.drop(['rmean', 'model'], axis=1)

    lengths = pd.unique(df['length']).tolist()
    maxbcks = pd.unique(df['max_back']).tolist()

    for length in lengths:
        for maxbck in maxbcks:
            dfi = df.loc[(df['length'] == length) & (df['max_back'] == maxbck)]
            all_avg_error = np.average(dfi['error'], weights=dfi['srate'])
            all_avg_srate = np.average(dfi['srate'])
            error_std = np.average(dfi['std'], weights=dfi['srate'])
            error_q75 = np.average(dfi['q75'], weights=dfi['srate'])
            error_q90 = np.average(dfi['q90'], weights=dfi['srate'])
            df = df.append({'max_back': maxbck, 'device': 'all', 'length': length, 'error': all_avg_error,
                           'std': error_std, 'q75': error_q75, 'q90': error_q90, 'srate': all_avg_srate},
                           ignore_index=True)

    df = df.loc[df['device'] == 'all']

    max_backs = sorted(pd.unique(df['max_back']).tolist())
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=100)
    cmap = plt.cm.get_cmap('tab20')
    texts = []
    ax.axhline(y=0.10, c='k', linestyle='--', linewidth=1.5, alpha=0.70)
    for i, max_back in enumerate(max_backs):
        dfp = df.loc[(df['max_back'] == max_back)]
        if not dfp.empty:
            if max_back == 1:
                ax.plot(dfp['length'], dfp['error'], label='without backtracking', marker='o', color='k', linewidth=4,
                        linestyle='--', alpha=0.75)
            else:
                ax.plot(dfp['length'], dfp['error'], label=max_back - 1, marker='o', color=cmap(i * 2), linewidth=4,
                        alpha=0.75)
            for index, dfrow in dfp.iterrows():
                if dfrow['error'] < error_range[1]:
                    texts.append(ax.text(dfrow['length'], dfrow['error'], f"{100*dfrow['srate']:.0f}%",
                                         fontsize=14, fontweight='normal'))

    ax.set_xlabel('# beacons')
    ax.set_ylabel('error m/s')
    if beacons_range is None:
        ax.set_xlim(0.5, 9.5)
        ax.set_xticks(range(1, 10))
    else:
        ax.set_xlim(beacons_range[0] - .25, beacons_range[1] + .25)
        ax.set_xticks(range(beacons_range[0], beacons_range[1] + 1))

    if error_range is None:
        ax.set_ylim(0.00, 0.32)
        ax.set_yticks(np.arange(0, 0.305, 0.05))
    else:
        ax.set_ylim(error_range[0] - .001, error_range[1] + .001)
        ax.set_yticks(np.arange(error_range[0], error_range[1] + 0.01, 0.05))

    ax.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.20)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.suptitle('Average error for all smartwatches with window size of 13 using a backtracking strategy.')
    adjustText.adjust_text(texts, only_move={'points': 'x', 'texts': 'x'}, force_points=0.15,
                           arrowprops=dict(arrowstyle="->", color='k', lw=0.5))
    ax.legend(title='backtracking value')
    plt.tight_layout()

    filename = f'all_devices_with_backtracking_model{model}.png'
    plt.savefig(os.path.join('..', 'plots', filename))


def all_devices_rmean(model=0, rmean=13, show_all=False, beacons_range=None, error_range=None):
    df = pd.read_csv(os.path.join(RESULTS_PATH, 'stats.csv'))
    df = df.loc[(df['model'] == model)]
    if not show_all:
        df = df.loc[(df['device'] != 'all')]
    if beacons_range is not None:
        df = df.loc[(df['length'] >= beacons_range[0]) & (df['length'] <= beacons_range[1]) & (df['rmean'] == rmean)]

    df = df.drop(['rmean', 'model'], axis=1)
    for length in pd.unique(df['length']):
        dfi = df.loc[(df['length']) == length]
        error_mean = np.average(dfi['error'], weights=dfi['srate'])
        error_std = np.average(dfi['std'], weights=dfi['srate'])
        error_q75 = np.average(dfi['q75'], weights=dfi['srate'])
        error_q90 = np.average(dfi['q90'], weights=dfi['srate'])
        srate_mean = np.average(dfi['srate'])
        df = df.append({'device': 'all', 'length': length, 'error': error_mean, 'std': error_std, 'q75': error_q75,
                        'q90': error_q90, 'srate': srate_mean}, ignore_index=True)

    devices = pd.unique(df['device']).tolist()
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=100)
    cmap = plt.cm.get_cmap('tab20')
    texts = []
    ax.axhline(y=0.10, c='k', linestyle='--', linewidth=0.5, alpha=0.70)
    for i, device in enumerate(devices):
        dfp = df.loc[(df['device'] == device)]
        if not dfp.empty:
            if device == 'all':
                ax.plot(dfp['length'], dfp['error'], label='all devices avg.', marker='o', color='k', linewidth=4,
                        linestyle='--', alpha=0.75)
            else:
                ax.plot(dfp['length'], dfp['error'], label=device, marker='o', color=cmap(i * 2), linewidth=4,
                        alpha=0.75)
            for index, dfrow in dfp.iterrows():
                if dfrow['error'] < error_range[1]:
                    texts.append(ax.text(dfrow['length'], dfrow['error'], f"{100*dfrow['srate']:.0f}%",
                                         fontsize=14, fontweight='normal'))

    ax.set_xlabel('# beacons')
    ax.set_ylabel('error m/s')
    if beacons_range is None:
        ax.set_xlim(0.5, 9.5)
        ax.set_xticks(range(1, 10))
    else:
        ax.set_xlim(beacons_range[0] - .25, beacons_range[1] + .25)
        ax.set_xticks(range(beacons_range[0], beacons_range[1] + 1))

    if error_range is None:
        ax.set_ylim(0.00, 0.32)
        ax.set_yticks(np.arange(0, 0.305, 0.05))
    else:
        ax.set_ylim(error_range[0] - .001, error_range[1] + .001)
        ax.set_yticks(np.arange(error_range[0], error_range[1] + 0.01, 0.05))

    ax.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.20)
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    plt.suptitle('Average error for each smartwatch with window size of 13.')
    adjustText.adjust_text(texts, only_move={'points': 'x', 'texts': 'x'}, force_points=0.15,
                           arrowprops=dict(arrowstyle="->", color='k', lw=0.5))
    ax.legend()
    plt.tight_layout()

    filename = f'all_devices_model_{model}.png'
    plt.savefig(os.path.join('..', 'plots', filename))


if __name__ == '__main__':
    grouped_rmeans(model=0)
    grouped_rmeans(model=1)
    all_devices_rmean(model=0, rmean=13, show_all=False, beacons_range=[3, 9], error_range=[0.05, 0.20])
    all_devices_rmean_backtracking(model=0, beacons_range=[3, 9], error_range=[0.05, 0.20])
