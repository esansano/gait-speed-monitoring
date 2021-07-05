import os
import itertools
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from data_manager import RESULTS_PATH


def stats_backtracking(max_backtracking=5, max_back_rmean=5):
    # walk_id,direction,device,speed,combination,len,model,rssi0_min,rssi0_max,gamma_min,gamma_max,rmean,pred_mean,
    # pred_median,error_mean,error_median
    rdf = pd.read_csv(os.path.join(RESULTS_PATH, 'stats_errors_results.csv'))
    rdf = rdf.drop(['rssi0_min', 'rssi0_max', 'gamma_min', 'gamma_max', 'pred_median', 'error_median'], axis=1)
    rmeans = [13]
    devices = pd.unique(rdf['device']).tolist()
    walk_ids = pd.unique(rdf['walk_id']).tolist()
    lens = list(range(1, 10))
    models = [0]

    df = pd.DataFrame(columns=['max_back', 'device', 'rmean', 'length', 'model', 'error', 'std', 'srate', 'q75', 'q90'])
    df = df.astype(dtype={'max_back': int, 'device': str, 'rmean': int, 'length': int, 'model': int, 'error': float,
                          'std': float, 'srate': float, 'q75': float, 'q90': float})

    grid = list(itertools.product(models, lens, rmeans))

    min_rmean = min(rmeans)
    max_rmean = max(rmeans)
    with alive_bar(total=len(grid)) as bar:
        for count, (model, length, rmean) in enumerate(grid):
            m_df = rdf.loc[(rdf['model'] == model)]
            curr_rmeans = [rmean]
            for i in range(1, max_back_rmean + 1):
                if min_rmean <= rmean - i <= max_rmean:
                    curr_rmeans.append(rmean - i)
                if min_rmean <= rmean + i <= max_rmean:
                    curr_rmeans.append(rmean + i)
            for k, max_back in enumerate(range(1, max_backtracking + 1)):
                all_errors = []
                all_walks = set()
                device_walks = {}
                for device in devices:
                    dm_df = m_df.loc[(m_df['device'] == device)]
                    errors = []
                    device_walks[device] = {}
                    walks = []
                    n_walks = 0
                    for curr_len in range(length, max(0, length - max_back), -1):
                        dml_df = dm_df.loc[(dm_df['len'] == curr_len)]
                        if dml_df.empty:
                            continue
                        for curr_rmean in curr_rmeans:
                            dmlr_df = dml_df.loc[(dml_df['rmean'] == curr_rmean)]
                            if dmlr_df.empty:
                                continue
                            for walk_id in walk_ids:
                                if walk_id not in walks:
                                    dmlrw_df = dmlr_df.loc[(dmlr_df['walk_id'] == walk_id)]
                                    if not dmlrw_df.empty:
                                        errors.extend(dmlrw_df['error_mean'].tolist())
                                        walks.append(walk_id)
                                        all_walks.add(walk_id)
                                        n_walks += 1
                                        device_walks[device][walk_id] = dmlrw_df['error_mean'].tolist()

                    if len(errors) > 0:
                        error = np.mean(errors)
                        q75 = np.percentile(errors, 75)
                        q90 = np.percentile(errors, 90)
                        std = np.std(errors)
                        srate = n_walks / len(walk_ids)
                        df = df.append({'max_back': max_back, 'device': device, 'rmean': rmean, 'length': length,
                                        'model': model, 'error': error, 'std': std, 'srate': srate, 'q75': q75,
                                        'q90': q90}, ignore_index=True)

                    all_errors.extend(errors)
                if len(all_errors) > 0:
                    comp_walks = set.intersection(*[set(device_walks[d].keys()) for d in device_walks.keys()])
                    cw_errors = []
                    for walk_id in walk_ids:
                        if walk_id in comp_walks:
                            cw_errors.extend(list(itertools.chain(*[device_walks[device][walk_id] for device in device_walks.keys()])))

                    if len(cw_errors) > 0:
                        error = np.mean(cw_errors)
                        q75 = np.percentile(cw_errors, 75)
                        q90 = np.percentile(cw_errors, 90)
                        std = np.std(cw_errors)
                        srate = len(all_walks) / len(walk_ids)
                        df = df.append({'max_back': max_back, 'device': 'all', 'rmean': rmean, 'length': length,
                                        'model': model, 'error': error, 'std': std, 'srate': srate, 'q75': q75,
                                        'q90': q90}, ignore_index=True)

            bar.text(f'{100 * count/len(grid):6.2f}% ')
            bar()

    df.to_csv(os.path.join(RESULTS_PATH, 'stats_with_backtracking.csv'), index=False)


if __name__ == '__main__':
    stats_backtracking()
