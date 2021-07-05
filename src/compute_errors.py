import os
import itertools
import multiprocessing
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from joblib import Parallel, delayed
from data_manager import RESULTS_PATH, UJI_D0


def process_best_df(df_, w, device, combination):
    epsilon = 0.001
    beacons = set([UJI_D0[i] for i in combination])
    df_ = df_.loc[(df_['mac'].isin(beacons))
                  & (df_['walk_id'] == w) & (df_['device'] == device)
                  & (df_['v'] < 1.8 - epsilon) & (df_['v'] > 0.2 + epsilon)
                  ]
    if not df_.empty and set(pd.unique(df_['mac'])) == beacons:
        return get_stats(df_, w, device, combination)

    return None


def get_stats(df_, w, device, combination):
    pred_mean = np.mean(df_['v'].values)
    pred_median = np.median(df_['v'].values)

    speed = df_['speed'].values[0]
    direction = df_['direction'].values[0]
    error_mn = abs(speed - pred_mean)
    error_mdn = abs(speed - pred_median)
    c = ' '.join([str(item) for item in combination])
    nbeacons = len(combination)

    rw = {
        'walk_id': w, 'direction': direction, 'device': device, 'speed': speed, 'combination': c, 'len': nbeacons,
        'pred_mean': pred_mean, 'pred_median': pred_median, 'error_mean': error_mn, 'error_median': error_mdn
    }

    return rw


if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count() - 1

    m0 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
    m1 = [1, 3, 7, 9, 11, 13, 15, 17, 19]  # n 5 does not work
    models = {0: m0, 1: m1}

    # walk_id,mac,direction,device,speed,n,rssi0_min,rssi0_max,gamma_min,gamma_max,dini_min,dini_max,v_min,v_max,dh_min,
    # dh_max,rssi0,gamma,dini,v,dh,rmean,error
    df = pd.read_csv(os.path.join(RESULTS_PATH, 'pathloss_fit_results.csv'))

    rmeans = pd.unique(df['rmean']).tolist()
    bound_confs = df[['rssi0_min', 'rssi0_max', 'gamma_min', 'gamma_max']].drop_duplicates().values.tolist()
    n_beacons = range(1, 10)
    all_stats_filename = os.path.join(RESULTS_PATH, f'stats_errors_results.csv')
    max_it = len(rmeans) * len(models) * 9 * len(bound_confs)

    with alive_bar(total=len(rmeans) * len(n_beacons) * len(models) * len(bound_confs)) as bar:
        for rmean in rmeans:
            for n in n_beacons:
                for model in models:
                    bar.text(f'  <-->  rmean: {rmean:2d} n_beacons: {n} model: {model}')
                    for bound_conf in bound_confs:
                        dfi = df.loc[(df['rssi0_min'] == bound_conf[0]) & (df['rssi0_max'] == bound_conf[1]) &
                                     (df['gamma_min'] == bound_conf[2]) & (df['gamma_max'] == bound_conf[3]) &
                                     (df['rmean'] == rmean)]
                        if dfi.empty:
                            continue

                        all_tracks = pd.unique(dfi['walk_id']).tolist()
                        all_devices = pd.unique(dfi['device']).tolist()
                        combinations = list(itertools.combinations(models[model], n))
                        grid_len = len(all_tracks) * len(all_devices) * len(combinations)
                        grid_lst = list(itertools.product(all_tracks, all_devices, combinations))

                        df_stats = pd.DataFrame(columns=['walk_id', 'direction', 'device', 'speed', 'combination',
                                                         'len', 'model', 'rssi0_min', 'rssi0_max', 'gamma_min',
                                                         'gamma_max', 'rmean', 'pred_mean', 'pred_median',
                                                         'error_mean', 'error_median'],
                                                index=range(grid_len))

                        count = 0
                        rows = Parallel(n_jobs=num_cores)(delayed(process_best_df)(dfi, *params) for params in grid_lst)
                        for row in rows:
                            if row is not None:
                                row['rmean'] = rmean
                                row['model'] = model
                                row['rssi0_min'] = bound_conf[0]
                                row['rssi0_max'] = bound_conf[1]
                                row['gamma_min'] = bound_conf[2]
                                row['gamma_max'] = bound_conf[3]
                                df_stats.iloc[count] = row
                                count += 1
                        df_stats = df_stats.iloc[:count]
                        if count > 0:
                            error_mean = np.mean(df_stats["error_mean"].values)
                            error_std = np.std(df_stats["error_mean"].values)

                            if os.path.exists(all_stats_filename):
                                df_stats.to_csv(all_stats_filename, mode='a', header=False, index=False)
                            else:
                                df_stats.to_csv(all_stats_filename, mode='w', header=True, index=False)
                        bar()
