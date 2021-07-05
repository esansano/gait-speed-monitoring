import os
import random
import itertools
import multiprocessing
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import OptimizeWarning
from alive_progress import alive_bar
from joblib import Parallel, delayed
from data_manager import DATA_PATH, RESULTS_PATH


def pathloss_f(t, rssi0, gamma, dini, v, dh):
    return rssi0 - 5 * gamma * np.log10((dini + v * t) ** 2 + dh ** 2)


def fit_pl(dff, maxfev, rssi0, gamma, dini, v, dh, rmean):
    epsilon = 0.01
    speed = dff.iloc[0]['speed']
    device = dff.iloc[0]['device']
    direction = dff.iloc[0]['direction']
    walk_id = dff.iloc[0]['walk_id']
    mac = dff.iloc[0]['mac']
    ts_data = (dff.timestamp.values - dff.iloc[0]['timestamp']) / 1000
    n = len(ts_data)
    rssi_bounds = [rssi0[0], rssi0[1]]
    gamma_bounds = [gamma[0], gamma[1]]
    result = []
    bounds = ([rssi_bounds[0], gamma_bounds[0], dini[0], v[0], dh[0]], [rssi_bounds[1], gamma_bounds[1],
                                                                        dini[1], v[1], dh[1]])
    if n > rmean:
        if rmean <= 1:
            rssi = dff.rssi.values
        else:
            rssi = dff.rssi.rolling(rmean, min_periods=1).mean().values
        try:
            params, params_cov = optimize.curve_fit(pathloss_f, ts_data, rssi, method='dogbox', bounds=bounds,
                                                    maxfev=maxfev)
            error = abs(params[3] - speed)
            if v[0] + epsilon < params[3] < v[1] - epsilon:
                result.append({
                    'walk_id': walk_id, 'mac': mac, 'direction': direction, 'device': device, 'speed': speed, 'n': n,
                    'rssi0_min': rssi_bounds[0], 'rssi0_max': rssi_bounds[1],
                    'gamma_min': gamma_bounds[0], 'gamma_max': gamma_bounds[1],
                    'dini_min': dini[0], 'dini_max': dini[1],
                    'v_min': v[0], 'v_max': v[1],
                    'dh_min': dh[0], 'dh_max': dh[1],
                    'rssi0': params[0], 'gamma': params[1], 'dini': params[2], 'v': params[3], 'dh': params[4],
                    'rmean': rmean, 'error': error}
                )
        except (RuntimeError, OptimizeWarning) as e:
            pass
    return result


if __name__ == '__main__':
    random.seed(42)
    nc = multiprocessing.cpu_count() - 1
    df = pd.read_csv(os.path.join(DATA_PATH, 'ble-gspeed.csv'))

    grid = {
        'rssi0': [(-85, -20)],
        'gamma': [(0.5, 3.0)],
        'dini': [(-10, -2)],
        'v': [(0.2, 2.0)],
        'dh': [(2.0, 3.5)],
        'rmean': list(range(1, 31))
    }

    all_tracks = pd.unique(df['walk_id']).tolist()
    all_macs = pd.unique(df['mac'])
    all_devices = pd.unique(df['device'])
    grid_it = [item for item in itertools.product(grid['rssi0'], grid['gamma'], grid['dini'],
                                                  grid['v'], grid['dh'], grid['rmean'])]
    main_loop = [item for item in itertools.product(all_tracks, all_macs, all_devices)]
    size = len(main_loop) * len(grid['rssi0']) * len(grid['gamma']) * len(grid['dini']) * len(grid['v']) * \
           len(grid['dh']) * len(grid['rmean'])
    results = pd.DataFrame(columns=['walk_id', 'mac', 'direction', 'device', 'speed', 'n',
                                    'rssi0_min', 'rssi0_max',
                                    'gamma_min', 'gamma_max',
                                    'dini_min', 'dini_max',
                                    'v_min', 'v_max',
                                    'dh_min', 'dh_max',
                                    'rssi0', 'gamma', 'dini', 'v', 'dh', 'rmean', 'error'], index=range(size * 7))
    count = 0
    with alive_bar(total=len(main_loop)) as bar:
        for walk_id, mac, device in main_loop:
            bar.text(f'  -->  {walk_id:3d} {mac}')
            df_fit = df.loc[(df['walk_id'] == walk_id) & (df['mac'] == mac) & (df['device'] == device)]

            if not df_fit.empty and df_fit.shape[0] > 5:
                direction = df_fit.iloc[0]['direction']
                df_fit = df_fit[['walk_id', 'direction', 'mac', 'rssi', 'timestamp', 'speed']].sort_values(
                    by=['timestamp'])
                df_fit['device'] = device
                rows = Parallel(n_jobs=nc)(delayed(fit_pl)(df_fit, 1e6, *params) for params in grid_it)
                for row in rows:
                    for r in row:
                        results.iloc[count] = r
                        count += 1
            bar()
    results = results.iloc[:count]
    filename = os.path.join(RESULTS_PATH, 'pathloss_fit_results.csv')
    results.to_csv(filename, index=False)
    print(f'results saved to file {filename}')
