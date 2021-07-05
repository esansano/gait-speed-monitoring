import os
import itertools
import numpy as np
import pandas as pd
from alive_progress import alive_bar
from data_manager import RESULTS_PATH


def stats():
    # walk_id,direction,device,speed,combination,len,model,rssi0_min,rssi0_max,gamma_min,gamma_max,rmean,pred_mean,
    # pred_median,error_mean,error_median
    rdf = pd.read_csv(os.path.join(RESULTS_PATH, 'stats_errors_results.csv'))
    rdf = rdf.drop(['rssi0_min', 'rssi0_max', 'gamma_min', 'gamma_max', 'pred_median', 'error_median'], axis=1)
    rmeans = pd.unique(rdf['rmean']).tolist()
    devices = pd.unique(rdf['device']).tolist()
    walk_ids = pd.unique(rdf['walk_id']).tolist()
    lens = list(range(1, 10))
    models = [0, 1]

    df = pd.DataFrame(columns=['device', 'rmean', 'length', 'model', 'error', 'std', 'srate', 'q75', 'q90'])
    df = df.astype(dtype={'device': str, 'rmean': int, 'length': int, 'model': int, 'error': float,
                          'std': float, 'srate': float, 'q75': float, 'q90': float})

    grid = list(itertools.product(models, lens, rmeans))

    with alive_bar(total=len(grid)) as bar:
        for count, (model, length, rmean) in enumerate(grid):
            mlr_df = rdf.loc[(rdf['model'] == model) & (rdf['len'] == length) & (rdf['rmean'] == rmean)]
            all_errors = []
            all_walks = set()
            device_walks = {}
            srates = []
            for device in devices:
                mlrd_df = mlr_df.loc[(mlr_df['device'] == device)]
                errors = []
                n_walks = 0
                device_walks[device] = {}
                for walk_id in walk_ids:
                    mlrdw_df = mlrd_df.loc[(mlrd_df['walk_id'] == walk_id)]
                    if not mlrdw_df.empty:
                        errors.extend(mlrdw_df['error_mean'].tolist())
                        all_walks.add(walk_id)
                        n_walks += 1
                        device_walks[device][walk_id] = mlrdw_df['error_mean'].tolist()
                if len(errors) > 0:
                    error = np.mean(errors)
                    q75 = np.percentile(errors, 75)
                    q90 = np.percentile(errors, 90)
                    std = np.std(errors)
                    srate = n_walks / len(walk_ids)
                    srates.append(srate)
                    df = df.append({'device': device, 'rmean': rmean, 'length': length, 'model': model, 'error': error,
                                    'std': std, 'srate': srate, 'q75': q75, 'q90': q90}, ignore_index=True)

                all_errors.extend(errors)
            if len(all_errors) > 0:
                # error = np.mean(all_errors)
                # q75 = np.percentile(all_errors, 75)
                # q90 = np.percentile(all_errors, 90)
                # std = np.std(all_errors)
                # srate = len(all_walks) / len(walk_ids)
                # df = df.append({'device': 'all', 'rmean': rmean, 'length': length, 'model': model, 'error': error,
                #                 'std': std, 'srate': srate, 'q75': q75, 'q90': q90}, ignore_index=True)
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
                    # srate = len(comp_walks) / len(walk_ids)
                    srate = len(all_walks) / len(walk_ids)
                    # srate = np.mean(srates)
                    df = df.append({'device': 'all', 'rmean': rmean, 'length': length, 'model': model, 'error': error,
                                    'std': std, 'srate': srate, 'q75': q75, 'q90': q90}, ignore_index=True)

            bar.text(f'{100 * count/len(grid):6.2f}% ')
            bar()

    df.to_csv(os.path.join(RESULTS_PATH, 'stats.csv'), index=False)


if __name__ == '__main__':
    stats()
