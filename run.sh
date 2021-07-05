#!/bin/bash

cd src || exit
python path_loss_fit.py
python compute_errors.py
python stats.py
python stats_backtracking.py
python plot_stats.py