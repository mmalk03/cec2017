import os
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import optuna


class GapsoTrial:
    def __init__(self, dimensions, functions, repeats):
        self._dimensions = dimensions
        self._functions = functions
        self._repeats = repeats

    def objective(self, trial: optuna.Trial):
        stagnation = trial.suggest_categorical('stagnation', [10, 20, 40])
        collapse = trial.suggest_categorical('collapse', [1e-4, 1e-3, 1e-2])
        # collapse_type = trial.suggest_categorical('collapse_type', ['LINEAR', 'VARIANCE'])
        results_dir = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._run_gapso(stagnation, collapse, results_dir)
        # TODO: read results and calculate score similarly as in main.py
        multiplier = trial.suggest_int('multiplier', 4, 32, 4)
        return np.sum(self._dimensions) * multiplier + np.sum(self._functions) * multiplier

    def _run_gapso(self, stagnation, collapse, results_dir):
        """ Runs GAPSO appropriate number of times using queue.sh script and waits till it finishes """
        command = ' '.join([
            './queue.sh',
            f"--dimensions {self._dimensions}",
            f"--functions {self._functions}",
            f"--repeats {self._repeats}",
            f"--stagnation {stagnation}",
            f"--collapse {collapse}",
            f"--results-dir {results_dir}",
        ])
        os.system(command)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dimensions', type=str, default='10,30', help='Comma-separated list of dimensions')
    parser.add_argument('--functions', type=str, default='4,5,6', help='Comma-separated list of functions')
    parser.add_argument('--repeats', type=int, default=10, help='Number of repeats for each function and dimension pair')
    args = parser.parse_args([])
    args.dimensions = [int(d) for d in args.dimensions.split(',')]
    args.functions = [int(f) for f in args.functions.split(',')]
    return args


if __name__ == '__main__':
    args = parse_args()
    trial = GapsoTrial(args.dimensions, args.functions)
    search_space = {'multiplier': [4, 8, 16, 20, 24, 28, 32]}
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(sampler=sampler)
    study.optimize(trial.objective, n_trials=7)

# TODO:
# - add collapse and stagnation command line arguments to Java runner
# - add results directory command line argument to Java runner, which specifies a directory where CEC2017 logs should be saved
# - modify queue.sh such that it doesn't run all functions, dimensions and repeats in parallel
