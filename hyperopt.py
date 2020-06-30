import io
import os
import re
import subprocess
from argparse import ArgumentParser
from datetime import datetime

import joblib
import numpy as np
import optuna
import pandas as pd


class GapsoTrial:
    def __init__(self, dimensions, functions, repeats, working_directory):
        self._dimensions = dimensions
        self._functions = functions
        self._repeats = repeats
        self._working_directory = working_directory
        self._output_directory = datetime.now().strftime('%Y%m%d_%H%M%S')

    def objective(self, trial: optuna.Trial):
        stagnation = trial.suggest_categorical('stagnation', [10, 20, 40])
        collapse = trial.suggest_categorical('collapse', [1e-4, 1e-3, 1e-2])
        # collapse_type = trial.suggest_categorical('collapse_type', ['LINEAR', 'VARIANCE'])
        score = self._run_gapso(stagnation, collapse, trial.number)
        return score

    def _run_gapso(self, stagnation, collapse, trial_number):
        dfs = []
        for dimension, repeats in zip(self._dimensions, self._repeats):
            for function in self._functions:
                num_jobs = self._num_parallel_repeats(dimension)
                errors = joblib.Parallel(n_jobs=num_jobs)(
                    joblib.delayed(self._run_gapso_single_repeat)(
                        dimension, function, stagnation, collapse
                    )
                    for _ in range(repeats)
                )
                errors_padded = np.zeros((14, repeats))
                for i, v in enumerate(errors):
                    errors_padded[:len(v), i] = v
                df = pd.DataFrame(errors_padded)
                filename = os.path.join(
                    self._output_directory,
                    f"trial{trial_number}_stagnation{stagnation}_collapse{collapse}",
                    f"fun{function}_dim{dimension}_rep{repeats}.txt"
                )
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                df.to_csv(filename, sep=' ', index=False, header=False)
                df['evaluations'] = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                df = df.melt(id_vars=['evaluations'], var_name='repeat', value_name='error')
                df['dimensions'] = dimension
                df['function'] = function
                dfs.append(df)
        df = pd.concat(dfs)
        return self._calculate_score(df)

    def _run_gapso_single_repeat(self, dimension, function, stagnation, collapse):
        command = ' '.join([
            f"java -jar {self._working_directory}/gapso-cec2017-experiment-1.0.0.jar",
            f"--cec2017.dimensions {dimension}",
            f"--cec2017.function {function}",
            '--cec2017.repeats 1',
            f"--gapso.restarts.stagnation {stagnation}",
            f"--gapso.restarts.collapse {collapse}"
        ])
        process = subprocess.Popen(command.split(), cwd=self._working_directory, stdout=subprocess.PIPE)
        lines = [line for line in io.TextIOWrapper(process.stdout) if line.startswith('[StdoutCec2017Logger]')]
        return [float(re.findall(r"value: (.*)\n", line)[0]) for line in lines]

    def _calculate_score(self, df: pd.DataFrame):
        coefficients = []
        if 10 in self._dimensions: coefficients.append(0.1)
        if 30 in self._dimensions: coefficients.append(0.2)
        if 50 in self._dimensions: coefficients.append(0.3)
        if 100 in self._dimensions: coefficients.append(0.4)
        sum_of_errors = df.query('function != "2"').query('evaluations == 1.00')
        sum_of_errors = sum_of_errors.groupby(['dimensions', 'function'])['error'].mean()
        sum_of_errors = sum_of_errors.groupby(['dimensions']).sum()
        return np.sum(sum_of_errors * coefficients)

    @staticmethod
    def _num_parallel_repeats(dimensions: int):
        if dimensions == 10:
            return 16
        elif dimensions == 30:
            return 8
        elif dimensions == 50:
            return 8
        else:
            return 4


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dimensions', type=str, default='10,30', help='Comma-separated list of dimensions')
    parser.add_argument('--functions', type=str, default='4,5,6', help='Comma-separated list of functions')
    parser.add_argument('--repeats', type=str, default='16,8', help='Comma-separated number of repeats for corresponding dimension')
    parser.add_argument('--gapso-jar-directory', type=str, default='basic-pso-de-hybrid/cec2017/target', help='Directory with jar and properties')
    args = parser.parse_args()
    args.dimensions = [int(d) for d in args.dimensions.split(',')]
    args.functions = [int(f) for f in args.functions.split(',')]
    args.repeats = [int(r) for r in args.repeats.split(',')]
    return args


if __name__ == '__main__':
    args = parse_args()
    trial = GapsoTrial(args.dimensions, args.functions, args.repeats, args.gapso_jar_directory)
    search_space = {
        'stagnation': [10, 20, 40],
        'collapse': [1e-4, 1e-3, 1e-2]
    }
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(sampler=sampler)
    study.optimize(trial.objective, n_trials=9)
