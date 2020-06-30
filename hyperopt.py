import io
import re
import subprocess
from argparse import ArgumentParser

import joblib
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler


class GapsoTrial:
    def __init__(self, dimensions, functions, repeats, working_directory):
        self._dimensions = dimensions
        self._functions = functions
        self._repeats = repeats
        self._working_directory = working_directory

    def objective(self, trial: optuna.Trial):
        args = {
            "gapso.restarts.stagnation": trial.suggest_categorical('stagnation', [0, 10, 20, 40]),
            "gapso.restarts.collapse": trial.suggest_categorical('collapse_type', ['LINEAR', 'VARIANCE']),
            "gapso.restarts.collapse.type": trial.suggest_loguniform('collapse', 1e-8, 1e-1),
            "gapso.population.multiplier": trial.suggest_int('population_multiplier', 1, 25),
            "gapso.samples.memory": trial.suggest_categorical('samples_memory', [1, 10, 100, 1000, 10000, 100000]),
            "gapso.history.length": trial.suggest_categorical('history_length', [1, 10, 50, 100, 1000]),
            "gapso.exploration.weight.random.point.exploration": trial.suggest_uniform("random_point_exploration", 0.0, 1.0)
            if trial.suggest_categorical('random_point_exploration_enabled', [True, False]) else -1.0,
            "gapso.exploration.weight.best.point.exploration": trial.suggest_uniform("best_point_exploration", 0.0, 1.0)
            if trial.suggest_categorical('best_point_exploration_enabled', [True, False]) else -1.0,
            "gapso.exploration.weight.difference.point.exploration": trial.suggest_uniform("difference_point_exploration", 0.0, 1.0)
            if trial.suggest_categorical('difference_point_exploration_enabled', [True, False]) else -1.0,
            "gapso.exploration.weight.complete.bounds.reset": trial.suggest_uniform("complete_bounds_reset", 0.0, 1.0)
            if trial.suggest_categorical('complete_bounds_reset_enabled', [True, False]) else -1.0,
            "gapso.exploration.weight.roulette": trial.suggest_uniform("roulette", 0.0, 1.0)
            if trial.suggest_categorical('roulette_enabled', [True, False]) else -1.0,
            "gapso.exploration.weight.value.heuristic": trial.suggest_uniform("value_heuristic", 0.0, 1.0)
            if trial.suggest_categorical('value_heuristic_enabled', [True, False]) else -1.0,
            "gapso.exploration.weight.model.heuristic": trial.suggest_uniform("model_heuristic", 0.0, 1.0)
            if trial.suggest_categorical('model_heuristic_enabled', [True, False]) else -1.0,
            "gapso.exploration.weight.ea.swapping": trial.suggest_uniform("ea_swapping", 0.0, 1.0)
            if trial.suggest_categorical('ea_swapping_enabled', [True, False]) else -1.0,
        }

        score = self._run_gapso(args)
        return score

    def _run_gapso(self, args_dict):
        dfs = []
        for dimension, repeats in zip(self._dimensions, self._repeats):
            for function in self._functions:
                num_jobs = self._num_parallel_repeats(dimension)
                errors = joblib.Parallel(n_jobs=num_jobs)(
                    joblib.delayed(self._run_gapso_single_repeat)(
                        {**args_dict, **{
                            'cec2017.function': function,
                            'cec2017.dimensions': dimension
                        }}
                    )
                    for _ in range(repeats)
                )
                errors_padded = np.zeros((14, repeats))
                for i, v in enumerate(errors):
                    errors_padded[:len(v), i] = v
                df = pd.DataFrame(errors_padded)
                df['evaluations'] = [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                df = df.melt(id_vars=['evaluations'], var_name='repeat', value_name='error')
                df['dimensions'] = dimension
                df['function'] = function
                dfs.append(df)
        df = pd.concat(dfs)
        return self._calculate_score(df)

    def _run_gapso_single_repeat(self, args_dict):
        args = []
        for key, value in args_dict.items():
            args.append(f"--{key}={value}")

        args.append('--cec2017.repeats 1')
        command = ' '.join([
            f"java -jar {self._working_directory}/gapso-cec2017-experiment-1.0.0.jar",
        ] + args)

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
        return float(np.sum(sum_of_errors * coefficients))

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
    args.dimensions = [int(d) for d in str(args.dimensions).split(',')]
    args.functions = [int(f) for f in str(args.functions).split(',')]
    args.repeats = [int(r) for r in str(args.repeats).split(',')]
    return args


def main():
    args = parse_args()
    trial = GapsoTrial(args.dimensions, args.functions, args.repeats, args.gapso_jar_directory)
    sampler = TPESampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(trial.objective)


if __name__ == '__main__':
    main()
