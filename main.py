import argparse
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

CACHE_FILE = 'data.pkl'


def calculate_scores(df: pd.DataFrame):
    sum_of_errors = df.query('function != "2"').query('evaluations == 1.00')
    sum_of_errors = sum_of_errors.groupby(['algorithm', 'dimensions', 'function'])['error'].mean()
    sum_of_errors = sum_of_errors.groupby(['algorithm', 'dimensions']).sum()
    sum_of_errors = sum_of_errors.groupby('algorithm').aggregate(lambda x: np.sum(x * [0.1, 0.2, 0.3, 0.4]))
    scores_1 = (1 - (sum_of_errors - sum_of_errors.min()) / sum_of_errors) * 50

    sum_of_ranks = df.query('function != "2"').query('evaluations == 1.00')
    sum_of_ranks = sum_of_ranks.groupby(['algorithm', 'dimensions', 'function'])['error'].mean()
    sum_of_ranks = sum_of_ranks.reset_index()
    sum_of_ranks['rank'] = sum_of_ranks.groupby(['dimensions', 'function'])['error'].rank()
    sum_of_ranks = sum_of_ranks.groupby(['algorithm', 'dimensions'])['rank'].sum()
    sum_of_ranks = sum_of_ranks.groupby('algorithm').aggregate(lambda x: np.sum(x * [0.1, 0.2, 0.3, 0.4]))
    scores_2 = (1 - (sum_of_ranks - sum_of_ranks.min()) / sum_of_ranks) * 50

    return scores_1 + scores_2


def load_data_frame():
    if os.path.exists(CACHE_FILE):
        return pd.read_pickle(CACHE_FILE)

    result_filename_pattern = re.compile(r"([\w_-]+)_(\d{1,3})_(\d{1,3})\.(?:txt|csv)")
    directory = 'data'
    results = []
    for paper_id in os.listdir(directory):
        paper_dir = os.path.join(directory, paper_id)
        for f in tqdm(os.listdir(paper_dir), desc=paper_id):
            matches = re.findall(result_filename_pattern, f)
            if paper_id == 'E-17322':  # they messed up filenames
                dimensions = matches[0][1]
                function_name = matches[0][2]
            else:
                function_name = matches[0][1]
                dimensions = matches[0][2]
            absolute_path = os.path.join(paper_dir, f)
            if paper_id == '17420':  # they messed up separator
                result_table = pd.read_table(absolute_path, header=None, sep=',')
            elif paper_id == 'E-17260':  # they messed up everything
                result_table = pd.read_table(absolute_path, skiprows=1, header=None, sep='\s?,\s?').drop(0, axis=1)
                result_table.columns = list(range(len(result_table.columns)))
            else:
                result_table = pd.read_table(absolute_path, header=None, sep='\s+')
            result_table = result_table.transpose().reset_index()
            result_table.columns = ['run'] + [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            result_table = result_table.melt(id_vars=['run'], var_name='evaluations', value_name='error')
            result_table['evaluations'] = pd.to_numeric(result_table['evaluations'])
            result_table['paper_id'] = paper_id
            result_table['algorithm'] = matches[0][0]
            result_table['function'] = function_name
            result_table['dimensions'] = int(dimensions)
            result_table['absolute_path'] = absolute_path
            results += [result_table]
    df = pd.concat(results, ignore_index=True)
    df['error'] = df.apply(lambda row: 0.0 if row['error'] < 1e-8 else row['error'], axis=1)
    df.to_pickle(CACHE_FILE)


def plot_specific_comparison(df, function, dimensions):
    query = f"function == '{function}' and dimensions == '{dimensions}'"
    fig, ax = plt.subplots()
    sns.lineplot('evaluations', 'error', hue='algorithm',
                 data=df.query(query),
                 ax=ax)
    ax.set_xlim(0.01, 0.1)
    ax.set_title(query)
    plt.savefig(f'comparison_fun_{function}_dim_{dimensions}.pdf', dpi=300)
    plt.show()


def plot_overall_comparison(df):
    g = sns.FacetGrid(df, row='function', col='dimensions', hue='algorithm')
    g.map(sns.lineplot, 'evaluations', 'error')
    plt.show()
    plt.savefig('comparison.pdf', dpi=300)


def append_logs_from_gapso(df, gapso_logs_path):
    # for filename in glob(os.path.join(gapso_logs_path, ""))
    pass


def main():
    sns.set_style('whitegrid')

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--functions', type=int, nargs='+',
                           help='functions for which do plots',
                           default=[1] + list(range(3, 31)))
    arg_parse.add_argument('--dimensions', type=int, nargs='+',
                           help='dimensions for which do plots',
                           default=[10, 30, 50, 100])
    arg_parse.add_argument('--gapso-logs-path', type=str,
                           help='path to GAPSO files for parsing',
                           default=None)

    args = arg_parse.parse_args()

    df = load_data_frame()
    if args.gapso_logs_path:
        append_logs_from_gapso(df, args.gapso_logs_path)

    scores = calculate_scores(df)
    print(scores.sort_values(ascending=False))

    for function in args.functions:
        for dimensions in args.dimensions:
            plot_specific_comparison(df, function, dimensions)

    # plot_overall_comparison(df)


if __name__ == '__main__':
    main()
