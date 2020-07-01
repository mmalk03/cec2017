import argparse
import math
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from utils import dir_path

CACHE_FILE = 'data.pkl'
RESULT_FILENAME_PATTERN = re.compile(r"([\w_-]+)_(\d{1,3})_(\d{1,3})\.(?:txt|csv)")


def calculate_scores(df: pd.DataFrame):
    sum_of_errors = df.query('function != "2"').query('evaluations == 1.00')
    not_complete_results = sum_of_errors[['algorithm', 'dimensions', 'function']].drop_duplicates().groupby('algorithm').count()
    not_complete_results = not_complete_results[not_complete_results != not_complete_results.max()].dropna()
    not_complete_results_algorithms = not_complete_results.index.tolist()

    sum_of_errors = sum_of_errors[~sum_of_errors['algorithm'].isin(not_complete_results_algorithms)]
    sum_of_errors = sum_of_errors.groupby(['algorithm', 'dimensions', 'function'])['error'].mean()
    sum_of_errors = sum_of_errors.groupby(['algorithm', 'dimensions']).sum()
    sum_of_errors = sum_of_errors.groupby('algorithm').aggregate(lambda x: np.sum(x * [0.1, 0.2, 0.3, 0.4]))
    scores_1 = (1 - (sum_of_errors - sum_of_errors.min()) / sum_of_errors) * 50

    sum_of_ranks = df.query('function != "2"').query('evaluations == 1.00')
    sum_of_ranks = sum_of_ranks[~sum_of_ranks['algorithm'].isin(not_complete_results_algorithms)]
    sum_of_ranks = sum_of_ranks.groupby(['algorithm', 'dimensions', 'function'])['error'].mean()
    sum_of_ranks = sum_of_ranks.reset_index()

    dims_ranks = sum_of_ranks.groupby(['algorithm', 'dimensions'])['error'].sum()
    dims_ranks = dims_ranks.reset_index()
    dims_ranks['rank'] = dims_ranks.groupby(['dimensions'])['error'].rank().astype(np.int32)
    dims_ranks = dims_ranks.drop(columns=['error'])
    dims_ranks = dims_ranks.pivot(index='algorithm', columns='dimensions')

    sum_of_ranks['rank'] = sum_of_ranks.groupby(['dimensions', 'function'])['error'].rank()
    sum_of_ranks = sum_of_ranks.groupby(['algorithm', 'dimensions'])['rank'].sum()
    sum_of_ranks = sum_of_ranks.groupby('algorithm').aggregate(lambda x: np.sum(x * [0.1, 0.2, 0.3, 0.4]))
    scores_2 = (1 - (sum_of_ranks - sum_of_ranks.min()) / sum_of_ranks) * 50

    final_score = scores_1 + scores_2
    final_score.name = "score"
    return dims_ranks.join(final_score)


def load_data_frame():
    if os.path.exists(CACHE_FILE):
        return pd.read_pickle(CACHE_FILE)

    directory = 'data'
    results = []
    for paper_id in os.listdir(directory):
        paper_dir = os.path.join(directory, paper_id)
        results.extend(get_results_from_dir(paper_dir, paper_id))
    df = pd.concat(results, ignore_index=True)
    df['error'] = df.apply(lambda row: 0.0 if row['error'] < 1e-8 else row['error'], axis=1)
    df.to_pickle(CACHE_FILE)

    return df


def get_results_from_dir(path, paper_id):
    results = []
    for f in tqdm(os.listdir(path), desc=paper_id):
        matches = re.findall(RESULT_FILENAME_PATTERN, f)
        if paper_id == 'E-17322':  # they messed up filenames
            dimensions = matches[0][1]
            function_name = matches[0][2]
        else:
            function_name = matches[0][1]
            dimensions = matches[0][2]
        absolute_path = os.path.join(path, f)
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
    return results


def plot_specific_comparison(df, function, dimensions, only_save=False, scores=None, only_gapso=False, max_algorithms_per_ax=4):
    query = f"function == '{function}' and dimensions == '{dimensions}'"
    data = df.query(query)

    if scores is not None:
        algorithms = scores.sort_values(ascending=False).index.tolist()
    else:
        algorithms = data['algorithm'].unique().tolist()

    algorithms += list(set(data['algorithm'].unique()) - set(algorithms))

    gapso_algorithms = [algorithm for algorithm in algorithms if "gapso" in algorithm.lower()]
    other_algorithms = [algorithm for algorithm in algorithms if "gapso" not in algorithm.lower()]

    if only_gapso:
        algorithms = gapso_algorithms
        other_algorithms = []

    palette = sns.color_palette(palette="Paired", n_colors=len(other_algorithms))

    other_algorithms_colors = {algorithm: palette[idx % ((len(other_algorithms) // 2) + 1) * 2  + (1 if idx > (len(other_algorithms) // 2) else 0)] for idx, algorithm in enumerate(other_algorithms)}
    gapso_algorithms_colors = {algorithm: (idx * 1. / len(gapso_algorithms), idx * 1.0 / len(gapso_algorithms), idx * 1.0 / len(gapso_algorithms)) for idx, algorithm in enumerate(gapso_algorithms)}

    algorithms_colors = {**other_algorithms_colors, **gapso_algorithms_colors}

    if len(algorithms) > max_algorithms_per_ax:
        ncols = 2
    else:
        ncols = 1

    nrows = math.ceil(math.ceil(len(algorithms) / float(max_algorithms_per_ax)) / float(ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.4 * (1 + 0.5 * (ncols - 1)), 4.8 * (1 + 0.5 * (nrows - 1))))
    axs = np.squeeze(np.reshape(axs, (1, -1)), axis=0)

    for idx in range(axs.shape[0] - ncols):
        axs[idx].set_xticks([])

    max_error = data['error'].quantile(0.70)
    min_error = data['error'].min()
    for idx, other_algorithms_subset in enumerate(np.array_split(other_algorithms, len(axs))):
        algorithms_subset = gapso_algorithms + other_algorithms_subset.tolist()
        sns.lineplot('evaluations', 'error', hue='algorithm',
                     data=data[data["algorithm"].isin(algorithms_subset)],
                     ax=axs[idx],
                     palette=algorithms_colors,
                     hue_order=algorithms_subset,
                     style='algorithm' if len(other_algorithms_colors) == 0 else None)
        # axs[idx].set_xlim(0.01, 0.1)
        axs[idx].set_ylim(min_error, max_error)
        axs[idx].legend(fontsize='x-small', loc=1)
    fig.suptitle(f"function {function}, dimension {dimensions}")

    directory = f"comparison/dim_{dimensions}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, f'fun_{function}_dim_{dimensions}.pdf'), dpi=300)

    if not only_save:
        plt.show()

    plt.close()


def plot_per_dim_comparison(df, dimensions, only_save=False, scores=None, only_gapso=False, max_algorithms_per_ax=4, omit_worse=False):
    query = f"dimensions == '{dimensions}'"
    data = df.query(query)

    if scores is not None:
        algorithms = scores.sort_values(ascending=False).index.tolist()
    else:
        algorithms = data['algorithm'].unique().tolist()

    worse_algorithms = []
    if omit_worse:
        gapso_score = scores['M-GAPSO']
        worse_algorithms = scores[scores < gapso_score].index.tolist()
        algorithms = [algorithm for algorithm in algorithms if algorithm not in worse_algorithms]

    algorithms += list(set(data['algorithm'].unique()) - set(algorithms) - set(worse_algorithms))

    gapso_algorithms = [algorithm for algorithm in algorithms if "gapso" in algorithm.lower()]
    other_algorithms = [algorithm for algorithm in algorithms if "gapso" not in algorithm.lower()]

    if only_gapso:
        algorithms = gapso_algorithms
        other_algorithms = []

    palette = sns.color_palette(palette="Paired", n_colors=len(other_algorithms))

    other_algorithms_colors = {algorithm: palette[
        idx % math.ceil(len(other_algorithms) / 2) * 2 + (1 if idx > (len(other_algorithms) // 2) else 0)] for
                               idx, algorithm in enumerate(other_algorithms)}
    gapso_algorithms_colors = {algorithm: (
    idx * 1. / len(gapso_algorithms), idx * 1.0 / len(gapso_algorithms), idx * 1.0 / len(gapso_algorithms)) for
                               idx, algorithm in enumerate(gapso_algorithms)}

    algorithms_colors = {**other_algorithms_colors, **gapso_algorithms_colors}

    if len(algorithms) > max_algorithms_per_ax:
        ncols = 2
    else:
        ncols = 1

    nrows = math.ceil(math.ceil(len(algorithms) / float(max_algorithms_per_ax)) / float(ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(6.4 * (1 + 0.5 * (ncols - 1)), 4.8 * (1 + 0.5 * (nrows - 1))))
    axs = np.squeeze(np.reshape(axs, (1, -1)), axis=0)

    for idx in range(axs.shape[0] - ncols):
        axs[idx].set_xticks([])

    data = data.groupby(['evaluations', 'algorithm', 'function']).agg({'error': 'mean'}).reset_index()
    data = data.groupby(['evaluations', 'algorithm']).agg({'error': 'mean'}).reset_index()

    max_error = float(data[data['evaluations'] == 0.9].groupby('algorithm')['error'].mean().quantile(0.5, interpolation='linear'))
    min_error = float(data['error'].min())
    for idx, other_algorithms_subset in enumerate(np.array_split(other_algorithms, len(axs))):
        algorithms_subset = gapso_algorithms + other_algorithms_subset.tolist()
        sns.lineplot('evaluations', 'error', hue='algorithm',
                     data=data[data["algorithm"].isin(algorithms_subset)],
                     ax=axs[idx],
                     palette=algorithms_colors,
                     hue_order=algorithms_subset,
                     style='algorithm' if len(other_algorithms_colors) == 0 else None)
        axs[idx].set_ylim(min_error, max_error)
        axs[idx].legend(fontsize='x-small', loc=1)
    fig.suptitle(f"dimension {dimensions}")

    directory = f"comparison_per_dim"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, f'dim_{dimensions}.pdf'), dpi=300)

    if not only_save:
        plt.show()

    plt.close()


def plot_per_fun_comparison(df, function, only_save=False, scores=None, only_gapso=False, max_algorithms_per_ax=4, omit_worse=False):
    query = f"function == '{function}'"
    data = df.query(query)

    if scores is not None:
        algorithms = scores.sort_values(ascending=False).index.tolist()
    else:
        algorithms = data['algorithm'].unique().tolist()

    worse_algorithms = []
    if omit_worse:
        gapso_score = scores['M-GAPSO']
        worse_algorithms = scores[scores < gapso_score].index.tolist()
        algorithms = [algorithm for algorithm in algorithms if algorithm not in worse_algorithms]

    algorithms += list(set(data['algorithm'].unique()) - set(algorithms) - set(worse_algorithms))

    gapso_algorithms = [algorithm for algorithm in algorithms if "gapso" in algorithm.lower()]
    other_algorithms = [algorithm for algorithm in algorithms if "gapso" not in algorithm.lower()]

    if only_gapso:
        algorithms = gapso_algorithms
        other_algorithms = []

    palette = sns.color_palette(palette="Paired", n_colors=len(other_algorithms))

    other_algorithms_colors = {algorithm: palette[
        idx % math.ceil(len(other_algorithms) / 2) * 2 + (1 if idx > (len(other_algorithms) // 2) else 0)] for
                               idx, algorithm in enumerate(other_algorithms)}
    gapso_algorithms_colors = {algorithm: (
    idx * 1. / len(gapso_algorithms), idx * 1.0 / len(gapso_algorithms), idx * 1.0 / len(gapso_algorithms)) for
                               idx, algorithm in enumerate(gapso_algorithms)}

    algorithms_colors = {**other_algorithms_colors, **gapso_algorithms_colors}

    if len(algorithms) > max_algorithms_per_ax:
        ncols = 2
    else:
        ncols = 1

    nrows = math.ceil(math.ceil(len(algorithms) / float(max_algorithms_per_ax)) / float(ncols))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(6.4 * (1 + 0.5 * (ncols - 1)), 4.8 * (1 + 0.5 * (nrows - 1))))
    axs = np.squeeze(np.reshape(axs, (1, -1)), axis=0)

    for idx in range(axs.shape[0] - ncols):
        axs[idx].set_xticks([])

    data = data.groupby(['evaluations', 'algorithm', 'dimensions']).agg({'error': 'mean'}).reset_index()
    data = data.groupby(['evaluations', 'algorithm']).agg({'error': 'mean'}).reset_index()

    max_error = float(data[data['evaluations'] == 0.9].groupby('algorithm')['error'].mean().quantile(0.5, interpolation='linear'))
    min_error = float(data['error'].min())
    for idx, other_algorithms_subset in enumerate(np.array_split(other_algorithms, len(axs))):
        algorithms_subset = gapso_algorithms + other_algorithms_subset.tolist()
        sns.lineplot('evaluations', 'error', hue='algorithm',
                     data=data[data["algorithm"].isin(algorithms_subset)],
                     ax=axs[idx],
                     palette=algorithms_colors,
                     hue_order=algorithms_subset,
                     style='algorithm' if len(other_algorithms_colors) == 0 else None)
        axs[idx].set_ylim(min_error, max_error)
        axs[idx].legend(fontsize='x-small', loc=1)
    fig.suptitle(f"function {function}")

    directory = f"comparison_per_function"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, f'fun_{function}.pdf'), dpi=300)

    if not only_save:
        plt.show()

    plt.close()


def plot_overall_comparison(df):
    g = sns.FacetGrid(df, row='function', col='dimensions', hue='algorithm')
    g.map(sns.lineplot, 'evaluations', 'error')
    plt.show()
    plt.savefig('comparison.pdf', dpi=300)


def append_logs_online(df, additional_results_paths, rename_additional_results_algorithms):
    if len(rename_additional_results_algorithms) != len(additional_results_paths):
        rename_additional_results_algorithms = [None] * len(additional_results_paths)

    results = []
    for (path, algorithm) in zip(additional_results_paths, rename_additional_results_algorithms):
        results_local = get_results_from_dir(path, "M-GAPSO")

        if algorithm is not None:
            for df_result in results_local:
                df_result['algorithm'] = algorithm

        results.extend(results_local)
    return df.append(pd.concat(results, ignore_index=True))


def main():
    sns.set_style('whitegrid')

    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('--functions', type=int, nargs='+',
                           help='functions for which do plots',
                           default=[1] + list(range(3, 31)))
    arg_parse.add_argument('--dimensions', type=int, nargs='+',
                           help='dimensions for which do plots',
                           default=[10, 30, 50, 100])
    arg_parse.add_argument('--without-score-print',
                           action='store_true',
                           help='do not print overall score',
                           default=False)
    arg_parse.add_argument('--with-specific-comparison',
                           action='store_true',
                           help='save and show specific comparison plots',
                           default=False)
    arg_parse.add_argument('--with-per-dim-comparison',
                           action='store_true',
                           help='save and show per dim comparison plots',
                           default=False)
    arg_parse.add_argument('--with-per-fun-comparison',
                           action='store_true',
                           help='save and show per fun comparison plots',
                           default=False)
    arg_parse.add_argument('--omit-worse',
                           action='store_true',
                           help='in plots omit algorithms which are worse than GAPSO',
                           default=False)
    arg_parse.add_argument('--only-save',
                           action='store_true',
                           help='do not show plot, only save as .pdf',
                           default=False)
    arg_parse.add_argument('--only-gapso',
                           action='store_true',
                           help='show only gapso algorithms',
                           default=False)
    arg_parse.add_argument('--additional-results-paths', type=dir_path, nargs='+',
                           help='paths from which recursively read .txt files with results (beside data/ directory)')
    arg_parse.add_argument('--rename-additional-results-algorithms', type=str, nargs='+',
                           help='names for algorithms additionally loaded (beside data/ directory)')

    args = arg_parse.parse_args()

    df = load_data_frame()
    if args.additional_results_paths:
        df = append_logs_online(df, args.additional_results_paths, args.rename_additional_results_algorithms)

    scores = calculate_scores(df)
    if not args.without_score_print:
        print(scores.sort_values(by=['score'], ascending=False))

    if not args.without_score_print:
        df['function'] = pd.to_numeric(df['function'])
        functions = {
            'unimodal': [1, 2],
            'multidmodal': list(range(3, 10)),
            'hybrid': list(range(10, 20)),
            'composition': list(range(20, 30))
        }
        for function_type, function_names in functions.items():
            scores = calculate_scores(df.query(f"function in {function_names}"))
            print(f"Ranking for {function_type} functions:")
            print(f"{scores.sort_values(by=['score'], ascending=False)}")

    if args.with_specific_comparison:
        for function in tqdm(args.functions, position=0):
            for dimensions in tqdm(args.dimensions, position=1):
                plot_specific_comparison(df, function, dimensions, only_save=args.only_save, scores=scores, only_gapso=args.only_gapso)

    if args.with_per_dim_comparison:
        for dimensions in tqdm(args.dimensions):
            plot_per_dim_comparison(df, dimensions, only_save=args.only_save, scores=scores, only_gapso=args.only_gapso, omit_worse=args.omit_worse)

    if args.with_per_fun_comparison:
        for function in tqdm(args.functions):
            plot_per_fun_comparison(df, function, only_save=args.only_save, scores=scores, only_gapso=args.only_gapso, omit_worse=args.omit_worse)

    # plot_overall_comparison(df)


if __name__ == '__main__':
    main()
