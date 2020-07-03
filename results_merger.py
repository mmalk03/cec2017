import argparse
import operator
import os
from pathlib import Path
import pandas as pd
from pandas.core.dtypes.missing import array_equivalent
from tabulate import tabulate

from utils import dir_path


def duplicate_columns(frame):
    """
    Source: https://stackoverflow.com/a/32961145/1625856
    """
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:, j].values
                if array_equivalent(ia, ja):
                    dups.append(cs[i])
                    break

    return dups


def get_filename_path_tuple_from_path(paths):
    txt_file_paths = []
    for path in paths:
        txt_file_paths.extend(list(Path(path).rglob('*.txt')))

    return [(str(path).split('/')[-1], path) for path in txt_file_paths]


def merge_results(from_paths, to_path):
    if not os.path.exists(to_path):
        os.makedirs(to_path, exist_ok=True)

    repeats_count = {}

    df = pd.DataFrame(get_filename_path_tuple_from_path(from_paths), columns=['filename', 'path'])
    for filename, row in df.groupby('filename').agg(lambda x: list(x)).iterrows():
        paths = row['path']

        df_local = pd.concat([pd.read_csv(str(path), sep=r"\s+", header=None) for path in paths], axis=1)

        # duplicates = duplicate_columns(df_local)
        # df_local = df_local.drop(duplicates, axis=1)

        df_local.to_csv(os.path.join(to_path, filename), header=False, index=False, sep="\t")
        repeats_count[filename] = df_local.shape[1]

    repeats_count = [[k, v] for k, v in sorted(repeats_count.items(), key=operator.itemgetter(1))]
    print(tabulate(repeats_count, headers=["Filename", "Count"]))


def main():
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument('from-paths', type=dir_path, nargs='+',
                           help='paths from which recursively read .txt files with results')
    arg_parse.add_argument('to-path', type=str,
                           help='path where merged results will be saved')

    args = arg_parse.parse_args()
    args_dict = args.__dict__

    merge_results(args_dict['from-paths'], args_dict['to-path'])


if __name__ == '__main__':
    main()
