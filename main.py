import os
import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

result_filename_pattern = re.compile(r"([\w_-]+)_(\d{1,3})_(\d{1,3})\.txt")
directory = 'data'
df = pd.DataFrame(columns=['paper_id', 'algorithm', 'function', 'dimensions', 'absolute_path'])
for paper_id in os.listdir(directory):
    paper_dir = os.path.join(directory, paper_id)
    for f in tqdm(os.listdir(paper_dir)):
        matches = re.findall(result_filename_pattern, f)
        if paper_id == 'E-17322':  # they messed up filenames
            dimensions = matches[0][1]
            function_name = matches[0][2]
        else:
            function_name = matches[0][1]
            dimensions = matches[0][2]
        absolute_path = os.path.join(paper_dir, f)
        result_table = pd.read_table(absolute_path, header=None, sep='\s+').transpose().reset_index()
        result_table.columns = ['run'] + [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result_table = result_table.melt(id_vars=['run'], var_name='evaluations', value_name='error')
        result_table['evaluations'] = pd.to_numeric(result_table['evaluations'])
        result_table['paper_id'] = paper_id
        result_table['algorithm'] = matches[0][0]
        result_table['function'] = function_name
        result_table['dimensions'] = dimensions
        result_table['absolute_path'] = absolute_path
        df = df.append(result_table, ignore_index=True)

sns.set_style('whitegrid')
for function in ['1', '2', '3']:
    f = sns.lineplot('evaluations', 'error', hue='algorithm', data=df.query(f"function == '{function}' and dimensions == '10'"))
    f.axes.set_xlim(0.01, 0.1)
    plt.show()

g = sns.FacetGrid(df, row='function', col='dimensions', hue='algorithm')
g = g.map(sns.lineplot, 'evaluations', 'error')
g.fig.savefig('comparison.pdf')
