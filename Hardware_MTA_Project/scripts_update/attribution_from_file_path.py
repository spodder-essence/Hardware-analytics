

import os
import sys
import argparse
import logging
from datetime import date

import pandas as pd

from markovchain.attribution import Attribution, Markov

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
#parser.add_argument('.\MarkovChainAttr_22783112_placementid_path_BQ', help='path to file')
#parser.add_argument('path', choices=['node','path'], help='format of data')
#parser.add_argument('linear', nargs='+', choices=['markov','linear','first_touch','last_touch','any_touch'], help='type of attribution model to use')

parser.add_argument('file_path', help='path to file')
parser.add_argument('file_type', choices=['node','path'], help='format of data')
parser.add_argument('attr', nargs='+', choices=['markov','linear','first_touch','last_touch','any_touch'], help='type of attribution model to use')
args=parser.parse_args()


def main():

    models = args.attr
    file_type = args.file_type
    if ('markov' not in models or len(models) > 1) and file_type == 'node':
        raise Exception("In order to run attribution models other than markov, data format must be path")
    
    results = {}
    if file_type == 'node':
        columns=['start_node','end_node','count']
        df = pd.read_csv(args.file_path,skiprows=1,names=columns)
        attr = Markov(df, start_col='start_node', end_col='end_node', count_col='count')
        logging.info('Running markov attribution')
        results['markov'] = attr.attribute_conversions()
    elif file_type == 'path':
        columns=['path','conv','uniques']
        df = pd.read_csv(args.file_path,skiprows=1,names=columns)
        attr = Attribution(df, path_col='path', conv_col='conv', uniques_col='uniques')
        for model in models:
            logging.info(f'Running {model} attribution')
            method = getattr(attr, model)
            results[model] = method()

    return results

if __name__ == '__main__':
    
    file_path = args.file_path
    date = date.today().strftime("%Y-%m-%d")
    results = main()
    for name, df in results.items():
        logging.info(f'Saving {name} results to csv.')
        df.to_csv(f"{name}_{date}.csv",index=False)