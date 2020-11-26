import argparse
import pandas as pd
from data_utils.utils import load_data, save_data

LANGUAGES = ['da', 'en', 'ko']


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', type=str, nargs='+')
    p.add_argument('-o', '--output', type=str, default='result.txt')

    args = p.parse_args()
    dataframes = []

    for input_fname in args.input:
        df = load_data(input_fname)
        dataframes.append(df)

    result = pd.concat(dataframes, ignore_index=True)
    out_fname = args.output
    save_data(df=result, fname=out_fname)
