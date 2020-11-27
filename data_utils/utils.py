import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

SEP_CODE = "\u241E"


def load_data(fname):
    sep = '\t' if fname.endswith('.tsv') else SEP_CODE
    df = pd.read_csv(fname, sep=sep, index_col=0, encoding='utf-8')
    return df


def save_data(df, fname):
    out_dir, _ = os.path.splitext(fname)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.index.name = 'id'
    df.to_csv(fname, sep=SEP_CODE, encoding='utf-8')
    print(f'Output saved as {fname}')
    return None