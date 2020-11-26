import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=pd.errors.ParserWarning)

SEP_CODE = "\u241E"


def load_data(fname):
    df = pd.read_csv(fname, sep=SEP_CODE, index_col=0)
    return df


def save_data(df, fname):
    out_dir, _ = os.path.splitext(fname)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    df.to_csv(fname, sep=SEP_CODE)
    print(f'Output saved as {fname}')
    return None