import argparse
from data_utils.utils import load_data, save_data
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', type=str)
    p.add_argument('-o', '--output', type=str, default=None)
    p.add_argument('-n', '--num_samples', type=int, default=None)
    p.add_argument('-r', '--ratio', type=float, default=None)
    p.add_argument('--seed', type=int, default=123)

    args = p.parse_args()
    df = load_data(args.input)

    if args.num_samples is None and args.ratio is None:
        for train_size in [5, 10, 20, 50, 100, 200, 500, 1000, 2000]:
            train, _ = train_test_split(df, train_size=train_size,
                                        random_state=args.seed,
                                        stratify=df['label'])
            path, ext = os.path.splitext(args.input)
            out_fname = path + f'_{train_size}' + ext
            save_data(df=train, fname=out_fname)
    else:
        train_size = args.num_samples if args.num_samples >= 1 else args.ratio
        train, _ = train_test_split(df, train_size=train_size,
                                    random_state=args.seed,
                                    stratify=df['label'])
        if args.output is None:
            path, ext = os.path.splitext(args.input)
            out_fname = path + f'_{train_size}' + ext
        else:
            out_fname = args.output
        save_data(df=train, fname=out_fname)
