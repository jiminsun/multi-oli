import argparse
from data_utils.utils import load_data, save_data
from sklearn.model_selection import train_test_split
import os


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', type=str)
    p.add_argument('-n', '--num_samples', type=int, default=None)
    p.add_argument('-r', '--ratio', type=float, default=None)
    p.add_argument('--seed', type=int, default=123)

    args = p.parse_args()
    train_size = args.num_samples if args.num_samples > 0 else args.ratio
    df = load_data(args.input)

    for train_size in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]:
        train, _ = train_test_split(df, train_size=train_size,
                                    random_state=args.seed,
                                    stratify=df['label'])
        path, ext = os.path.splitext(args.input)
        out_fname = path + f'_{train_size}' + ext
        save_data(df=train, fname=out_fname)
