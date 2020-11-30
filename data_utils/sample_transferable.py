import argparse
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from data_utils.utils import load_data
from data_utils.utils import save_data


def sample_subset(df, num_samples, how='top'):
    # Sort samples low -> high english score
    df = df.sort_values(by='logit', ascending=True)
    if how == 'top':
        subset = df.head(n=num_samples)
    elif how == 'bottom':
        subset = df.tail(n=num_samples)
    elif how == 'random':
        subset = df.sample(n=num_samples)
    print(f'Sample for {how} {num_samples}')
    print(subset.head())
    return subset


def compute_label_n(df, n):
    # Compute subset size for each label category
    train, _ = train_test_split(df, train_size=n, stratify=df['label'])
    cnt = train['label'].value_counts()
    return dict(cnt)


def sampling(df, num_samples, how, stratify=False):
    if stratify:
        n_for_each_label = compute_label_n(df, num_samples)
        outputs = []
        for label, n_for_label in n_for_each_label.items():
            output = sample_subset(df=df[df['label'] == label],
                                   num_samples=n_for_label,
                                   how=how)
            outputs.append(output)
        result = pd.concat(outputs, ignore_index=True)
    else:
        result = sample_subset(df, num_samples, how)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', type=str)
    p.add_argument('-o', '--out_dir', type=str, default='../dataset/en_transferable/')
    p.add_argument('-n', '--num_samples', type=int, default=1000)
    p.add_argument('--how', type=str, default=None)
    p.add_argument('--stratify', action='store_true', default=False)
    args = p.parse_args()

    df = load_data(args.input)

    if args.num_samples == 0:
        for n in [25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, len(df)]:
            subset = sampling(df, num_samples=n, how=args.how)
            out_fname = f'../dataset/en_transferable/{args.how}/train_{n}.txt'
            save_data(subset, out_fname)
    else:
        if args.how is None:
            for how in ['top', 'bottom', 'random']:
                subset = sampling(df=df, num_samples=args.num_samples, how=how, stratify=args.stratify)
                fname = f'{how}_{args.num_samples}.txt'
                if args.stratify:
                    fname = 'str_' + fname
                out_fname = os.path.join(args.out_dir, fname)
                save_data(subset, out_fname)
        else:
            subset = sampling(df=df, num_samples=args.num_samples, how=args.how, stratify=args.stratify)
            fname = f'{args.how}_{args.num_samples}.txt'
            if args.stratify:
                fname = 'str_' + fname
            out_fname = os.path.join(args.out_dir, fname)
            save_data(subset, out_fname)