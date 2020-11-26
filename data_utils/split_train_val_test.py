import argparse
from sklearn.model_selection import train_test_split
import os
from data_utils.utils import load_data, save_data


def train_dev_test_split(df, dev_size, test_size, args):
    train, test = train_test_split(df,
                                   test_size=dev_size + test_size,
                                   stratify=df['label'].values,
                                   random_state=args.seed)
    dev, test = train_test_split(test,
                                 test_size=test_size,
                                 stratify=test['label'].values,
                                 random_state=args.seed)
    return train, dev, test


def save_dataset(train, dev, test, lang, out_dir):
    out_dir = os.path.join(out_dir, lang)
    for ftype, df in {'train': train, 'dev': dev, 'test': test}.items():
        out_fname = os.path.join(out_dir, f'{ftype}.txt')
        save_data(df=df, fname=out_fname)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=123)
    args = p.parse_args()
    size = {
        'ko': {
            'dev': 600,
            'test': 600
        },
        'en': {
            'dev': 1000,
            'test': 1000
        },
        'da': {
            'dev': 200,
            'test': 200
        }
    }
    for lang, sizes in size.items():
        dataset = load_data(f'../dataset/original/{lang}.txt')

        train, dev, test = train_dev_test_split(df=dataset,
                                                dev_size=sizes['dev'],
                                                test_size=sizes['test'],
                                                args=args)
        out_dir = f'../dataset_{args.seed}'
        save_dataset(train, dev, test, lang, out_dir)
