import pandas as pd
from data_utils.utils import save_data


if __name__ == "__main__":
    ko_train = pd.read_csv('../../data/korean-hate-speech/labeled/train.tsv', sep='\t')
    ko_dev = pd.read_csv('../../data/korean-hate-speech/labeled/dev.tsv', sep='\t')
    ko_dataset = pd.concat([ko_train, ko_dev], ignore_index=True)

    en_dataset = pd.read_csv('../../data/offenseval/en/olid-training-v1.0.tsv', sep='\t')
    da_dataset = pd.read_csv('../../data/offenseval/da/offenseval-da-training-v1.tsv', sep='\t')

    # match kor --> eng
    # slice relevant columns
    ko_dataset = ko_dataset[['comments', 'hate']]
    # slice relevant labels (we're not considering `hate`)
    print(f'Number of samples **before** filtering `hate`: {len(ko_dataset)}')
    ko_dataset = ko_dataset[ko_dataset['hate'] != 'hate']
    print(f'Number of samples **after** filtering `hate`: {len(ko_dataset)}')

    # reset indices
    ko_dataset.reset_index(drop=True, inplace=True)
    # make an idx column
    ko_dataset.reset_index(inplace=True)

    # match column names
    ko_dataset.columns = ['id', 'sample', 'label']

    # match label names
    label_mapping = {'none': 'NOT', 'offensive': 'OFF'}
    ko_dataset['label'] = ko_dataset['label'].apply(lambda x: label_mapping[x])

    # match column names for english as well
    en_dataset.columns = ['id', 'sample', 'label']

    # danish as well
    da_dataset.columns = ['id', 'sample', 'label']

    ko_dataset = ko_dataset[['sample', 'label']]
    ko_dataset.index.name = 'id'

    en_dataset = en_dataset[['sample', 'label']]
    en_dataset.index.name = 'id'

    ko_dataset['lang'] = 'ko'
    en_dataset['lang'] = 'en'
    da_dataset['lang'] = 'da'

    save_data(ko_dataset, '../dataset/original/ko.txt')
    save_data(en_dataset, '../dataset/original/en.txt')
    save_data(da_dataset, '../dataset/original/da.txt')