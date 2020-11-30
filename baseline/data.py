import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from baseline.utils import load_tokenizer
from data_utils.preprocessing import preprocess
from data_utils.utils import SEP_CODE, load_data


class ArgsBase:
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--data_dir',
                            type=str,
                            default='./dataset/',
                            help='dataset directory')

        parser.add_argument('--train_file',
                            type=str,
                            nargs='+',
                            default='train.txt',
                            help='train file')

        parser.add_argument('--val_file',
                            type=str,
                            default='dev.txt',
                            help='validation file')

        parser.add_argument('--test_file',
                            type=str,
                            default='test.txt',
                            help='test file')

        parser.add_argument('--max_seq_len',
                            type=int,
                            default=512,
                            help='')
        return parser


class OLIDataset(Dataset):
    def __init__(self, filepath, enc_model, max_seq_len=512, include_samples=False):
        super().__init__()
        self._label_to_id = {'NOT': 0, 'OFF': 1}
        self._id_to_label = ['NOT', 'OFF']
        self._lang_to_id = {'da': 0, 'en': 1, 'ko': 2, 'da_test': 3}
        self._id_to_lang = ['da', 'en', 'ko', 'da_test']
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.pad_token = '[PAD]'
        self.data = self.load_data(filepath)
        self.enc_model = enc_model
        self.tokenizer = load_tokenizer(enc_model)
        self.max_seq_len = max_seq_len
        self.include_samples = include_samples
        if enc_model == 'kobert':
            self.pad_token = self.tokenizer.vocab.padding_token
            self.pad_id = self.tokenizer.vocab.token_to_idx[self.pad_token]
        else:
            self.pad_token = self.tokenizer.pad_token
            self.pad_id = self.tokenizer._convert_token_to_id(self.pad_token)

    def load_data(self, filepath):
        print(f'Loading data from {filepath}')
        if isinstance(filepath, list):
            dataframes = [load_data(f) for f in filepath]
            data = pd.concat(dataframes, ignore_index=True)
        else:
            data = load_data(filepath)
        data['label'] = data['label'].apply(lambda x: self._label_to_id[x])
        return data

    def tokenize(self, doc):
        if self.enc_model == 'kobert':
            tokens = self.tokenizer(doc)
        else:
            tokens = self.tokenizer.tokenize(doc)
        tokens = [self.cls_token] + tokens + [self.sep_token]
        return tokens

    def __getitem__(self, index):
        record = self.data.iloc[index]
        doc = preprocess(str(record['sample']), record['lang'])
        label = int(record['label'])
        lang = self._lang_to_id[record['lang']]
        tokens = self.tokenize(doc)
        num_tokens = len(tokens)
        num_pad = self.max_seq_len - num_tokens
        attention_mask = [1] * num_tokens

        if num_tokens >= self.max_seq_len:
            tokens = tokens[:self.max_seq_len - 1] + [self.sep_token]
            attention_mask = attention_mask[:self.max_seq_len]
        else:
            tokens = tokens + [self.pad_id] * num_pad
            attention_mask = attention_mask + [0] * num_pad
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        data = {'input_ids': np.array(input_ids, dtype=np.int_),
                'attn_mask': np.array(attention_mask, dtype=np.float),
                'labels': np.array(label, dtype=np.int_),
                'lang': np.array(lang, dtype=np.int_)}
        if self.include_samples:
            data['samples'] = str(record['sample'])
        return data

    def __len__(self):
        return len(self.data)


class OLIDataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, test_file,
                 enc_model, max_seq_len, batch_size):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.enc_model = enc_model
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        return parser

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage=None):
        # split dataset
        if stage == 'fit' or stage is None:
            self.oli_train = OLIDataset(self.train_file,
                                        self.enc_model,
                                        self.max_seq_len)
            self.oli_val = OLIDataset(self.val_file,
                                      self.enc_model,
                                      self.max_seq_len)
        elif stage == 'test' or stage is None:
            self.oli_test = OLIDataset(self.test_file,
                                       self.enc_model,
                                       self.max_seq_len,
                                       include_samples=True)

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.oli_train,
                              batch_size=self.batch_size,
                              num_workers=5, shuffle=True)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.oli_val,
                            batch_size=self.batch_size,
                            num_workers=5, shuffle=False)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.oli_test,
                             batch_size=self.batch_size,
                             num_workers=5, shuffle=False)
        return test_dl

