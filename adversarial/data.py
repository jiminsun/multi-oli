import argparse

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import RandomSampler

from baseline.data import OLIDataset


class AdversarialLearningDataset(Dataset):
    def __init__(self, en_file, non_en_file, enc_model, max_seq_len=512):
        super().__init__()
        self._en_dataset = OLIDataset(en_file, enc_model, max_seq_len)
        self._non_en_dataset = OLIDataset(non_en_file, enc_model, max_seq_len)
        self.en_loader = self.init_loader(self._en_dataset)
        self.non_en_loader = self.init_loader(self._non_en_dataset)

    @staticmethod
    def init_loader(dataset):
        sampler = RandomSampler(data_source=dataset)
        loader = DataLoader(
            dataset=dataset,
            sampler=sampler
        )
        return loader

    def __getitem__(self, index):
        task_en_sample = next(iter(self.en_loader))
        task_input_ids = task_en_sample['input_ids']
        task_attn_mask = task_en_sample['attn_mask']
        task_labels = task_en_sample['labels']

        # Samples for generator training
        g_en_sample = next(iter(self.en_loader))
        g_en_input_ids = g_en_sample['input_ids']
        g_en_attn_mask = g_en_sample['attn_mask']

        g_non_en_sample = next(iter(self.en_loader))
        g_non_en_input_ids = g_non_en_sample['input_ids']
        g_non_en_attn_mask = g_non_en_sample['attn_mask']

        # Samples for discriminator training
        d_en_sample = next(iter(self.en_loader))
        d_en_input_ids = d_en_sample['input_ids']
        d_en_attn_mask = d_en_sample['attn_mask']

        d_non_en_sample = next(iter(self.en_loader))
        d_non_en_input_ids = d_non_en_sample['input_ids']
        d_non_en_attn_mask = d_non_en_sample['attn_mask']

        return {
            'task': {
                'input_ids': task_input_ids,
                'attn_mask': task_attn_mask,
                'labels': task_labels
            },
            'generator': {
                'en_input_ids': g_en_input_ids,
                'en_attn_mask': g_en_attn_mask,
                'non_en_input_ids': g_non_en_input_ids,
                'non_en_attn_mask': g_non_en_attn_mask
            },
            'discriminator': {
                'en_input_ids': d_en_input_ids,
                'en_attn_mask': d_en_attn_mask,
                'non_en_input_ids': d_non_en_input_ids,
                'non_en_attn_mask': d_non_en_attn_mask
            }
        }

    def __len__(self):
        return len(self._non_en_dataset)

    @staticmethod
    def collate_fn(batch):
        task_input_ids = torch.stack([item['task']['input_ids'] for item in batch]).squeeze(1)
        task_attn_mask = torch.stack([item['task']['attn_mask'] for item in batch]).squeeze(1)
        task_labels = torch.stack([item['task']['labels'] for item in batch]).squeeze(1)

        g_en_input_ids = torch.stack([item['generator']['en_input_ids'] for item in batch]).squeeze(1)
        g_en_attn_mask = torch.stack([item['generator']['en_attn_mask'] for item in batch]).squeeze(1)
        g_non_en_input_ids = torch.stack([item['generator']['non_en_input_ids'] for item in batch]).squeeze(1)
        g_non_en_attn_mask = torch.stack([item['generator']['non_en_attn_mask'] for item in batch]).squeeze(1)

        d_en_input_ids = torch.stack([item['discriminator']['en_input_ids'] for item in batch]).squeeze(1)
        d_en_attn_mask = torch.stack([item['discriminator']['en_attn_mask'] for item in batch]).squeeze(1)
        d_non_en_input_ids = torch.stack([item['discriminator']['non_en_input_ids'] for item in batch]).squeeze(1)
        d_non_en_attn_mask = torch.stack([item['discriminator']['non_en_attn_mask'] for item in batch]).squeeze(1)

        return {
            'task': {
                'input_ids': task_input_ids,
                'attn_mask': task_attn_mask,
                'labels': task_labels
            },
            'generator': {
                'en_input_ids': g_en_input_ids,
                'en_attn_mask': g_en_attn_mask,
                'non_en_input_ids': g_non_en_input_ids,
                'non_en_attn_mask': g_non_en_attn_mask
            },
            'discriminator': {
                'en_input_ids': d_en_input_ids,
                'en_attn_mask': d_en_attn_mask,
                'non_en_input_ids': d_non_en_input_ids,
                'non_en_attn_mask': d_non_en_attn_mask
            }
        }


class AdversarialLearningDataModule(pl.LightningDataModule):
    def __init__(self, en_train_file, non_en_train_file, val_file, test_file,
                 enc_model, max_seq_len, batch_size):
        super().__init__()
        self.en_train_file = en_train_file
        self.non_en_train_file = non_en_train_file
        # Suppose non-English dev set is unavailable
        self.val_file = val_file
        # Test is still for non-English task
        self.test_file = test_file
        self.enc_model = enc_model
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--train_with_both',
                            action='store_true',
                            default=False,
                            help='if true, uses the concatenated training dataset (en+da)')

        parser.add_argument('--val_with',
                            type=str,
                            default='en',
                            help='english (en) or target language (e.g. danish -> da)')

        return parser

    def setup(self, stage=None):
        # split dataset
        if stage == 'fit' or stage is None:
            print('** Loading train **')
            self.train = AdversarialLearningDataset(
                en_file=self.en_train_file,
                non_en_file=self.non_en_train_file,
                enc_model=self.enc_model,
                max_seq_len=self.max_seq_len,
            )
            print('** Loading dev **')
            self.val = OLIDataset(
                filepath=self.val_file,
                enc_model=self.enc_model,
                max_seq_len=self.max_seq_len
            )
        elif stage == 'test' or stage is None:
            print('** Loading test **')
            self.test = OLIDataset(
                filepath=self.test_file,
                enc_model=self.enc_model,
                max_seq_len=self.max_seq_len,
                include_samples=True
            )

    # return the dataloader for each split
    def train_dataloader(self):
        train_dl = DataLoader(self.train,
                              batch_size=self.batch_size,
                              num_workers=5, shuffle=True,
                              collate_fn=AdversarialLearningDataset.collate_fn)
        return train_dl

    def val_dataloader(self):
        val_dl = DataLoader(self.val,
                            batch_size=self.batch_size,
                            num_workers=5, shuffle=False)
        return val_dl

    def test_dataloader(self):
        test_dl = DataLoader(self.test,
                             batch_size=self.batch_size,
                             num_workers=5, shuffle=False)
        return test_dl
