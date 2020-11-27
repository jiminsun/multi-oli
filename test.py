import argparse
import os

import pytorch_lightning as pl
import sys
from torch.utils.data import DataLoader

from baseline.data import ArgsBase
from baseline.data import OLIDataModule
from baseline.data import OLIDataset
from baseline.model import ClassificationModule
from utils import find_best_ckpt


def test(test_file, args):
    test_dataset = OLIDataset(
        filepath=test_file,
        enc_model=args.bert,
        max_seq_len=args.max_seq_len
    )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=5, shuffle=False)

    if args.load_from.endswith('ckpt'):
        print(f'Loaded model from {args.load_from}')
        task_model = ClassificationModule.load_from_checkpoint(checkpoint_path=args.load_from,
                                                               args=args, strict=False)
    else:
        best_ckpt = find_best_ckpt(args.load_from, metric=f'val_{args.best}')
        print(f'Loaded model from {best_ckpt}')
        task_model = ClassificationModule.load_from_checkpoint(checkpoint_path=best_ckpt,
                                                               args=args, strict=False)

    task_model.eval()
    task_model.freeze()

    trainer = pl.Trainer(
        gpus=[args.device]
    )

    trainer.test(
        model=task_model,
        test_dataloaders=test_dataloader,
        # specifying ckpt here doesn't work, resulting in extremely low performance
        # ckpt_path=None,
        verbose=False,
    )


def main(args):

    # Load validation dataset
    data_dir = os.path.join(args.data_dir, args.lang)
    test_file = os.path.join(data_dir, args.val_file)
    test(test_file, args)

    # Load test dataset
    test_file = os.path.join(data_dir, args.test_file)
    test(test_file, args)


if __name__ == '__main__':
    sys.path.append(
        os.path.dirname(os.path.abspath(os.path.dirname("__file__")))
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--bert',
        type=str,
        default='mbert',
        help='pre-trained model to use: bert, kobert, mbert, xlm'
    )

    parser.add_argument(
        '--lang',
        type=str,
        default='da',
        help='task language: da, ko, en'
    )

    parser.add_argument(
        '--device',
        default=0,
        type=int,
    )

    parser.add_argument(
        '--load_from',
        default=None,
        type=str,
        help='path to load model to resume training'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16
    )

    parser.add_argument(
        '--best',
        type=str,
        default='f1'
    )

    parser = ArgsBase.add_model_specific_args(parser)
    parser = ClassificationModule.add_model_specific_args(parser)
    parser = OLIDataModule.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
