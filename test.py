import argparse
import os

import pytorch_lightning as pl
import sys

from data import OLIDataModule
from train import ArgsBase
from train import ClassificationModule


def main(args):
    model = ClassificationModule(args)
    model.freeze()
    model.eval()

    # init dataset
    data_dir = os.path.join(args.data_dir, args.lang)

    dm = OLIDataModule(
        train_file=os.path.join(data_dir, args.train_file),
        val_file=os.path.join(data_dir, args.val_file),
        test_file=os.path.join(data_dir, args.test_file),
        enc_model=args.model,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size
    )

    trainer = pl.Trainer(
        gpus=[args.device]
    )

    trainer.test(
        model=model,
        ckpt_path=args.load_from,
        verbose=True,
        datamodule=dm
    )


if __name__ == '__main__':
    sys.path.append(
        os.path.dirname(os.path.abspath(os.path.dirname("__file__")))
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model',
        type=str,
        default='kobert',
        help='pre-trained model to use: bert, kobert, mbert, xlm'
    )

    parser.add_argument(
        '--lang',
        type=str,
        default='da',
        help='task language: da, ko, en'
    )

    parser.add_argument(
        '--exp-name',
        default='',
        type=str,
        help='suffix to specify experiment name'
    )

    parser.add_argument(
        '--device',
        default=0,
        type=int,
    )

    parser.add_argument(
        '--load-from',
        default=None,
        type=str,
        help='path to load model to resume training'
    )

    parser = ArgsBase.add_model_specific_args(parser)
    parser = ClassificationModule.add_model_specific_args(parser)
    parser = OLIDataModule.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
