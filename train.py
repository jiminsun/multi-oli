import argparse
import os

import pytorch_lightning as pl
import sys

from adversarial.data import AdversarialLearningDataModule
from adversarial.model import AdversarialTrainingModule
from adversarial.trainer import train_adversarial
from baseline.data import ArgsBase
from baseline.data import OLIDataModule
from baseline.model import ClassificationModule
from baseline.trainer import train_baseline
from utils import generate_exp_name


def main(args):
    # fix random seeds for reproducibility
    SEED = args.seed
    pl.seed_everything(SEED)
    # generate experiment name
    exp_name = generate_exp_name(args)

    if args.task == 'base':
        train_baseline(args, exp_name)
    elif args.task == 'adv':
        print(f'Running adversarial learning')
        train_adversarial(args, exp_name)


if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.abspath(os.path.dirname("__file__")))
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--task',
        type=str,
        default='base',
        help='task to run: base (baseline) or adv (adversarial)'
    )

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
        '--exp_name',
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
        '--max_epochs',
        type=int,
        default=20
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=123
    )

    parser.add_argument(
        '--freeze_bert',
        action='store_true',
        default=False
    )

    parser = ArgsBase.add_model_specific_args(parser)
    parser = ClassificationModule.add_model_specific_args(parser)
    parser = AdversarialTrainingModule.add_model_specific_args(parser)
    parser = OLIDataModule.add_model_specific_args(parser)
    parser = AdversarialLearningDataModule.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
