import argparse
import os

import pytorch_lightning as pl
import sys
from torch.utils.data import DataLoader

from baseline.data import ArgsBase
from baseline.data import OLIDataModule
from baseline.data import OLIDataset
from baseline.model import ClassificationModule
from adversarial.model import AdversarialTrainingModule
from utils import find_best_ckpt
from utils import generate_output_name, save_prediction


def predict(test_file, args):
    test_dataset = OLIDataset(
        filepath=test_file,
        enc_model=args.bert,
        max_seq_len=args.max_seq_len,
        include_samples=True
    )

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=5, shuffle=False)

    trainer = pl.Trainer(
        gpus=[args.device]
    )

    if args.load_from.endswith('ckpt'):
        best_ckpt = args.load_from
    else:
        best_ckpt = find_best_ckpt(args.load_from, metric='val_f1')

    print(f'Loading model from {best_ckpt}')
    if args.task == 'off':
        model = ClassificationModule.load_from_checkpoint(checkpoint_path=best_ckpt,
                                                          args=args, strict=False)
    elif args.task == 'lang':
        model = AdversarialTrainingModule.load_from_checkpoint(checkpoint_path=best_ckpt,
                                                               args=args)
    model.eval()
    model.freeze()

    results = trainer.test(
        model=model,
        test_dataloaders=test_dataloader,
        verbose=False,
    )

    output_name = generate_output_name(args)
    save_prediction(results, output_name, args)


def main(args):
    # Load test dataset
    data_dir = os.path.join(args.data_dir, args.lang)
    test_file = os.path.join(data_dir, 'test.txt')
    predict(test_file, args)


if __name__ == '__main__':
    sys.path.append(
        os.path.dirname(os.path.abspath(os.path.dirname("__file__")))
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--task',
        type=str,
        default='off',
        help='task to run: off (offensiveness identification) or lang (language prediction)'
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

    parser = ArgsBase.add_model_specific_args(parser)
    parser = ClassificationModule.add_model_specific_args(parser)
    parser = OLIDataModule.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
