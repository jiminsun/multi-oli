import argparse
import os

import pytorch_lightning as pl
import sys
import torch

from adversarial.data import AdversarialLearningDataModule
from adversarial.model import AdversarialTrainingModule
from baseline.data import ArgsBase
from baseline.data import OLIDataModule
from baseline.data import OLIDataset
from baseline.model import ClassificationModule
from plot_utils.tsne import TSNE
from utils import generate_exp_name


def main(args):
    # fix random seeds for reproducibility
    SEED = 123
    pl.seed_everything(SEED)
    # generate experiment name
    exp_name = generate_exp_name(args)
    # Set device
    device = torch.device(f'cuda:{args.device}') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    if args.load_from is None:
        model = ClassificationModule(args)
    else:
        model = ClassificationModule.load_from_checkpoint(
            checkpoint_path=args.load_from,
            args=args,
            strict=False
        )
    model.eval()
    model.freeze()

    # Load dataset
    dataset = OLIDataset(
        filepath=args.input_file,
        enc_model=args.bert
    )

    tsne_plotter = TSNE(model, dataset, device, exp_name)
    tsne_plotter.visualize()


if __name__ == "__main__":
    sys.path.append(
        os.path.dirname(os.path.abspath(os.path.dirname("__file__")))
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task',
        type=str,
        default='plot'
    )

    parser.add_argument(
        '--input_file',
        type=str,
        nargs='+'
    )

    parser.add_argument(
        '--load_from',
        type=str
    )

    parser.add_argument(
        '--bert',
        type=str,
        default='mbert'
    )

    parser.add_argument(
        '--exp_name',
        type=str,
        default=''
    )

    parser.add_argument(
        '--device',
        type=int,
        default=0
    )
    parser = ArgsBase.add_model_specific_args(parser)
    parser = ClassificationModule.add_model_specific_args(parser)
    parser = AdversarialTrainingModule.add_model_specific_args(parser)
    parser = OLIDataModule.add_model_specific_args(parser)
    parser = AdversarialLearningDataModule.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)
