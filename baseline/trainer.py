import os

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from baseline.data import OLIDataModule
from baseline.model import ClassificationModule
from utils import find_best_ckpt


def train_baseline(args, exp_name):
    # init model
    if args.load_from is None:
        model = ClassificationModule(args)
    else:
        if not args.load_from.endswith('.ckpt'):
            args.load_from = find_best_ckpt(args.load_from)

        model = ClassificationModule.load_from_checkpoint(checkpoint_path=args.load_from,
                                                          args=args,
                                                          strict=False)

    if args.freeze_bert:
        for param in model.model.bert.parameters():
            param.requires_grad = False

    # init dataset
    data_dir = os.path.join(args.data_dir, args.lang)

    if isinstance(args.train_file, str):
        train_file = os.path.join(data_dir, args.train_file)
    elif len(args.train_file) == 1:
        train_file = os.path.join(data_dir, args.train_file[0])
    else:
        # support multiple training files
        train_file = args.train_file

    dm = OLIDataModule(
        train_file=train_file,
        val_file=os.path.join(data_dir, args.val_file),
        test_file=os.path.join(data_dir, args.test_file),
        enc_model=args.bert,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_f1',
        filepath=f'logs_{args.seed}/{exp_name}/' + '{epoch}-{val_loss:.3f}-{val_f1:.3f}',
        verbose=True,
        save_last=False,
        mode='max',
        save_top_k=3,
        prefix=f'{args.lang}'
    )

    # tensorboard logger
    logger = pl_loggers.TensorBoardLogger(
        save_dir=f'logs_{args.seed}/',
        name=exp_name,
    )

    # early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='max'
    )

    # train
    if torch.cuda.is_available() and args.device >= 0:
        gpus = [args.device]
    else:
        gpus = None

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stop_callback],
        max_epochs=args.max_epochs,
        gpus=gpus,
        # resume_from_checkpoint=args.load_from,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)
