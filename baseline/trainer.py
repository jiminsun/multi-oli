import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from baseline.data import OLIDataModule
from baseline.model import ClassificationModule


def train_baseline(args, exp_name):
    # init model
    if args.load_from is None:
        model = ClassificationModule(args)
    else:
        model = ClassificationModule.load_from_checkpoint(checkpoint_path=args.load_from,
                                                          args=args,
                                                          strict=False)

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
        monitor='val_loss',
        filepath=f'logs/{exp_name}/' + '{epoch}-{val_loss:.3f}-{val_f1:.3f}',
        verbose=True,
        save_last=False,
        mode='min',
        save_top_k=3,
        prefix=f'{args.lang}'
    )

    # tensorboard logger
    logger = pl_loggers.TensorBoardLogger(
        save_dir='logs/',
        name=exp_name,
    )

    # early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )

    # train
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stop_callback],
        max_epochs=args.max_epochs,
        gpus=[args.device],
        # resume_from_checkpoint=args.load_from,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)
