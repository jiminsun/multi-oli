import os

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from adversarial.data import AdversarialLearningDataModule
from adversarial.model import AdversarialTrainingModule


def train_adversarial(args, exp_name):
    # init model
    model = AdversarialTrainingModule(args)

    # init dataset
    en_data_dir = os.path.join(args.data_dir, 'en')
    non_en_data_dir = os.path.join(args.data_dir, args.lang)
    concat_data_dir = os.path.join(args.data_dir, f'en_{args.lang}')

    train_dir = concat_data_dir if args.train_with_both else en_data_dir
    val_dir = en_data_dir if (args.val_with == 'en') else non_en_data_dir

    dm = AdversarialLearningDataModule(
        en_train_file=os.path.join(train_dir, args.train_file),
        non_en_train_file=os.path.join(non_en_data_dir, args.train_file),
        val_file=os.path.join(val_dir, args.val_file),
        test_file=os.path.join(non_en_data_dir, args.test_file),
        enc_model=args.bert,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=f'logs_{args.seed}/{exp_name}/' + '{epoch}-{val_loss:.3f}-{val_f1:.3f}',
        verbose=True,
        save_last=False,
        mode='min',
        save_top_k=5,
        prefix=f'{args.lang}_{args.bert}'
    )

    # tensorboard logger
    logger = pl_loggers.TensorBoardLogger(
        save_dir=f'logs_{args.seed}/',
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
    if torch.cuda.is_available() and args.device >= 0:
        gpus = [args.device]
    else:
        gpus = None

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stop_callback],
        automatic_optimization=False,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        gpus=gpus,
        resume_from_checkpoint=args.load_from,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=10,
        # weights_summary='full'
    )
    trainer.fit(model, dm)
