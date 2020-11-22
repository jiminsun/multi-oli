import argparse
import os

import torch.nn as nn
import pytorch_lightning as pl
import sys
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from setproctitle import setproctitle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers.optimization import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from data import ArgsBase
from data import OLIDataModule
from utils import generate_exp_name
from model import Classifier


class BaseModule(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super().__init__()
        self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        # model architecture
        parser.add_argument('--timepool', type=str, default='mean', help='mean, max, both')
        # training
        parser.add_argument('--lr', type=float, default=5e-5, help='The initial learning rate')
        parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
        parser.add_argument('--max_grad_norm', type=float, default=1.0, help='gradient clipping')
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--max_epochs', type=int, default=20)
        return parser

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)
        # warm up lr
        num_train_steps = len(self.train_dataloader()) * \
                          self.hparams.max_epochs
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 'name': 'cosine_schedule_with_warmup',
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]


class ClassificationModule(BaseModule):
    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)
        self.model = Classifier(hparams)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.metric_acc = pl.metrics.classification.Accuracy()
        self.train_step = 0

    def training_step(self, batch, batch_idx):
        logits = self.model(batch['input_ids'],
                            batch['attention_mask'])
        loss = self.criterion(input=logits,
                              target=batch['label'])
        self.log('train_loss', loss,
                 on_step=True, on_epoch=False, logger=True, prog_bar=True)
        self.train_step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch['input_ids'],
                            batch['attention_mask'])
        # Compute validation loss
        loss = self.criterion(input=logits,
                              target=batch['label'])
        # self.log('val_loss', loss,
        #          on_step=True, on_epoch=True, logger=True, prog_bar=False)
        # self.logger.log_metrics(metrics={'val_loss_step': loss},
        #                         step=self.train_step)
        # Compute accuracy
        y_pred = logits.argmax(dim=-1)
        # accuracy = self.metric_acc(preds=preds,
        #                            target=batch['label'])
        # Compute metrics
        y_true = batch['label'].cpu().tolist()
        y_pred = y_pred.cpu().tolist()
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        # self.log('val_acc', accuracy,
        #          on_step=False, on_epoch=True, logger=True, prog_bar=False)
        # self.log('val_f1', f1,
        #          on_step=False, on_epoch=True, logger=True, prog_bar=False)
        metrics = {'loss': loss.clone().detach().item(),
                   'acc': accuracy,
                   'y_true': y_true,
                   'y_pred': y_pred}
        return metrics

    def validation_epoch_end(self, outputs):
        losses = []
        y_true = []
        y_pred = []

        for output in outputs:
            losses.append(output['loss'])
            y_true += output['y_true']
            y_pred += output['y_pred']

        avg_val_loss = sum(losses) / len(losses)
        avg_val_acc = accuracy_score(y_true=y_true, y_pred=y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=1,
            average='macro'
        )
        metrics = {
            'val_loss': avg_val_loss,
            'val_accuracy': avg_val_acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
        }

        print(metrics)
        self.logger.log_metrics(metrics, step=self.train_step)

        metrics['y_pred'] = y_pred

        print("\n")
        print("#" * 30)
        print(f"Dev  loss      | {avg_val_loss:.5f}")
        print(f"     accuracy  | {avg_val_acc:.3f}")
        print(f"     precision | {precision:.3f}")
        print(f"     recall    | {recall:.3f}")
        print(f"     f1 score  | {f1:.3f}")
        print("#" * 30)
        print("\n")
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        return metrics

    def test_epoch_end(self, outputs):
        y_true = []
        y_pred = []

        for output in outputs:
            y_true += output['y_true']
            y_pred += output['y_pred']

        test_acc = accuracy_score(y_true=y_true, y_pred=y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            pos_label=1,
            average='macro'
        )
        metrics = {'test_accuracy': test_acc,
                   'test_precision': precision,
                   'test_recall': recall,
                   'test_f1': f1}

        print("\n")
        print("#" * 30)
        print(f"Test accuracy  | {test_acc:.3f}")
        print(f"     precision | {precision:.3f}")
        print(f"     recall    | {recall:.3f}")
        print(f"     f1 score  | {f1:.3f}")
        print("#" * 30)
        print("\n")
        return metrics


def main(args):
    # fix random seeds for reproducibility
    SEED = 123
    pl.seed_everything(SEED)

    # generate experiment name
    exp_name = generate_exp_name(args)
    # set process title to exp name
    setproctitle(exp_name)

    # init model
    model = ClassificationModule(args)

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

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filepath=f'logs/{exp_name}/' + '{epoch}-{val_loss:.3f}-{val_f1:.3f}',
        verbose=True,
        save_last=True,
        mode='min',
        save_top_k=5,
        prefix=f'{args.lang}_{args.model}'
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
        patience=5,
        verbose=False,
        mode='min'
    )

    # train
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[early_stop_callback],
        max_epochs=args.max_epochs,
        gpus=[args.device],
        resume_from_checkpoint=args.load_from,
        checkpoint_callback=checkpoint_callback,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)

    # test
    trainer.test(ckpt_path='best',
                 datamodule=dm)


if __name__ == "__main__":
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

    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)

    parser = ArgsBase.add_model_specific_args(parser)
    parser = ClassificationModule.add_model_specific_args(parser)
    parser = OLIDataModule.add_model_specific_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    # logging.info(args)

    main(args)
