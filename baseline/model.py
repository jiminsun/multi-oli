import argparse
import pytorch_lightning as pl
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from transformers.optimization import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from baseline.pooling import MeanPool
from baseline.utils import load_pretrained


class BertClassifier(nn.Module):
    """
    Binary offensiveness classification model built on top of pre-trained encoder,
    followed by mean pooling over timesteps & linear layer.

    B : batch size
    E : embedding size
    L : max sequence length in batch

    """
    def __init__(self, args):
        super().__init__()
        self.bert = load_pretrained(args.bert)
        self.pool = MeanPool()
        self.out = nn.Linear(self.bert.config.hidden_size, 2, bias=True)

    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: input mapped to token ids                [B x L]
            attention_mask: mask to avoid attn on `[PAD]`       [B x L]
                            `[PAD]` mapped to 0, otherwise 1.

        Returns:
            logits: prediction scores                           [B x C]
        """
        last_hidden_state, _ = self.bert(input_ids=input_ids,
                                         attention_mask=attention_mask)     # [B x L x E]
        x = self.pool.forward(last_hidden_state=last_hidden_state,          # [B x E]
                              attention_mask=attention_mask)
        logits = self.out(x)                                                # [B x C]
        return logits


class BaseModule(pl.LightningModule):
    def __init__(self, hparams, **kwargs) -> None:
        super().__init__()
        self.hparams = hparams

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        # training
        parser.add_argument('--lr', type=float, default=5e-5, help='The initial learning rate')
        parser.add_argument('--warmup_ratio', type=float, default=0.1, help='warmup ratio')
        parser.add_argument('--max_grad_norm', type=float, default=1.0, help='gradient clipping')
        parser.add_argument('--batch_size', type=int, default=16)
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
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.model = BertClassifier(args)
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        self.metric_acc = pl.metrics.classification.Accuracy()
        self.train_step = 0

    def training_step(self, batch, batch_idx):
        logits = self.model(batch['input_ids'],
                            batch['attn_mask'])
        loss = self.criterion(input=logits,
                              target=batch['labels'])
        self.log('train_loss', loss,
                 on_step=True, on_epoch=False, logger=True, prog_bar=True)
        self.train_step += 1
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.model(batch['input_ids'],
                            batch['attn_mask'])
        # Compute validation loss
        loss = self.criterion(input=logits,
                              target=batch['labels'])
        # Compute accuracy
        y_pred = logits.argmax(dim=-1)
        # Compute metrics
        y_true = batch['labels'].cpu().tolist()
        y_pred = y_pred.cpu().tolist()
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
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

        print("#" * 30)
        print(f"Test accuracy  | {test_acc:.3f}")
        print(f"     precision | {precision:.3f}")
        print(f"     recall    | {recall:.3f}")
        print(f"     f1 score  | {f1:.3f}")
        print("#" * 30)
        print("\n")
        return metrics
