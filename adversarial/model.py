import argparse

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from baseline.model import ClassificationModule
from baseline.pooling import MeanPool
from baseline.utils import load_pretrained


class AdversarialTrainingModule(ClassificationModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.bert = load_pretrained(hparams.bert)
        self.pool = MeanPool()
        self.out = nn.ModuleDict({
            'offensive': nn.Linear(self.bert.config.hidden_size, 2, bias=True),
            'language': nn.Linear(self.bert.config.hidden_size, 1, bias=True)
        })
        self.criterion = nn.ModuleDict({
            'offensive': nn.CrossEntropyLoss(reduction='mean'),
            'language': nn.BCEWithLogitsLoss(reduction='mean')
        })
        self.train_step = 0

    def forward(self, input_ids, attention_mask, task='offensive'):
        last_hidden_state, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.pool.forward(
            last_hidden_state=last_hidden_state,
            attention_mask=attention_mask
        )
        logits = self.out[task](output)
        return logits

    def task_loss(self, logits, target):
        """ Computes task (offensiveness classification) loss (L_T)

        Args:
            logits: logit scores from language classification module
            target: language labels (EN: 1, NON-EN: 0)

        Returns:
            loss: scalar value
        """
        loss = self.criterion['offensive'](input=logits,
                                           target=target)
        return loss

    def generator_loss(self, logits, target):
        """ Computes generator loss (L_G)

        Args:
            logits: logit scores from language classification module
            target: language labels (EN: 1, NON-EN: 0)

        Returns:
            loss: scalar value
        """
        # map label 1 to 0, 0 to 1
        target = (target == 0).type_as(target)
        loss = self.criterion['language'](input=logits,
                                          target=target)
        return loss

    def discriminator_loss(self, logits, target):
        """ Computes discriminator loss (L_G)

        Args:
            logits: logit scores from language classification module
            target: language labels (EN: 1, NON-EN: 0)

        Returns:
            loss: scalar value
        """
        loss = self.criterion['language'](input=logits,
                                          target=target)
        return loss

    def task_step(self, input_ids, attn_mask, labels):
        logits = self(input_ids=input_ids,
                      attention_mask=attn_mask,
                      task='offensive')
        task_loss = self.task_loss(logits=logits, target=labels)
        return task_loss

    def generator_step(self, en_input_ids, en_attn_mask, non_en_input_ids, non_en_attn_mask):
        en_logits = self.forward(input_ids=en_input_ids,
                                 attention_mask=en_attn_mask,
                                 task='language')

        non_en_logits = self.forward(input_ids=non_en_input_ids,
                                     attention_mask=non_en_attn_mask,
                                     task='language')
        en_labels = torch.ones_like(en_logits)
        non_en_labels = torch.zeros_like(non_en_logits)
        logits = torch.stack([en_logits, non_en_logits], dim=-1)
        labels = torch.stack([en_labels, non_en_labels], dim=-1)
        g_loss = self.generator_loss(logits=logits, target=labels)
        return g_loss

    def discriminator_step(self, en_input_ids, en_attn_mask, non_en_input_ids, non_en_attn_mask):
        en_logits = self.forward(input_ids=en_input_ids,
                                 attention_mask=en_attn_mask,
                                 task='language')
        non_en_logits = self.forward(input_ids=non_en_input_ids,
                                     attention_mask=non_en_attn_mask,
                                     task='language')
        en_labels = torch.ones_like(en_logits)
        non_en_labels = torch.zeros_like(non_en_logits)
        logits = torch.stack([en_logits, non_en_logits], dim=-1)
        labels = torch.stack([en_labels, non_en_labels], dim=-1)
        d_loss = self.discriminator_loss(logits=logits, target=labels)
        return d_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        opt_task, opt_g, opt_d = self.optimizers()

        # 1. train w.r.t. task loss
        task_loss = self.task_step(**batch['task'])
        # self.log('task_loss', task_loss)
        self.manual_backward(task_loss, opt_task)
        opt_task.step()
        opt_task.zero_grad()

        # 2. train w.r.t. generator loss
        gen_loss = self.generator_step(**batch['generator'])
        # self.log('gen_loss', gen_loss)
        self.manual_backward(gen_loss, opt_g)
        opt_g.step()
        opt_g.zero_grad()

        # train w.r.t. discriminator loss
        disc_loss = self.discriminator_step(**batch['discriminator'])
        # self.log('disc_loss', disc_loss)
        self.manual_backward(disc_loss, opt_d)
        opt_d.step()
        opt_d.zero_grad()

        # log metrics
        self.logger.log_metrics(
            metrics={'task_loss': task_loss,
                     'gen_loss': gen_loss,
                     'disc_loss': disc_loss},
            step=self.train_step)
        self.train_step += 1

    def validation_step(self, batch, batch_idx):
        logits = self.forward(input_ids=batch['input_ids'],
                              attention_mask=batch['attn_mask'],
                              task='offensive')
        loss = self.task_loss(logits=logits, target=batch['labels'])
        y_pred = logits.argmax(dim=-1)

        # Compute metrics
        y_true = batch['labels'].cpu().tolist()
        y_pred = y_pred.cpu().tolist()
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        metrics = {'loss': loss.clone().detach().item(),
                   'acc': accuracy,
                   'y_true': y_true,
                   'y_pred': y_pred}
        self.log('val_loss', loss)
        return metrics

    def test_step(self, batch, batch_idx):
        logits = self.forward(input_ids=batch['input_ids'],
                              attention_mask=batch['attn_mask'],
                              task='offensive')
        loss = self.task_loss(logits=logits, target=batch['labels'])
        y_pred = logits.argmax(dim=-1)
        # Compute metrics
        y_true = batch['labels'].cpu().tolist()
        y_pred = y_pred.cpu().tolist()
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        metrics = {'loss': loss.clone().detach().item(),
                   'acc': accuracy,
                   'y_true': y_true,
                   'y_pred': y_pred}
        self.log('test_loss', loss)
        return metrics

    def configure_optimizers(self):
        lr_task = self.hparams.task_lr
        lr_g = self.hparams.gen_lr
        lr_d = self.hparams.disc_lr
        opt_params = list(self.bert.parameters()) + list(self.out['offensive'].parameters())
        opt_task = torch.optim.Adam(opt_params, lr=lr_task)
        opt_g = torch.optim.Adam(self.bert.parameters(), lr=lr_g)
        opt_d = torch.optim.Adam(self.out['language'].parameters(), lr=lr_d)
        return [opt_task, opt_g, opt_d], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--task-lr', type=float, default=2e-6, help="task loss learning rate")
        parser.add_argument('--gen-lr', type=float, default=2e-8, help="generator loss learning rate")
        parser.add_argument('--disc-lr', type=float, default=5e-5, help="discriminator loss learning rate")
        return parser