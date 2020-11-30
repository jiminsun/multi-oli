import argparse

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

from baseline.model import ClassificationModule


class AdversarialTrainingModule(ClassificationModule):
    def __init__(self, args):
        super().__init__(args)
        self.language_out = nn.Linear(self.model.bert.config.hidden_size, 1, bias=True)
        self.language_criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, input_ids, attention_mask):
        # performs offensiveness classification
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return logits

    def pred_language(self, input_ids, attention_mask):
        # performs language classification
        outputs = self.model.bert(input_ids=input_ids,
                                  attention_mask=attention_mask)  # [B x L x E]
        last_hidden_state = outputs[0]
        x = self.model.pool.forward(last_hidden_state=last_hidden_state,  # [B x E]
                                    attention_mask=attention_mask)
        logits = self.language_out(x)  # [B x C]
        return logits

    def task_loss(self, logits, target):
        """ Computes task (offensiveness classification) loss (L_T)

        Args:
            logits: logit scores from language classification module
            target: language labels (EN: 1, NON-EN: 0)

        Returns:
            loss: scalar value
        """
        loss = self.criterion(input=logits,
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
        loss = self.language_criterion(input=logits,
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
        loss = self.language_criterion(input=logits,
                                       target=target)
        return loss

    def task_step(self, input_ids, attn_mask, labels, **kwargs):
        logits = self(input_ids=input_ids,
                      attention_mask=attn_mask)
        task_loss = self.task_loss(logits=logits, target=labels)
        return task_loss

    def generator_step(self, en_input_ids, en_attn_mask, non_en_input_ids, non_en_attn_mask):
        en_logits = self.pred_language(input_ids=en_input_ids,
                                       attention_mask=en_attn_mask)
        non_en_logits = self.pred_language(input_ids=non_en_input_ids,
                                           attention_mask=non_en_attn_mask)
        en_labels = torch.ones_like(en_logits)
        non_en_labels = torch.zeros_like(non_en_logits)
        logits = torch.stack([en_logits, non_en_logits], dim=-1)
        labels = torch.stack([en_labels, non_en_labels], dim=-1)
        g_loss = self.generator_loss(logits=logits, target=labels)
        return g_loss

    def discriminator_step(self, en_input_ids, en_attn_mask, non_en_input_ids, non_en_attn_mask):
        en_logits = self.pred_language(input_ids=en_input_ids,
                                       attention_mask=en_attn_mask)
        non_en_logits = self.pred_language(input_ids=non_en_input_ids,
                                           attention_mask=non_en_attn_mask)
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
        self.manual_backward(task_loss, opt_task)
        opt_task.step()
        opt_task.zero_grad()

        # 2. train w.r.t. generator loss
        gen_loss = self.generator_step(**batch['generator'])
        self.manual_backward(gen_loss, opt_g)
        opt_g.step()
        opt_g.zero_grad()

        # train w.r.t. discriminator loss
        disc_loss = self.discriminator_step(**batch['discriminator'])
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
                              attention_mask=batch['attn_mask'])
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
        language_logits = self.pred_language(input_ids=batch['input_ids'],
                                             attention_mask=batch['attn_mask'])
        task_logits = self.forward(input_ids=batch['input_ids'],
                                   attention_mask=batch['attn_mask'])
        task_pred = task_logits.argmax(dim=-1)
        output = {
            'samples': batch['samples'],
            'lang_logits': language_logits.cpu().tolist(),
            'y_true': batch['labels'].cpu().tolist(),
            'y_pred': task_pred.cpu().tolist()
        }
        return output

    def test_epoch_end(self, outputs):
        metrics = super().test_epoch_end(outputs)
        samples = []
        lang_logits = []

        for output in outputs:
            samples += output['samples']
            lang_logits += output['lang_logits']

        metrics['samples'] = samples
        metrics['lang_logits'] = lang_logits
        return metrics


    def configure_optimizers(self):
        lr_task = self.hparams.task_lr
        lr_g = self.hparams.gen_lr
        lr_d = self.hparams.disc_lr
        opt_task = torch.optim.Adam(self.model.parameters(), lr=lr_task)
        opt_g = torch.optim.Adam(self.model.bert.parameters(), lr=lr_g)
        opt_d = torch.optim.Adam(self.language_out.parameters(), lr=lr_d)
        return [opt_task, opt_g, opt_d], []

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--task_lr', type=float, default=2e-6, help="task loss learning rate")
        parser.add_argument('--gen_lr', type=float, default=2e-8, help="generator loss learning rate")
        parser.add_argument('--disc_lr', type=float, default=5e-5, help="discriminator loss learning rate")
        return parser
