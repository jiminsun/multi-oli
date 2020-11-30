import os
import re
from datetime import datetime
from setproctitle import setproctitle

import pandas as pd

TASK_IDX_TO_LABEL = {0: 'NOT', 1: 'OFF'}
LAND_IDX_TO_LABEL = {0: 'da', 1: 'en'}


def generate_exp_name(args):
    try:
        exp_name = f'{args.lang}/{args.task}/{args.bert}/'
    except AttributeError:
        exp_name = f'logs/{args.task}/'
    if len(args.exp_name):
        exp_name = exp_name + f'{args.exp_name}'
    else:
        now = datetime.now().strftime("%m-%d-%H:%M")
        exp_name += now
    setproctitle(exp_name)
    print(exp_name)
    return exp_name


def parse_score(ckpt_path, metric='val_f1'):
    score = re.search(metric + '=[0-9]+\.[0-9]+', ckpt_path).group()
    score = float(re.search('[0-9]+\.[0-9]+', score).group())
    return score


def find_best_ckpt(fpath, metric='val_f1'):
    ckpts = [os.path.join(fpath, c) for c in os.listdir(fpath) if metric in c]
    ckpts = [(ckpt, parse_score(ckpt, metric)) for ckpt in ckpts]
    if metric == 'val_f1':
        # higest f1 score
        best_ckpt = sorted(ckpts, key=lambda x: x[1])[-1][0]
    elif metric == 'val_loss':
        # smallest loss
        best_ckpt = sorted(ckpts, key=lambda x: x[1])[0][0]
    return best_ckpt


def generate_output_name(args):
    if args.load_from is not None:
        model_dir, ckpt = os.path.split(args.load_from)
        model_name, _ = os.path.splitext(ckpt)
    output_name = f'results/{args.task}/{args.lang}_train_{model_name}.tsv'
    return output_name


def save_prediction(test_predictions, output_fname, args):
    test_predictions = test_predictions[0]
    output = pd.DataFrame({
        'sample': test_predictions['samples'],
        'label': test_predictions['y_true'],
        'pred': test_predictions['y_pred']
    })
    output['label'] = output['label'].apply(lambda x: TASK_IDX_TO_LABEL[x])
    output['pred'] = output['pred'].apply(lambda x: TASK_IDX_TO_LABEL[x])
    if args.task == 'lang':
        output['logit'] = test_predictions['lang_logits']
        output['logit'] = output['logit'].apply(lambda x: float(x[0]))
    output['lang'] = args.lang
    output.to_csv(output_fname, sep='\t', encoding='utf-8')
    print(f'Predicted outputs save as {output_fname}')