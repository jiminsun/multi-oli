# multi-oli

This repository contains code for the task - Multilingual Offensive Language Identification.
Cross-lingual transfer techniques are utilized to improve performance in the target (often resource-poor) language, using resource-rich, source languages.
 
There are three languages considered:
* Danish (`da`)
* Korean (`ko`)
* English (`en`)

## Data
The English dataset is from [OffensEval2019](https://sites.google.com/site/offensevalsharedtask/offenseval2019), and the Danish dataset is from [OffensEval2020](https://sites.google.com/site/offensevalsharedtask/results-and-paper-submission). These two datasets are collected from Twitter.
The Korean dataset comes from the [Korean HateSpeech Dataset](https://github.com/kocohub/korean-hate-speech), a human-annotated corpus of Korean entertainment news comments. 

Strictly speaking, the Korean dataset is originally constructed for Toxic Speech Detection, which is a different task from Offensive Language Identification.
Still, we examine whether the two somewhat related tasks can improve one another via cross-lingual / domain transfer.

### Size
* Danish (2.9K)
* Korean (6.3K)
* English (14K)

### Labels
* NOT (not offensive; neutral)
* OFF (offensive)

## Model
* The offensiveness classifier is simply a multilingual BERT (m-BERT) encoder, followed by a linear layer. 

## Usage
```
python train.py [--task TASK] [--bert BERT] [--lang LANG]
                [--exp_name EXP_NAME] [--device DEVICE]
                [--load_from LOAD_FROM] [--data_dir DATA_DIR]
                [--train_file TRAIN_FILE] [--val_file VAL_FILE]
                [--test_file TEST_FILE] [--max_seq_len MAX_SEQ_LEN] [--lr LR]
                [--warmup_ratio WARMUP_RATIO] [--max_grad_norm MAX_GRAD_NORM]
                [--batch_size BATCH_SIZE] [--max_epochs MAX_EPOCHS]
                [--task_lr TASK_LR] [--gen_lr GEN_LR] [--disc_lr DISC_LR]
                [--train_with_both] [--val_with VAL_WITH]
```
