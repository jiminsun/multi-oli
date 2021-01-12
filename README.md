# multi-oli

This repository contains code for the task "Multilingual Offensive Language Identification", which has been investigated as my Master's dissertation project.
The objective of this project is to utilize cross-lingual transfer techniques to improve offensive language identification performance in the target (often resource-poor) language, using resource-rich, transfer languages.

In particular, I adapted the language-adversarial training pipeline suggested in [Adversarial Learning with Contextual Embeddings for Zero-resource Cross-lingual Classification and NER (Keung et al., 2019)](https://www.aclweb.org/anthology/D19-1138/).
 
Three languages are considered:
* Danish (`da`) - Target language 1
* Korean (`ko`) - Target language 2
* English (`en`) - Transfer language 

## Data
The English dataset is from [OffensEval2019](https://sites.google.com/site/offensevalsharedtask/offenseval2019), and the Danish dataset is from [OffensEval2020](https://sites.google.com/site/offensevalsharedtask/results-and-paper-submission). These two datasets are collected from Twitter.
The Korean dataset comes from the [Korean HateSpeech Dataset](https://github.com/kocohub/korean-hate-speech), a human-annotated corpus of Korean entertainment news comments. 

Strictly speaking, the Korean dataset is originally constructed for Toxic Speech Detection, which is a different task from Offensive Language Identification.
Still, we examine whether the two somewhat related tasks can improve one another via cross-lingual / domain transfer. (However, it turned out that using English data was not so helpful to improve the Korean task performance, at least in the methods that I have tried.)

Data in all three languages are processed into a unified dataframe format with columns: `id|sample|label|lang`, and saved as `./dataset/original/{LANGUAGE}.txt`.

Training, dev, test set for each language are prepared in `./dataset/{LANGUAGE}/`, but splitting with other random seeds or sizes can be done again by tweaking and running `./data_utils/split_train_val_test.py`.

### Size
* Danish (2.9K)
* Korean (6.3K)
* English (14K)

### Labels
* NOT (not offensive; neutral)
* OFF (offensive)

## Model
* The offensiveness classifier is simply a multilingual BERT (m-BERT) encoder, followed by a linear layer. 
* The baseline model in `./baseline` trains the classifier with the specified language's training data.
* The adversarial model in `./adversarial` performs language-adversarial training. In this case, a language-discriminator layer is added to the architecture. The model uses the targeted language's training data in addition to the English training data.

## Usage

### Requirements
```
pip install -r requirements.txt
```

### Start training
```
python train.py
```

Optional command-line arguments for training are detailed below:

```
  --task TASK           task to run: base (baseline) or adv (adversarial)
  --bert BERT           pre-trained model to use: bert, kobert, mbert, xlm
  --lang LANG           task language: da, ko, en
  --exp_name EXP_NAME   suffix to specify experiment name
  --device DEVICE
  --load_from LOAD_FROM
                        path to load model to resume training
  --batch_size BATCH_SIZE
  --max_epochs MAX_EPOCHS
  --min_epochs MIN_EPOCHS
  --seed SEED
  --freeze_bert
  --data_dir DATA_DIR   dataset directory
  --train_file TRAIN_FILE [TRAIN_FILE ...]
                        train file
  --val_file VAL_FILE   validation file
  --test_file TEST_FILE
                        test file
  --max_seq_len MAX_SEQ_LEN
  --lr LR               The initial learning rate
  --use_warmup
  --warmup_ratio WARMUP_RATIO
                        warmup ratio
  --max_grad_norm MAX_GRAD_NORM
                        gradient clipping
  --task_lr TASK_LR     task loss learning rate
  --gen_lr GEN_LR       generator loss learning rate
  --disc_lr DISC_LR     discriminator loss learning rate
  --train_with_both     if true, uses the concatenated training dataset of target language and English
```

Running this file will create a subdirectory inside `logs_{SEED}` directory, according to the experimental configuration (e.g., task, language, bert, experiment name), where all model checkpoints are saved during training.

### Run evaluation

To evaluate model performance on the test set, run:
```
python test.py --load_from MODEL_CHECKPOINT
```

The `MODEL_CHECKPOINT` can be either a `.ckpt` file, or a experiment directory with multiple `.ckpt` files. In the latter case, the model with the best dev set performance (F1 score or loss) will be selected automatically.

This process will print the evaluation metrics on command-line.

### Inference

To look into actual model predictions, run:

```
python inference.py --load_from MODEL_CHECKPOINT
```

The predicted results will be saved in the experiment directory that contains the model checkpoint, in a `.tsv` format.

