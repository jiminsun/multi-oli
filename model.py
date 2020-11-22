import pytorch_lightning as pl
import torch.nn as nn
from transformers import BertModel
from transformers import XLMModel
from transformers import XLMRobertaModel
from transformers import AutoModel

from models.pooling import TimePool

import gluonnlp as nlp
from transformers import AutoTokenizer
from transformers import BertTokenizer
from transformers import XLMTokenizer
from transformers import XLMRobertaTokenizer

from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer

"""
: Models
: 1) Classifier
"""


class Classifier(pl.LightningModule):
    """
    Binary classification
    model built on top of pre-trained encoder,
    followed by pooling mechanisms & linear layer.

    B : batch size
    E : embedding size
    L : max sequence length in batch

    """

    def __init__(self, args):
        super().__init__()
        self.encoder = load_pretrained(args.model)
        self.time_pool = TimePool(args.timepool)
        self.hidden_size = self.encoder.config.hidden_size * 2 if args.timepool == 'both' \
            else self.encoder.config.hidden_size
        self.out = nn.Linear(self.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        """

        Args:
            input_ids: input mapped to token ids                    [B x L]
            attention_mask: mask to avoid attn on padding tokens.
                            [PAD] -> 0                              [B x L]

        Returns:

        """
        outputs = self.encoder(  # [B x L x H]
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # outputs: last_hidden_state, pooler_output, hidden_states (optional)
        #   - last_hidden_state: [B x L x H]
        #   - pooler_output: [B x H]
        #   - hidden_states: tuple of [B x L x H]'s. one for each layer.
        last_hidden_state = outputs[0]
        x = self.time_pool(  # [B x H]
            hiddens=last_hidden_state,
            input_mask=attention_mask
        )
        logits = self.out(x)  # [B x 2]
        return logits


def load_pretrained(model_name):
    if model_name == 'mbert':
        model = BertModel.from_pretrained('bert-base-multilingual-uncased',
                                          output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    elif model_name == 'bert':
        model = BertModel.from_pretrained('bert-base-uncased',
                                          output_hidden_states=True)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'kobert':
        from kobert.pytorch_kobert import get_pytorch_kobert_model
        model, vocab = get_pytorch_kobert_model()
        # tokenizer = get_tokenizer()
        # tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    elif model_name == 'kcbert':
        model = AutoModel.from_pretrained('beomi/kcbert-base')
        tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')
    elif model_name == 'dabert':
        raise NotImplementedError
    elif model_name == 'xlm':
        model = XLMModel.from_pretrained('xlm-mlm-100-1280',
                                         output_hidden_states=True)
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
    elif model_name == 'xlm-r':
        model = XLMRobertaModel.from_pretrained('xlm-roberta-base',
                                                output_hidden_states=True)
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    else:
        raise ValueError("model name should be one of ['mbert', 'bert', 'kobert', 'kcbert', 'xlm', 'xlm-r']")

    if model_name in ['mbert', 'bert', 'xlm', 'xlm-r']:
        tokenizer.add_tokens(['@user'])
        model.resize_token_embeddings(len(tokenizer))
    return model
