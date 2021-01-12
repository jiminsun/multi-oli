from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BertModel
from transformers import BertTokenizer
from transformers import XLMModel
from transformers import XLMRobertaModel
from transformers import XLMRobertaTokenizer
from transformers import XLMTokenizer

from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer


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
    elif model_name == 'kcbert':
        model = AutoModel.from_pretrained('beomi/kcbert-base',
                                          output_hidden_states=True)
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
        tokenizer.add_tokens(['@user', '@USER'])
        model.resize_token_embeddings(len(tokenizer))

    return model


def load_tokenizer(model_name):
    if model_name == 'mbert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    elif model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif model_name == 'kobert':
        import gluonnlp as nlp
        _, vocab = get_pytorch_kobert_model()
        tokenizer = get_tokenizer()
        tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    elif model_name == 'kcbert':
        tokenizer = AutoTokenizer.from_pretrained('beomi/kcbert-base')
    elif model_name == 'xlm':
        tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-100-1280')
    elif model_name == 'xlm-r':
        tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    else:
        raise ValueError("model name should be one of ['mbert', 'bert', 'kobert', 'kcbert', 'xlm', 'xlm-r']")

    if model_name in ['mbert', 'bert', 'xlm', 'xlm-r']:
        tokenizer.add_tokens(['@user', '@USER'])

    return tokenizer