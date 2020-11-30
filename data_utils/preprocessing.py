import re
import string
import emoji
from soynlp.normalizer import repeat_normalize


emojis = ''.join(emoji.UNICODE_EMOJI.keys())
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
url_pattern = re.compile(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')


def preprocess(doc, lang):
    if lang == 'ko':
        return preprocess_ko(doc)
    elif 'da' in lang or lang == 'en':
        return preprocess_tweet(doc)
    else:
        raise ValueError('language should be one of [da, ko, en].')


def preprocess_ko(x):
    # https://github.com/Beomi/KcBERT
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x


def preprocess_tweet(x):
    x = replace_urls(x)
    x = limit_mentions(x, 2)
    x = x.lower()
    x = x.strip()
    return x


def replace_urls(sent):
    return sent.replace('URL', 'http')


def limit_pattern(doc, pattern, num_repeats):
    if pattern in string.punctuation:
        re_pattern = re.escape(pattern)
    else:
        re_pattern = f'(({pattern})[\s]*)'
        pattern = pattern + ' '
    pattern_regex = re_pattern + '{' + str(num_repeats + 1) + ',}'
    return re.sub(pattern_regex, lambda match: pattern * num_repeats, doc)


def limit_mentions(sent, keep_num):
    return limit_pattern(sent, '@USER', keep_num)