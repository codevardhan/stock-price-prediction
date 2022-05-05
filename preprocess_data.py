from nltk.stem import PorterStemmer
import pandas as pd
import regex as re
import nltk
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from spellchecker import SpellChecker

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')


def filter_list(char):
    if char == "<p>" or not re.search('[a-zA-Z]', char) or re.search('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', char):
        return None
    if ".com" in char or "www." in char:
        return None
    return char.lower()


def process(df):
    stop = stopwords.words('english')

    df["cleaned_data"] = df["raw_data"].apply(lambda x: ' '.join(
        [word for word in x.split() if filter_list(word)]))
    df['cleaned_no_stop_punc_data'] = df['cleaned_data'].apply(
        lambda x: ' '.join([word.lower() for word in x.split() if word not in (stop)]))
    df['cleaned_no_stop_punc_data'] = df['cleaned_data'].apply(
        lambda x: "".join([c for c in list(x) if c not in string.punctuation]))
    return df
