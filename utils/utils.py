import string
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import ngrams
from nltk import tokenize
from nltk.corpus import stopwords
from matplotlib import pyplot as plt


DATASET_PATH = {
    'acl-imdb': 'data/prepared/acl-imdb.csv'
}


def load_data(data_name):
    if data_name in DATASET_PATH:
        return pd.read_csv(DATASET_PATH[data_name])
    raise FileNotFoundError


def word_tokenize(text, stop_words=False):
    tokens = [token.lower() for token in tokenize.word_tokenize(text)]
    for i, token in enumerate(tokens):
        tokens[i] = token.translate(str.maketrans('', '', string.punctuation))
        if tokens[i] == 'nt':
            tokens[i] = 'not'
    tokens = [token for token in tokens if token.strip()]
    if stop_words:
        stops = stopwords.words('english')
        tokens = [token for token in tokens if token not in stops]
    return tokens


def pos_filter(pos_texts):
    population = _pos_populate(pos_texts)
    for word in population.keys():
        population[word] = max(population[word], key=population[word].get)
    return [[(pair[0], population[pair[0]]) for pair in pos_text] for pos_text in pos_texts]


def _pos_populate(pos_texts):
    population = dict()
    for text in pos_texts:
        for pair in text:
            word, pos = pair
            if word not in population:
                population[word] = dict()
            population[word][pos] = population[word].get(pos, 0) + 1
    return population


def ngrams_sents(lst_of_lst, n):
    result = []
    for lst in lst_of_lst:
        result.append(list(ngrams(lst, n=n)))
    return result


def chunks(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def sturges_rule(n):
    return int(1 + np.floor(3.322 * np.log10(n)))


def frequency_filter(f_data):
    while any(f_data['expected'] < 5) and len(f_data) > 4:
        to_remove = f_data['expected'].argmin()
        remove_interval = f_data.index[to_remove]
        if to_remove == 0:
            to_union = 1
        elif to_remove == len(f_data) - 1:
            to_union = to_remove - 1
        else:
            to_union = to_remove - 1 if f_data['expected'][to_remove - 1] <= f_data['expected'][to_remove + 1] else to_remove + 1
        f_data.iloc[to_union, :] += f_data.iloc[to_remove, :]
        f_data.drop(f_data.index[to_remove], inplace=True)
        if to_union < to_remove:
            f_data.index.values[to_union] = pd.Interval(f_data.index[to_union].left, remove_interval.right)
        else:
            f_data.index.values[to_union - 1] = pd.Interval(remove_interval.left, f_data.index[to_union - 1].right)
            