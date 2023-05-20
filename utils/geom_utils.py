import numpy as np
import pandas as pd
from . import utils
from scipy.stats import geom, chi2


def populate(texts):
    population = dict()
    for text in texts:
        for j, ngram in enumerate(text):
            if ngram not in population:
                population[ngram] = []
            population[ngram].append(j)
    return population


def geom_filter(population):
    geom_population = dict()
    for word, values in population.items():
        p_est = len(values) / (sum(values) + len(values))
        if _chi2test(values, p_est):
            geom_population[word] = p_est
    return geom_population


def _chi2test(values, p, alpha=0.05):
    N = len(values)
    f_data = pd.value_counts(
        values,
        bins=[*np.unique(np.linspace(0, max(values), utils.sturges_rule(N), dtype=int)), np.inf],
        sort=False,
    ).to_frame(name='observed')
    f_data['expected'] = N * (geom.cdf(f_data.index.right, p) - geom.cdf(f_data.index.left, p))
    utils.frequency_filter(f_data)
    if all(f_data['expected'] > 5):
        f_data /= N
        stat = N * sum((f_data['observed'] - f_data['expected']) ** 2 / f_data['expected'])
        if stat < chi2.ppf(1 - alpha, len(f_data) - 1):
            return True
    return False


# ---


import seaborn as sns
from matplotlib import pyplot as plt


def hist(df, title='', dpi=100):
    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    sns.histplot(df['positive'], stat='density', bins='auto', color='darkorange', alpha=0.4, ax=ax, label='positive')
    sns.histplot(df['negative'], stat='density', bins='auto', color='dodgerblue', alpha=0.4, ax=ax, label='negative')
    fig.suptitle(title)
    ax.set_xlabel(r'$p$')
    ax.legend()
    plt.show()
