import warnings
import numpy as np
import pandas as pd
from . import utils
from scipy.stats import nbinom, chi2
from statsmodels.api import NegativeBinomial


def populate(lst_of_lst):
    population = dict()
    for i, lst in enumerate(lst_of_lst):
        for element in lst:
            if element not in population:
                population[element] = np.zeros(len(lst_of_lst))
            population[element][i] += 1
    return population


def nb_filter(population):
    nb_population = dict()
    for word, values in population.items():
        n_est, p_est = _mle(values)
        if not n_est and not p_est:
            continue
        if _chi2test(values, n_est, p_est):
            nb_population[word] = [n_est, p_est]
    return nb_population


def _mle(values):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nb = NegativeBinomial(values, np.ones_like(values)).fit(disp=False)
    except Exception:
        return None, None
    mu, alpha = np.exp(nb.params[0]), nb.params[1]
    p_exp = 1 / (1 + mu * alpha)
    n_exp = mu * p_exp / (1 - p_exp)
    return n_exp, p_exp


def _chi2test(values, n, p, alpha=0.05):
    N = len(values)
    f_data = pd.value_counts(
        values,
        bins=[*np.unique(np.linspace(0, max(values), utils.sturges_rule(N), dtype=int)), np.inf],
        sort=False,
    ).to_frame(name='observed')
    f_data['expected'] = N * (nbinom.cdf(f_data.index.right, n, p) - nbinom.cdf(f_data.index.left, n, p))
    utils.frequency_filter(f_data)
    if all(f_data['expected'] > 5):
        f_data /= N
        stat = N * sum((f_data['observed'] - f_data['expected']) ** 2 / f_data['expected'])
        if stat < chi2.ppf(1 - alpha, len(f_data) - 3):
            return True
    return False


# --- 


import seaborn as sns
from matplotlib import pyplot as plt

    
def scatter(dicts, title='', xlabel='', ylabel='', colors=('darkorange', 'dodgerblue'), annotate=False, lines=False, x_right=None, y_top=None, dpi=100):
    fig, ax = plt.subplots(1, 1, figsize=(12, 6), dpi=dpi)
    for color, (label, label_dict) in zip(colors, dicts.items()):
        x, y = zip(*label_dict.values())
        ax.scatter(x, y, s=12, color=color, label=label)
        if annotate:
            for key, value, in label_dict.items():
                ax.annotate(key, (value[0], value[1]), fontsize=1)
    if lines:
        ax.axhline(y=0, linewidth=0.1, linestyle='--', color='black')
        ax.axvline(x=0, linewidth=0.1, linestyle='--', color='black')
    fig.suptitle(title)
    ax.set_xlim(left=0, right=x_right)
    ax.set_ylim(bottom=0, top=y_top)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc='lower right')
    fig.tight_layout()
    

def hist(df, title='', dpi=100):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), dpi=dpi)
    sns.histplot(df['r_pos'], stat='density', bins='auto', color='darkorange', alpha=0.4, ax=ax1, label='positive')
    sns.histplot(df['r_neg'], stat='density', bins='auto', color='dodgerblue', alpha=0.4, ax=ax1, label='negative')
    sns.histplot(df['p_pos'], stat='density', bins='auto', color='darkorange', alpha=0.4, ax=ax2, label='positive')
    sns.histplot(df['p_neg'], stat='density', bins='auto', color='dodgerblue', alpha=0.4, ax=ax2, label='negative')
    fig.suptitle(title)
    ax1.set_xlabel(r'$r$')
    ax2.set_xlabel(r'$p$')
    ax1.legend()
    ax2.legend()
    plt.show()
