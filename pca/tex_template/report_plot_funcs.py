import sys
import os
from sklearn import __version__ as sklearn_version
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import subprocess

# Change matplotlib figures to LaTeX style
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Import PCA_2class
sys.path.append(os.path.abspath(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\pca"))
from PCA_2class import *

# Initialise test_data
test_data = Data.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\test_data.csv")
test_data.set_dataset_classes(control='RRMS', case='SPMS', class_labels={'control': -1, 'case': 1})
loadings_matrix, scores_matrix, vars_array = test_data.get_loadings(n_components=2), test_data.get_scores(), test_data.get_vars(ratio=True)
quantiles_matrix = test_data.get_quantiles(loadings_matrix, q=0.95)
ranked_loadings_matrix = test_data.rank_loadings()
ttests = test_data.run_ttests(sort_p_values=True)

upper_quantile = test_data.q
lower_quantile = round(1 - test_data.q, 2)

def summary():
    """
    Creates a summary plot, including scores, loadings, and variance per principle component.
    """

    fig, axs = plt.subplots(2, 2)
    fig.set_figwidth(483.69684 / 72.27)
    fig.set_figheight(483.69684 / 72.27)

    axs[0, 0] = test_data.plot_vars(vars_array, figure=(fig, axs[0, 0]))
    axs[1, 0] = test_data.plot_scores(scores_matrix, figure=(fig, axs[1, 0]))
    axs[1, 1] = test_data.plot_loadings(quantiles_matrix, figure=(fig, axs[1, 1]))

    fig.tight_layout()
    fig.savefig("summary_figs.pdf")

def hotelings_scores():
    pass

def ranked_loadings():
    """
    Creates ranked loadings plots based off PC1 values, with threshold lines for upper 
    and lower quantiles, as well as labels for significant loadings. 
    """

    fig, axs = plt.subplots(2, 1)
    fig.set_figwidth(483.69684 / 72.27)
    fig.set_figheight(650 / 72.27)

    axs[0] = test_data.plot_ranked_loadings(ranked_loadings_matrix, figure=(fig, axs[0]))
    axs[1] = test_data.plot_ranked_loadings(ranked_loadings_matrix, figure=(fig, axs[1]), threshold=False, labels=True)

    fig.tight_layout()
    fig.savefig("ranked_loadings.pdf")

def p_values_tables(top_loadings: bool=False):

    p_value_text = ""

    for index, row in zip(ttests.p_values.index, ttests.p_values.to_numpy()):

        if top_loadings:
            if index in test_data._sig_loadings_labels:
                p_value_text += index + " & " + " & ".join(row) + "\\\\\n"

        else:
            p_value_text += index + " & " + " & ".join(row) + "\\\\\n"
    
    return p_value_text[:-3]

def sig_bar_charts():
    fig, axs = plt.subplots(5, 2)
    fig.set_figwidth(483.69684 / 72.27)
    fig.set_figheight(650 / 72.27)


