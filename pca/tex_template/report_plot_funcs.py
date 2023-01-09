import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import sys

# from pca.pca.PCA_2class import Data
# from gui.gui import GUIData

sys.path.append(os.path.abspath(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\pca"))

# os.chdir(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\pca")
from pca.PCA_2class import Data

# Change matplotlib figures to LaTeX style
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

class PCAPlot():

    def __init__(self):
        # super().__init__()
        pass

    @classmethod
    def run_all(cls, guidata):

        cls.pcaplot = Data.new_from_csv(guidata.filepath)
        cls.pcaplot.set_dataset_classes(control=guidata.control, case=guidata.case, class_labels={'control': -1, 'case': 1})

        cls.pcaplot.upper_quantile = guidata.q
        cls.pcaplot.lower_quantile = round(1 - guidata.q, 2)

        cls.pcaplot.get_loadings(n_components=2)
        cls.pcaplot.get_scores()
        cls.pcaplot.get_vars(ratio=True)
        cls.pcaplot.get_quantiles(cls.pcaplot.loadings_matrix, q=guidata.q)
        cls.pcaplot.rank_loadings()
        cls.pcaplot.run_ttests(sort_p_values=True)

        cls.summary(pcaplot=cls.pcaplot)
        cls.ranked_loadings(pcaplot=cls.pcaplot)
        cls.p_values_tables(pcaplot=cls.pcaplot)

        return cls

    @staticmethod
    def summary(pcaplot: Data):
        """
        Creates a summary plot, including scores, loadings, and variance per principle component.
        """

        fig, axs = plt.subplots(2, 2)
        fig.set_figwidth(483.69684 / 72.27)
        fig.set_figheight(483.69684 / 72.27)

        axs[0, 0] = pcaplot.plot_vars(pcaplot.vars_array, figure=(fig, axs[0, 0]))
        axs[1, 0] = pcaplot.plot_scores(pcaplot.scores_matrix, figure=(fig, axs[1, 0]))
        axs[1, 1] = pcaplot.plot_loadings(pcaplot.quantiles_matrix, figure=(fig, axs[1, 1]))

        fig.tight_layout()
        fig.savefig("summary_figs.pdf")
    
    @staticmethod
    def ranked_loadings(pcaplot: Data):
        """
        Creates ranked loadings plots based off PC1 values, with threshold lines for upper 
        and lower quantiles, as well as labels for significant loadings. 
        """

        fig, axs = plt.subplots(2, 1)
        fig.set_figwidth(483.69684 / 72.27)
        fig.set_figheight(650 / 72.27)

        axs[0] = pcaplot.plot_ranked_loadings(pcaplot.ranked_loadings_matrix, figure=(fig, axs[0]))
        axs[1] = pcaplot.plot_ranked_loadings(pcaplot.ranked_loadings_matrix, figure=(fig, axs[1]), threshold=False, labels=True)

        fig.tight_layout()
        fig.savefig("ranked_loadings.pdf")

    @staticmethod
    def p_values_tables(pcaplot: Data, top_loadings: bool=False):

        p_value_text = ""

        for index, row in zip(pcaplot.ttests.p_values.index, pcaplot.ttests.p_values.to_numpy()):

            if top_loadings:
                if index in pcaplot._sig_loadings_labels:
                    p_value_text += index + " & " + " & ".join(row) + "\\\\\n"

            else:
                p_value_text += index + " & " + " & ".join(row) + "\\\\\n"
        
        return p_value_text[:-3]


def hotelings_scores():
    pass

def sig_bar_charts():
    fig, axs = plt.subplots(5, 2)
    fig.set_figwidth(483.69684 / 72.27)
    fig.set_figheight(650 / 72.27)


