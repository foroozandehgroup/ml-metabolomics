import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import re

from Metabolomics_ML.pca.PCA_2class import PCAData
from Metabolomics_ML.gui.gui import GUIData


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
        pass

    @classmethod
    def run_all(cls, guidata: GUIData):

        cls.pcaplot = PCAData.new_from_csv(guidata.filepath)
        cls.pcaplot.set_dataset_classes(control=guidata.control, case=guidata.case, class_labels={'control': -1, 'case': 1})
        
        cls.pcaplot.guidata = guidata

        cls.pcaplot.upper_quantile = guidata.q
        cls.pcaplot.lower_quantile = round(1 - guidata.q, 2)

        cls.pcaplot.get_loadings()
        cls.pcaplot.get_scores()
        cls.pcaplot.get_vars(ratio=True)
        cls.pcaplot.get_quantiles(cls.pcaplot.loadings_matrix, q=guidata.q)
        cls.pcaplot.rank_loadings()
        cls.pcaplot.run_ttests(sort_p_values=True)

        cls.summary(pcaplot=cls.pcaplot)
        cls.ranked_loadings(pcaplot=cls.pcaplot)
        cls.p_values_tables(pcaplot=cls.pcaplot)
        cls.bar_charts(pcaplot=cls.pcaplot)

        return cls

    @staticmethod
    def summary(pcaplot: PCAData):
        """
        Creates a summary plot, including scores, loadings, and variance per principle component.
        """

        fig, axs = plt.subplots(2, 2)
        fig.set_figwidth(483.69684 / 72.27)
        fig.set_figheight(483.69684 / 72.27)

        axs[0, 0] = pcaplot.plot_vars(pcaplot.vars_array, figure=(fig, axs[0, 0]))
        axs[1, 0] = pcaplot.plot_scores(pcaplot.scores_matrix, figure=(fig, axs[1, 0]), colours=(pcaplot.guidata.control_colour, pcaplot.guidata.case_colour), hotelling=pcaplot.guidata.q)
        axs[1, 1] = pcaplot.plot_loadings(pcaplot.quantiles_matrix, figure=(fig, axs[1, 1]))

        fig.tight_layout()
        fig.savefig(rf"{pcaplot.guidata.save_dir_path}\summary_figs.pdf")
    
    @staticmethod
    def ranked_loadings(pcaplot: PCAData):
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
        fig.savefig(rf"{pcaplot.guidata.save_dir_path}\ranked_loadings.pdf")

    @staticmethod
    def p_values_tables(pcaplot: PCAData, top_loadings: bool=False):

        p_value_text = ""
        
        # regex for customising index labels
        regex = r"(-?\d+(?:\.\d+)?).*?(-?\d+(?:\.\d+)?)"

        for index, row in zip(pcaplot.ttests.p_values.index, pcaplot.ttests.p_values.to_numpy()):

            if top_loadings:
                if index in pcaplot._sig_loadings_labels:
                    
                    vals = re.search(regex, index).groups()
                    new_index = f"{vals[0]} -- {vals[1]}"
                    p_value_text += new_index + " & " + " & ".join(row) + "\\\\\n"

            else:
                vals = re.search(regex, index).groups()
                new_index = f"{vals[0]} -- {vals[1]}"
                p_value_text += new_index + " & " + " & ".join(row) + "\\\\\n"
        
        return p_value_text[:-3]
    
    @staticmethod
    def bar_charts(pcaplot: PCAData):
        """
        Plots bar charts for the p-values of all significant loadings for the top PCs.
        """
        
        ttest_figs = pcaplot.plot_ttests(pcaplot.ttests, colours=(pcaplot.guidata.control_colour, pcaplot.guidata.case_colour))

        for i, (fig, fig_pc) in enumerate(ttest_figs):
            fig.set_figwidth(483.69684 / 72.27)
            fig.set_figheight(650 / 72.27)
            fig.savefig(rf"{pcaplot.guidata.save_dir_path}\ttest_{i:03}_{fig_pc}")


def hotelings_scores():
    pass

def sig_bar_charts():
    fig, axs = plt.subplots(5, 2)
    fig.set_figwidth(483.69684 / 72.27)
    fig.set_figheight(650 / 72.27)


if __name__ == "__main__":
    pass