# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from numpy import linalg as la
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict, train_test_split
from dataclasses import dataclass

from Metabolomics_ML.base.data import Data
from Metabolomics_ML.pls.opls import OPLS
from Metabolomics_ML.pca.PCA_2class import PCAData
from Metabolomics_ML.algorithms.nipals import nipals
from Metabolomics_ML.pls.plsda import PLSData

@dataclass
class OPLSData(Data):
    """
    OPLSData inherits from Data class, where class is instantiated from .csv or pandas
    DataFrame.
    """

    def _get_opls_components(self, n_components: int):
        
        if self.opls is None:

            self._scale_data()

            self._opls = OPLS(n_components=n_components)
            self._opls.fit(self.scaled_data, self.scaled_test_data.loc[:, 'Class'])
        
        return self.opls
    
    def _optimise_components(self):
        return 3

    def get_scores(self, n_components: int=None, keep_classes: bool=True):
        """
        Takes in number of components as an input - by default, the number is optimised by finding
        the maximum of the Q2 distribution. Returns tuple of scores, (T, T_ortho), both pd.DataFrames.
        """

        if self.n_components is None:
            if n_components is None:
                self._n_components = self._optimise_components()
            else:
                self._n_components = n_components

         
        self._get_opls_components(n_components=self.n_components)

        (ids, integs), y_data = self._split_data(keep_id=True)
        cols_t = [f'T{i}' for i in range(1, self.n_components+1)]
        cols_t_ortho = [f'O{i}' for i in range(1, self.n_components+1)]

        if keep_classes:
            y_data = y_data.reshape(-1, 1)
            t_matrix = np.hstack((y_data, self.opls.T))
            cols_t.insert(0, 'Class')
            self.t_matrix = pd.DataFrame(t_matrix, columns=cols_t, index=ids)
            t_ortho_matrix = np.hstack((y_data, self.opls.T_ortho))
            cols_t_ortho.insert(0, 'Class')
            self.t_ortho_matrix = pd.DataFrame(t_ortho_matrix, columns=cols_t_ortho, index=ids)
            return (self.t_matrix, self.t_ortho_matrix)
        
        else:
            self.t_matrix = pd.DataFrame(self.opls.T, columns=cols_t, index=ids)
            self.t_ortho_matrix = pd.DataFrame(self.opls.T_ortho, columns=cols_t_ortho, index=ids)
            return (self.t_matrix, self.t_ortho_matrix)
    
    def plot_scores(self, t_matrix: pd.DataFrame, t_ortho_matrix: pd.DataFrame, components: tuple=(-1, 1), figure: tuple=None, hotelling: float=None, fontsize: int=5, colours: tuple=('blue', 'green')):

        if figure is not None:
            fig, ax = figure
        else:
            fig, ax = plt.subplots()

        control_colour, case_colour = colours
        colour_list = []

        id_labels = t_matrix.index
        np_scores = t_matrix.to_numpy()
        np_scores_ortho = t_ortho_matrix.to_numpy()

        for class_ in np_scores[:, 0]:
            if class_ == self.control:
                colour_list.append(control_colour)
            else:
                colour_list.append(case_colour)

        legend_elements = [
            Line2D([0], [0], label=self.original_control, color=control_colour, marker='o', markeredgecolor='black', alpha=0.7),
            Line2D([0], [0], label=self.original_case, color=case_colour, marker='o', markeredgecolor='black', alpha=0.7)
        ]

        for handle in legend_elements:
            handle.set_linestyle("")
        
        first_comp, second_comp = components

        ax.scatter(np_scores[:, first_comp], np_scores_ortho[:, second_comp], c=colour_list, edgecolors='black', alpha=0.7)
        for i, id in enumerate(id_labels):
            ax.annotate(id, (np_scores[i, first_comp], np_scores_ortho[i, second_comp]), fontsize=fontsize)

        ax.set_title("Scores Plot (OPLS-DA)") 
        ax.set_xlabel(f"T{first_comp}")
        ax.set_ylabel(f"O{second_comp}")
        ax.grid(linestyle="--")
        ax.legend(handles=legend_elements, loc="lower left", title="Classes", prop={"size": 8})
            
    
    @property
    def n_components(self):
        return getattr(self, "_n_components", None)
    
    @property
    def opls(self) -> OPLS:
        return getattr(self, "_opls", None)




if __name__ == "__main__":
    pls = PLSData.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv")
    pls.set_dataset_classes(control='RRMS', case='SPMS', class_labels={'control': -1, 'case': 1})
    scores_matrix = pls.get_scores(n_components=3)
    pls.plot_scores(scores_matrix, fontsize=10)

    test_data = OPLSData.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv")
    test_data.set_dataset_classes(control='RRMS', case='SPMS', class_labels={'control': -1, 'case': 1})
    t_matrix, t_ortho_matrix = test_data.get_scores(n_components=3)
    test_data.plot_scores(t_matrix, t_ortho_matrix, fontsize=10)

    plt.show()