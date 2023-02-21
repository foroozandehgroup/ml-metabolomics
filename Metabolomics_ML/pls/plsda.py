# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict, train_test_split
from dataclasses import dataclass

from Metabolomics_ML.base.data import Data
from Metabolomics_ML.pca.PCA_2class import PCAData

@dataclass
class PLSData(Data):
    """
    PLSData inherits from Data class, where class is instantiated from .csv or pandas
    DataFrame.
    """
    
    def _get_pls_components(self, n_components: int):
        
        if self.pls is None:

            # initialises the scaled_test_data attribute
            self._scale_data()
            
            # scale set to False - sklearn uses different definition of standard
            # deviation for unit variance scaling 
            self._pls = PLSRegression(n_components=n_components, scale=False)
            self.x_scores, self.y_scores = self._pls.fit_transform(self.scaled_data, self.scaled_test_data.loc[:, 'Class'])

        return self.pls
    
    def _optimise_components(self):
        
        if self.scaled_data is None:
            self._scale_data()
        
        # getting X and Y data
        x_data = self.scaled_test_data.iloc[:, 1:]
        y_data = self.scaled_test_data.loc[:, 'Class']

        cov_matrix = x_data.cov(numeric_only=False).values
        tss_x = cov_matrix.trace()

        # initialise arrays for Q2 and R2
        q2_x, r2_x = [], []
        q2_y, r2_y = [], []

        if self.n_components is None:
            comps_up_to = len(self.test_data)
        else:
            comps_up_to = self.n_components
        
        for nc in range(1, comps_up_to + 1):
            # 7-fold internal cross-validation - default used by SIMCA
            kf = KFold(n_splits=10)

            pls = PLSRegression(n_components=nc)
            x_scores, y_scores = pls.fit_transform(self.scaled_data, y_data)

            # calculation of R2X
            recon_x, recon_y = pls.inverse_transform(x_scores, y_scores)
            r2_xi = r2_score(self.scaled_data, recon_x)
            r2_x.append(r2_xi)
            
            # calculation of R2Y
            y_pred = pls.predict(self.scaled_data)
            r2_yi = r2_score(y_data, y_pred)
            r2_y.append(r2_yi)

            # initialise a reconstructed matrix for the q2 calculation
            mat_test_x = np.zeros(shape=x_data.shape)
            mat_test_y = np.zeros(shape=y_data.shape)
            mat_test_y = mat_test_y.reshape(-1, 1)

            for train_index, test_index in kf.split(x_data, y_data):
                x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
                y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]

                pls_cv = PLSRegression(n_components=nc)
                pls_cv.fit(x_train, y_train)

                x_scores_test, y_scores_test = pls_cv.transform(x_test, y_test)
                x_recon_test, y_recon_test = pls_cv.inverse_transform(x_scores_test, y_scores_test)

                y_pred = pls_cv.predict(x_test)

                mat_test_x[test_index, :] = x_recon_test
                mat_test_y[test_index] = y_pred
            
            residual_x = x_data - mat_test_x
            press_x = residual_x.cov(numeric_only=False).values.trace()

            q2_xi = 1 - press_x/tss_x
            q2_x.append(q2_xi)

            q2_yi = r2_score(y_data, mat_test_y)
        
            if self.n_components is None:
                if nc == 1 or q2_yi - q2_y[-1] > q2_y[-1]:
                    q2_y.append(q2_yi)
                else:
                    break
            else:
                q2_y.append(q2_yi)
        
        print(q2_y)

        return 3
    
    def get_loadings(self, n_components: int=None):
        
        if self.n_components is None:
            if n_components is None:
                self._n_components = 2
            else:
                self._n_components = n_components
        
    def get_scores(self, n_components: int=None, keep_classes: bool=True):
        """
        Takes in number of components as an input - by default, the number is optimised
        by finding the maximum of the Q2 distribution. Returns tuple of x_scores and y_scores, 
        both pd.DataFrames.
        """
        
        if self.n_components is None:
            if n_components is None:
                self._n_components = self._optimise_components()
            else:
                self._n_components = n_components
        
        # initialises x_scores and y_scores attributes
        self._get_pls_components(n_components=self.n_components)

        (ids, integs), y_data = self._split_data(keep_id=True)
        
        cols = [f'T{i}' for i in range(1, self.n_components+1)]

        if keep_classes:
            y_data = y_data.reshape(-1, 1)
            scores_matrix = np.hstack((y_data, self.x_scores))
            cols.insert(0, 'Class')
            self.x_scores_matrix = pd.DataFrame(scores_matrix, columns=cols, index=ids)
            return self.x_scores_matrix
        else:
            self.x_scores_matrix = pd.DataFrame(self.x_scores, columns=cols, index=ids)
            return self.x_scores_matrix

    def plot_scores(self, scores_matrix: pd.DataFrame, components: tuple=(1, 2), figure: tuple=None, hotelling: float=None, fontsize: int=5, colours: tuple=('blue', 'green')):
        """
        Plots x-scores. By default, plots first 2 PLS components. Optional tuple input 'components' allows
        you to choose which components to plot (e.g. components=(1, 3) plots T1 vs T3). Takes float hotelling
        as a parameter for scores plot to contain Hotelling's T2 confidence interval.
        """

        if figure is not None:
            fig, ax = figure
        else:
            fig, ax = plt.subplots()

        control_colour, case_colour = colours
        colour_list = []
        
        # get class values from numpy to speed up run time
        # Bug fix: assumes class labels are kept in scores_matrix
        id_labels = scores_matrix.index
        np_scores = scores_matrix.to_numpy()
        
        for class_ in np_scores[:, 0]:
            if class_ == self.control:
                colour_list.append(control_colour)
            else:
                colour_list.append(case_colour)

        # create custom legend
        legend_elements = [
            Line2D([0], [0], label=self.original_control, color=control_colour, marker='o', markeredgecolor='black', alpha=0.7),
            Line2D([0], [0], label=self.original_case, color=case_colour, marker='o', markeredgecolor='black', alpha=0.7)
        ]

        for handle in legend_elements:
            handle.set_linestyle("")

        first_comp, second_comp = components

        ax.scatter(np_scores[:, first_comp], np_scores[:, second_comp], c=colour_list, edgecolors='black', alpha=0.7)
        for i, id in enumerate(id_labels):
            ax.annotate(id, (np_scores[i, first_comp], np_scores[i, second_comp]), fontsize=fontsize)

        ax.set_title('Scores Plot (PLS-DA)')
        ax.set_xlabel(f'T{first_comp}')
        ax.set_ylabel(f'T{second_comp}')
        ax.grid(linestyle='--')
        ax.legend(handles=legend_elements, loc='lower left', title='Classes', prop={'size': 8})
        
    @property
    def n_components(self):
        return getattr(self, "_n_components", None)

    @property
    def pls(self) -> PLSRegression:
        return getattr(self, "_pls", None)

if __name__ == "__main__":
    test_data = PLSData.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv")
    test_data.set_dataset_classes(control='RRMS', case='SPMS', class_labels={'control': -1, 'case': 1})
    scores_matrix = test_data.get_scores(n_components=4)
    test_data.plot_scores(scores_matrix, colours=('blue', 'green'), components=(1, 2))
    
    test_data._optimise_components()

    plt.show()


