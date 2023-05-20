# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from dataclasses import dataclass
from scipy import stats

from Metabolomics_ML.base.data import Data
from Metabolomics_ML.pypls.pypls.cross_validation import CrossValidation

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
    
    def _optimise_components(self, dq2: bool=False):
        
        if self.scaled_data is None:
            self._scale_data()
        
        # getting X and Y data
        x_data = self.scaled_test_data.iloc[:, 1:]
        y_data = np.array(self.scaled_test_data.loc[:, 'Class'])
        y_data = y_data.reshape(-1, 1)
        # y_data = self._static_scale(y_data)

        # initialise arrays for Q2 and R2
        q2_x, r2_x = [], []
        q2_y, r2_y = [], []

        if self.n_components is None:
            comps_up_to = len(self.test_data)
        else:
            comps_up_to = self.n_components
        
        for nc in range(1, comps_up_to + 1):
            # 7-fold internal cross-validation - default used by SIMCA
            kf = KFold(n_splits=7, shuffle=True)

            pls = PLSRegression(n_components=nc)
            x_scores, y_scores = pls.fit_transform(self.scaled_data, y_data)

            # calculation of R2X (cumulative)
            recon_x, recon_y = pls.inverse_transform(x_scores, y_scores)
            r2_xi = r2_score(self.scaled_data, recon_x)
            r2_x.append(r2_xi)
            
            # calculation of R2Y (cumulative)
            y_pred = pls.predict(self.scaled_data)
            r2_yi = r2_score(y_data, y_pred)
            r2_y.append(r2_yi)

            # initialise a reconstructed matrix for the q2 calculation
            mat_test_x = np.zeros(shape=x_data.shape)
            mat_test_y = np.zeros(shape=y_data.shape)
            mat_test_y = mat_test_y.reshape(-1, 1)

            for train_index, test_index in kf.split(x_data, y_data):
                x_train, x_test = x_data.iloc[train_index], x_data.iloc[test_index]
                y_train, y_test = y_data[train_index], y_data[test_index]

                # check if all y values are the same (since scaling will give NaNs)
                # if len(np.unique(y_test)) == 1:
                #     y_test = np.zeros(y_test.shape)
                # else:
                #     y_test = self._static_scale(y_test)
                
                # apply independent scaling
                x_train, x_test = self._static_scale(x_train), self._static_scale(x_test)
                # y_train = self._static_scale(y_train)

                pls_cv = PLSRegression(n_components=nc)
                pls_cv.fit(x_train, y_train)

                x_scores_test, y_scores_test = pls_cv.transform(x_test, y_test)
                x_recon_test, y_recon_test = pls_cv.inverse_transform(x_scores_test, y_scores_test)

                y_pred = pls_cv.predict(x_test)

                if dq2:
                    # dq2 calculation (Westerhuis)
                    y_pred = np.clip(y_pred, -1 ,1)

                mat_test_x[test_index, :] = x_recon_test
                mat_test_y[test_index] = y_pred

            q2_xi = r2_score(x_data, mat_test_x)
            q2_x.append(q2_xi)

            q2_yi = r2_score(y_data, mat_test_y)
        
            if self.n_components is None:
                if nc == 1 or q2_yi > q2_y[-1]:
                    q2_y.append(q2_yi)
                else:
                    break
            else:
                q2_y.append(q2_yi)

        self.q2_x = q2_x
        self.r2_x = r2_x
        self.q2_y = q2_y
        self.r2_y = r2_y
        
        return max(2, len(self.q2_y))
    
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
                self._n_components = 3 #self._optimise_components()
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
        if fontsize:
            for i, id in enumerate(id_labels):
                ax.annotate(id, (np_scores[i, first_comp], np_scores[i, second_comp]), fontsize=fontsize)

        ax.set_title('Scores Plot (PLS-DA)')
        ax.set_xlabel(f'T{first_comp}')
        ax.set_ylabel(f'T{second_comp}')
        ax.grid(linestyle='--')
        ax.legend(handles=legend_elements, loc='lower left', title='Classes', prop={'size': 8})

        ellipse_data = []
        if hotelling:
            for i, colour in zip((self.control, self.case), colours):
                ell_data = self._plot_hotelling(np_scores[np_scores[:, 0] == i][:, components[0]], np_scores[np_scores[:, 0] == i][:, components[1]], hotelling, (fig, ax), colour)
                ellipse_data.append(ell_data)
        
        # rescaling of plot for ellipses
        x_extremes, y_extremes = [], []
        x_max, x_min, y_max, y_min = np.max(np_scores[:, components[0]]), np.min(np_scores[:, components[0]]), np.max(np_scores[:, components[1]]), np.min(np_scores[:, components[1]])
        x_extremes.extend([x_max, x_min])
        y_extremes.extend([y_max, y_min])
        
        for data in ellipse_data:
            centre, width, height, angle = data
            angle *= np.pi/180

            R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            half_width = width / 2
            half_height = height / 2
            vertices_rel = np.array([[-half_width, -half_height], [half_width, -half_height], [half_width, half_height], [-half_width, half_height]])
            vertices_rot = np.dot(R, vertices_rel.T).T
            vertices_abs = np.empty_like(vertices_rot)

            for i in range(4):
                vertices_abs[i, :] = vertices_rot[i, :] + centre.T

            x_extremes.extend([np.min(vertices_abs[:, 0]), np.max(vertices_abs[:, 0])])
            y_extremes.extend([np.min(vertices_abs[:, 1]), np.max(vertices_abs[:, 1])])

        ax.set_xlim(min(x_extremes)*1.05, max(x_extremes)*1.05)
        ax.set_ylim(min(y_extremes)*1.05, max(y_extremes)*1.05)

    def _plot_hotelling(self, first_scores: np.ndarray, second_scores: np.ndarray, q: float, figure: tuple, colour: str):
        
        fig, ax = figure
        
        first_mean, second_mean = np.mean(first_scores), np.mean(second_scores)
        scores = np.column_stack((first_scores, second_scores))

        cov = np.cov(scores.astype(float), rowvar=False)

        chi2_val = stats.chi2.ppf(q, 2)

        eig_vals, eig_vecs = np.linalg.eigh(cov)
        eig_order = eig_vals.argsort()[::-1]
        eig_vals, eig_vecs = eig_vals[eig_order], eig_vecs[:, eig_order]

        angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(chi2_val) * np.sqrt(eig_vals)

        ellipse = Ellipse(xy=(first_mean, second_mean), width=width, height=height, angle=angle, alpha=0.2, color=colour)
        ax.add_artist(ellipse)

        return (np.array([[first_mean], [second_mean]]), width, height, angle)

    def vip_scores(self, x_weights: np.ndarray, x_scores: np.ndarray, y_weights: np.ndarray):
        """
        PLS algorithm for VIP as in Galindo-Prieto et al. Returns a list of tuples of the 
        form (label, vip_score).
        """

        n_var = x_scores.shape[0]
        n_features = x_weights.shape[0]
        n_comp = x_scores.shape[1]

        ssy = np.zeros(n_comp)

        squared_y = np.square(np.multiply(x_scores, y_weights))

        for i in range(n_comp):
            ssy[i] = np.sum(squared_y[:, i])

        vip = np.sqrt(x_weights.shape[0] * np.sum((x_weights ** 2) * np.tile(ssy, (x_weights.shape[0], 1)), axis=1) / np.sum(ssy))

        return [(label, score) for label, score in zip(self.labels, vip)]
    
    
    @property
    def n_components(self):
        return getattr(self, "_n_components", None)

    @property
    def pls(self) -> PLSRegression:
        return getattr(self, "_pls", None)

if __name__ == "__main__":
    test_data = PLSData.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv")
    test_data = PLSData.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\Metabolomics_ML\thesis\datasets\test_data_shuffled.csv")
    test_data.set_dataset_classes(control='RRMS', case='SPMS', class_labels={'control': -1, 'case': 1})
    
    scores_matrix = test_data.get_scores(n_components=3)
    test_data.plot_scores(scores_matrix, colours=('blue', 'green'), components=(1, 2), hotelling=0.95)

    # test_data._optimise_components()

    coef = test_data.pls.coef_
    x_scores = test_data.pls.x_scores_
    y_scores = test_data.pls.y_scores_
    x_rot = test_data.pls.x_rotations_
    y_rot = test_data.pls.y_rotations_
    x_load = test_data.pls.x_loadings_
    y_load = test_data.pls.y_loadings_
    x_weight = test_data.pls.x_weights_
    y_weight = test_data.pls.y_weights_
    intercept = test_data.pls.intercept_

    vipvn = test_data.vip_scores(x_weight, x_scores, y_weight)
    
    # x = test_data.scaled_data
    # y = test_data.scaled_test_data.iloc[:, 0].to_numpy()
    # y = y.reshape(-1, 1)
    # matrix = x @ x.T @ y @ y.T
    # print(matrix.shape)

    # eigs = np.linalg.eigh(matrix.astype(float))
    # eigv = eigs[0].reshape(-1, 1)
    # print(eigv)
    # print(x_scores[:, 0])

    y = test_data.scaled_test_data.loc[:, 'Class'].to_numpy()
    y = y.reshape(-1, 1)
    y = test_data._static_scale(y)
    x = test_data.scaled_data

    u = y
    w = x.T @ u / (u.T @ u)
    w /= np.linalg.norm(w)
    t = x @ w
    c = y.T @ t / (t.T @ t)
    c /= np.linalg.norm(c)
    c = c[0][0]
    u = y.T * c

    # x -= t @ p.T
    # y -= u 

    # p = x.T @ t / (t.T @ t)
    # q = y @ u / (u.T @ u)

    # gv = x.T @ y @ y.T @ x @ w * 0.17411361259353422 / 9824.407775555617

    preds = x @ x_load @ y_load.T
    preds_ = [-1 if num < 0 else 1 for num in preds]
    tot = []
    for true, pred in zip(y, preds_):
        if (true < 0 and pred < 0) or (true > 0 and pred > 0):
            tot.append(0)
        else:
            tot.append(1)


    # print(x_load @ y_load.T)
    # print(x @ x_load - x_scores)
    # print(x_scores)

    print(x_scores)
    print(y_weight)

    a = np.array([[1, 2, 3], [2, 3, 4]])
    b = np.array([3, 3, 3])

    print(np.multiply(a, b))

    # plt.show()


