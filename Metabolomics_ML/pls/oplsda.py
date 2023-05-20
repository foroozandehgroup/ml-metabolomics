# import libraries
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from dataclasses import dataclass
from scipy import stats

from Metabolomics_ML.base.data import Data
from Metabolomics_ML.pls.opls import OPLS
from Metabolomics_ML.pls.plsda import PLSData

@dataclass
class OPLSData(Data):
    """
    OPLSData inherits from Data class, where class is instantiated from .csv or pandas
    DataFrame.
    """

    def _get_opls_components(self, n_components: int):
        
        if self.opls is None:
            
            if self.scaled_data is None:
                self._scale_data()
            
            self._opls = OPLS(n_components=n_components)
            self._opls.fit(self.scaled_data, self.scaled_test_data.loc[:, 'Class'])

        return self.opls
    
    def _optimise_components(self, dq2: bool=False):
       
        if self.scaled_data is None:
            self._scale_data()
        
        # getting X and Y data
        x_data = self.scaled_test_data.iloc[:, 1:]
        y_data = np.array(self.scaled_test_data.loc[:, 'Class'])
        # y_data = y_data.reshape(-1, 1)
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

            opls = OPLS(n_components=nc)
            x_scores, y_scores = opls.fit_transform(self.scaled_data, y_data)

            # calculation of R2X (cumulative)
            recon_x = opls.inverse_transform(x_scores)
            r2_xi = r2_score(self.scaled_data, recon_x)
            r2_x.append(r2_xi)
            
            # calculation of R2Y (cumulative)
            y_pred = opls.predict(self.scaled_data)
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

                opls_cv = OPLS(n_components=nc)
                opls_cv.fit(x_train, y_train)

                x_scores_test = opls_cv.transform(x_test)
                x_recon_test = opls_cv.inverse_transform(x_scores_test)

                y_pred = opls_cv.predict(x_test)
                y_pred = y_pred.reshape(-1, 1)

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
        if fontsize:
            for i, id in enumerate(id_labels):
                ax.annotate(id, (np_scores[i, first_comp], np_scores_ortho[i, second_comp]), fontsize=fontsize)

        ax.set_title(f"Scores Plot (OPLS-DA): n components = {first_comp}") 
        ax.set_xlabel(f"T{first_comp}")
        ax.set_ylabel(f"O{second_comp}")
        ax.grid(linestyle="--")
        ax.legend(handles=legend_elements, loc="lower left", prop={"size": 8}, title_fontsize=8, title="Classes")

        ellipse_data = []
        if hotelling:
            for i, colour in zip((self.control, self.case), colours):
                ell_data = self._plot_hotelling(np_scores[np_scores[:, 0] == i][:, components[0]], np_scores_ortho[np_scores_ortho[:, 0] == i][:, components[1]], hotelling, (fig, ax), colour)
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




    @property
    def n_components(self):
        return getattr(self, "_n_components", None)
    
    @property
    def opls(self) -> OPLS:
        return getattr(self, "_opls", None)


if __name__ == "__main__":
    test_data = OPLSData.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv")
    test_data.set_dataset_classes(control='RRMS', case='SPMS', class_labels={'control': -1, 'case': 1})
    t_matrix, t_ortho_matrix = test_data.get_scores(n_components=3)
    test_data.plot_scores(t_matrix, t_ortho_matrix, fontsize=10, colours=('blue', 'red'), components=(3, 3), hotelling=0.95)

    w_o = test_data.opls.W_ortho
    p_o = test_data.opls.P_ortho

    x = w_o @ np.linalg.inv(p_o.T @ w_o)

    x_corr = test_data.opls.correct(test_data.scaled_data)
    print(x)

    # plt.show()

    # print(test_data.r2_x)

