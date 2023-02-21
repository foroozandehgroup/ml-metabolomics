# import libraries
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from scipy import stats
from dataclasses import dataclass

from Metabolomics_ML.base.data import Data

@dataclass
class PCAData(Data):
    """
    PCAData inherits from Data class, where class is instantiated from .csv or pandas
    DataFrame.
    """
            
    def _get_pcs(self, n_components):
        """
        If the pca attribute exists, it is returned. Otherwise, this methods finds the optimal
        number of principal components through a combination of K-fold cross-validation and 
        calculating the Q2 statistic for each successive principal component - the component
        where Q2 is at a maximum is chosen for the PCA model.
        """
        if self.pca is None:

            # scale the data and initialise the scaled_data attribute
            self._scale_data()

            self._pca = PCA(n_components=n_components, svd_solver='randomized')
            self._pca.fit_transform(self.scaled_data)

        return self.pca

    def _optimise_pcs(self):

            if self.scaled_data is None:
                self._scale_data()

            scaled_data_pd = self.scaled_test_data.iloc[:, 1:]

            # get covariance matrix of the scaled data
            cov_matrix = scaled_data_pd.cov().values

            # get trace of covariance matrix i.e. TSS (total sum of squares)
            tss = cov_matrix.trace()
            
            # initialise arrays to store q2 and r2 values
            q2, r2 = [], []
            
            # iterate through increasing numbers of PCs - if n_components attribute exists
            # (e.g. initialised in get_loadings method), then all of these components show up in
            # final q2 array

            if self.n_components is None:
                pcs_up_to = len(self.test_data)
            else:
                pcs_up_to = self.n_components

            for pc in range(1, pcs_up_to + 1):
                # initialise folds - 7-fold used in this case
                kf = KFold(n_splits=7)

                # fit PCA model on scaled data
                pca = PCA(n_components=pc)
                scores = pca.fit_transform(scaled_data_pd)
                
                # reconstruct data matrix from scores
                recon = pca.inverse_transform(scores)

                # use in-built r2 function between initial scaled data and the reconstructed data for i PCs
                r2_i = r2_score(scaled_data_pd, recon)

                # add r2 value to array of r2 values
                r2.append(r2_i)
                
                # initialise a matrix for the reconstructed matrix for q2 calculation
                mat_test = np.zeros(shape=scaled_data_pd.shape)

                # get indices of columns in train and test sets according to cross-valdiation method
                for train_index, test_index in kf.split(scaled_data_pd):

                    # split data into train and test sets for each fold
                    x_train, x_test = scaled_data_pd.iloc[train_index], scaled_data_pd.iloc[test_index]

                    # fit PCA model on training data
                    pca_cv = PCA(n_components=pc)
                    pca_cv.fit(x_train)

                    # calculate scores of test set and reconstruct data
                    scores_test = pca_cv.transform(x_test)
                    recon_test = pca_cv.inverse_transform(scores_test)

                    # insert reconstructed data into columns of final reconstructed matrix
                    mat_test[test_index, :] = recon_test
                
                # get residual matrix for this PC
                res = scaled_data_pd - mat_test
                
                # calculate PRESS (prediction error sum of squares) as the trace of the covariance matrix of residuals
                press = res.cov().values.trace()

                # q2 for ith PC calculated as 1 - PRESS/TSS
                q2_i = 1 - press/tss
                
                if self.n_components is None:
                    if pc == 1 or q2_i - q2[-1] > q2[-1]:
                        # add q2 for current PC to array of q2 values if q2 is higher than the previous q2
                        q2.append(q2_i)
                    else:
                        break
                else:
                    q2.append(q2_i)

            q2 = np.array(q2)

            # find differences in q2 i.e. q2 for the ith component as oppposed to cumulative q2
            self.q2_array = np.insert(q2[1:] - q2[0:-1], 0, q2[0])

            # returns the optimal number of PCs (minimum 2 PCs to allow plotting)
            return max(2, len(self.q2_array))

    def get_loadings(self, n_components=None):
        """
        Returns loadings matrix as a Pandas DataFrame. By default, optimises number of
        components to plot by calculating Q2 statistic. Takes in optional input n_components
        which can manually choose number of components.
        """
        if self.n_components is None:
            if n_components is None:
                self._n_components = self._optimise_pcs()
            else:
                self._n_components = n_components

        pca = self._get_pcs(n_components=self.n_components)
        cols = [f'PC{i}' for i in range(1, self.n_components+1)]
        self.loadings_matrix = pd.DataFrame(pca.components_.T, columns=cols, index=self.labels)
        self.loadings = [(a,b) for a, b in zip(self.labels, self.loadings_matrix.to_numpy())]

        return self.loadings_matrix
        ### check loading matrix scaling: https://stats.stackexchange.com/questions/104306/what-is-the-difference-between-loadings-and-correlation-loadings-in-pca-and/104640#10464
    
    def get_scores(self, n_components: int=2, keep_classes: bool=True):
        """
        Returns scores matrix as a Pandas DataFrame. Takes boolean input keep_classes:
        if True, outputs scores_matrix with additional column for respective class, as 
        assigned by set_dataset_classes method.
        """
        if self.n_components is None:
            self._n_components = n_components
        
        if self.scaled_data is None:
            self._scale_data()

        x_data, y_data = self._split_data(keep_id=True)
        ids, integs = x_data

        scores_matrix = self.scaled_data @ np.array(self.get_loadings(self.n_components))

        cols = [f'PC{i}' for i in range(1, self.n_components+1)]

        if keep_classes:
            y_data = y_data.reshape(-1, 1)
            scores_matrix = np.hstack((y_data, scores_matrix))
            cols.insert(0, 'Class')
            self.scores_matrix = pd.DataFrame(scores_matrix, columns=cols, index=ids)
            return self.scores_matrix
        else:
            self.scores_matrix = pd.DataFrame(scores_matrix, columns=cols, index=ids)
            return self.scores_matrix
    
    def get_vars(self, n_components: int=2, ratio: bool=False):
        """
        Set ratio to True if desired output is the proportion of the total variance
        for each PC.
        """
        if self.n_components is None:
            self._n_components = n_components

        pca = self._get_pcs(n_components=self.n_components)

        if ratio:
            self.vars_array = pca.explained_variance_ratio_
            return self.vars_array
        else:
            self.vars_array = pca.explained_variance_
            return self.vars_array
    
    def get_quantiles(self, loadings_matrix: pd.DataFrame, q: float=0.95):
        
        self.q = q

        self._sig_labels_list = []
        """
        List of lists where each list represents all significant labels for that PC (index 0
        represents PC1 etc.)
        """

        self._sig_loadings_labels = []
        """
        Labels of all significant loadings for all PCs chosen.
        """

        for i in range(1, self.n_components+1):
            current_pc_labels, current_pc = np.array(loadings_matrix.index), loadings_matrix.loc[:,f'PC{i}'].to_numpy()
            upper_quantile, lower_quantile = np.quantile(current_pc, self.q), np.quantile(current_pc, 1-self.q)
            
            # saving upper and lower quantiles for PC1 for plotting ranked loadings
            if i == 1:
                self._upper_quantile_pc1 = upper_quantile
                self._lower_quantile_pc1 = lower_quantile

            sig_labels = [
                label for label, loading in zip(current_pc_labels, current_pc) if loading >= upper_quantile or loading <= lower_quantile
            ]

            self._sig_labels_list.append(sig_labels)

            # labels of loadings outside upper/lower quartiles for the current PC
            self._sig_loadings_labels += sig_labels

        
        # list of significant loadings: tuples in the form (label, np.array (of loadings)) 
        self.sig_loadings = [loadings for loadings in self.loadings if loadings[0] in self._sig_loadings_labels]

        labels_column = [
            label if label in self._sig_loadings_labels else "" for label in self.labels
            ]

        loadings_matrix['Labels'] = np.array(labels_column).T

        self.quantiles_matrix = loadings_matrix

        return self.quantiles_matrix
    
    def rank_loadings(self):
        """
        Sorts loadings from smallest to largest. Sorts based off the value for PC1.
        """
        self.ranked_loadings_matrix = sorted(self.loadings, key= lambda x: x[1][0])
        return self.ranked_loadings_matrix
    
    def run_ttests(self, sort_p_values: bool=False):
        """
        Takes pandas DataFrame of test data as input (e.g. pc1_data). Returns DataFrame of p_values and significance
        levels for each label (i.e. frequency range).
        """

        # get test_data from class
        data = self.test_data

        # initialise p_values DataFrame populated with zeros
        p_values = pd.DataFrame(np.zeros((len(data.columns[1:]), 3)), columns=['p-value', 'Significance', 'Bonferroni'], index=[data.columns[1:]])

        # initialise stats list of dictionaries for all PCs - index 0 shows stats dict for PC1 etc.
        pcs_stats = []

        ### iterating through pandas dataframe for t-test is slow - code to be optimised using numpy

        # query dataframe to give only control/case data
        self._control_data = data.query(f'Class == {self.control}')
        self._case_data = data.query(f'Class == {self.case}')
        bonf_factor = len(data.columns[1:])

        for pc_labels in self._sig_labels_list:
            
            # initialise dictionary for storing stats for each label
            pc_stats = {}

            for i, col in enumerate(data.columns[1:]):

                # initialise DataFrame for mean, SD, and SEM for each column
                stat = pd.DataFrame(np.zeros((2, 3)), columns=['Mean', 'SD', 'SEM'], index=[self.control, self.case])

                # set mean, SD, and SEM for control and case for each column
                stat.loc[self.control, 'Mean'] = np.mean(self._control_data.loc[:, col])
                stat.loc[self.case, 'Mean'] = np.mean(self._case_data.loc[:, col])
                stat.loc[self.control, 'SD'] = np.std(self._control_data.loc[:, col])
                stat.loc[self.case, 'SD'] = np.std(self._case_data.loc[:, col])
                stat.loc[self.control, 'SEM'] = (stat.loc[self.control, 'SD'])/np.sqrt(self.num_control)
                stat.loc[self.case, 'SEM'] = (stat.loc[self.case, 'SD'])/np.sqrt(self.num_case)

                # store stats for top loadings in current PC- for plotting bar charts
                if col in pc_labels:
                    pc_stats[col] = stat
                
                # run t-test
                p_value = stats.ttest_ind_from_stats(mean1=stat.loc[self.control, 'Mean'], std1=stat.loc[self.control, 'SD'], nobs1=self.num_control,
                                                    mean2=stat.loc[self.case, 'Mean'], std2=stat.loc[self.case, 'SD'], nobs2=self.num_case)
                
                if p_value[1] < 0.001:
                    significance = '***'
                elif p_value[1] < 0.01:
                    significance = '**'
                elif p_value[1] <0.05:
                    significance = '*'
                else:
                    significance = 'NS'

                if i == 0:
                    p_values = pd.DataFrame([np.array([np.round(p_value[1], 6), significance, np.round(p_value[1] * bonf_factor, 6)])], columns=['p-value', 'Significance', 'Bonferroni'], index=[col])

                else:    
                    p_values.loc[col] = np.array([np.round(p_value[1], 6), significance, np.round(p_value[1] * bonf_factor, 6)])
            
            # add this PCs stats to pcs_stats list
            pcs_stats.append(pc_stats)
        
        if sort_p_values:
            p_values = p_values.sort_values(['p-value', 'Bonferroni'], key=lambda val: val.astype(float))

        self.ttests = TTest(p_values=p_values, pcs_stats=pcs_stats)
        
        return self.ttests
 
    def plot_loadings(self, loadings_matrix: pd.DataFrame, sig_labels: bool=True, figure: tuple=None):
        """
        Plot loadings (only 2D supported currently).
        """
        if figure is not None:
            fig, ax = figure
        else:
            fig, ax = plt.subplots()

        alphas = []
        if sig_labels:
            for label in loadings_matrix.loc[:, 'Labels']:
                if label == "":
                    alphas.append(0.2)
                else:
                    alphas.append(1)
            ax.scatter(loadings_matrix['PC1'], loadings_matrix['PC2'], color='black', s=10, alpha=alphas)
        else:
            ax.scatter(loadings_matrix['PC1'], loadings_matrix['PC2'], color='black', s=10, alpha=0.5)

        self._add_labels_loadings(ax)
    
    def plot_ranked_loadings(self, ranked_loadings: list, threshold: bool=True, labels: bool=False, figure: tuple=None):
        """
        Creates a plot of loadings in incresing order for PC1. Option to plot threshold lines 
        for upper and lower quartiles, and data labels for significant loadings (i.e above and 
        below the quartiles).

        Parameters
        ----------
        ranked_loadings: list
            List of tuples in the form (str, np.array)
        """
        if figure is not None:
            fig, ax = figure
        else:
            fig, ax = plt.subplots()

        x_points = [i+1 for i, entry in enumerate(ranked_loadings)]
        y_points = [entry[1][0] for entry in ranked_loadings]

        ax.scatter(x_points, y_points, c='black', s=10, alpha=0.5)

        if labels:
            for i in range(len(self._sig_labels_list[0])):
                ax.annotate(ranked_loadings[i][0], (x_points[i], y_points[i]), fontsize=3)
                ax.annotate(ranked_loadings[-i-1][0], (x_points[-i-1], y_points[-i-1]), fontsize=3)

        ax.set_title('Ranked Loadings')
        ax.set_xlabel('Rank')
        ax.set_ylabel('PC1 Loadings')
        ax.margins(x=0)

        # quantile threshold lines
        if threshold:
            ax.plot(x_points, [self._upper_quantile_pc1 for i in x_points], c='red', linestyle='--')
            ax.plot(x_points, [self._lower_quantile_pc1 for i in x_points], c='red', linestyle='--')
        
    def plot_scores(self, scores_matrix: pd.DataFrame, pcs: tuple=(1, 2), figure: tuple=None, hotelling: float=None, fontsize: int=5, colours: tuple=('blue', 'green')):
        """
        Plots scores. By default, plots the first 2 PCs. Optional tuple input pcs allows you to choose
        which PCs to plot (e.g. pcs=(1, 3) plots PC1 against PC3). Takes float hotelling as a parameter if you would like the
        scores plot to contain Hotelling's T2 confidence interval around the scores
        for each class. Float represents confidence interval.
        """
        control_colour, case_colour = colours
        
        colour_list = []
        # get class values from numpy to speed up run time
        # Bug fix: assumes class labels are kept in scores_matrix
        id_labels = scores_matrix.index
        np_scores = scores_matrix.to_numpy()

        for class_ in np_scores[:,0]:
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

        self._plot_2d_scores(np_scores=np_scores, colors=colour_list, legend_elements=legend_elements, id_labels=id_labels, pcs=pcs, figure=figure, hotelling=hotelling, fontsize=fontsize)    
        # elif self.n_components >= 3:
        #     self._plot_3d_scores(np_scores=np_scores, colors=colors, legend_elements=legend_elements, id_labels=id_labels, figure=figure, fontsize=fontsize)

    def _plot_2d_scores(self, np_scores, colors, legend_elements, id_labels, pcs: tuple=(1, 2), hotelling: float=None, figure: tuple=None, fontsize: int=5):
        if figure is not None:
            fig, ax = figure
        else:  
            fig, ax = plt.subplots()

        first_pc, second_pc = pcs

        ax.scatter(np_scores[:, first_pc], np_scores[:, second_pc], c=colors, s=25, edgecolors='black', alpha=0.7)
        for i, id in enumerate(id_labels):
            ax.annotate(id, (np_scores[i, first_pc], np_scores[i, second_pc]), fontsize=fontsize)
        self._add_labels_scores(ax, legend_elements, pcs=pcs)

        if hotelling:
            self._plot_hotelling(np_scores, q=hotelling, figure=(fig, ax))
      
    def _plot_3d_scores(self, np_scores, colors, legend_elements, id_labels, figure: tuple=None, fontsize: int=5):
        if figure is not None:
            fig, ax = figure
        else:       
            fig= plt.figure()

        ax = fig.add_subplot(projection='3d')
        ax.scatter(np_scores[:, 1], np_scores[:, 2], np_scores[:, 3], c=colors, s=25, edgecolors='black', alpha=0.5)
        self._add_labels_scores(ax, legend_elements)
        ### add text labels for 3d plot
    
    @staticmethod
    def _add_labels_scores(ax: plt.Axes, legend_elements, pcs: tuple):
        
        first_pc, second_pc = pcs

        ax.set_title('Scores Plot')
        ax.set_xlabel(f'PC{first_pc}')
        ax.set_ylabel(f'PC{second_pc}')
        ax.grid(linestyle='--')
        ax.legend(handles=legend_elements, loc='lower left', title='Classes', prop={'size': 8})
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel('PC3')
    
    @staticmethod
    def _add_labels_loadings(ax: plt.Axes):
        ax.set_title('Loadings')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(linestyle='--')
    
    @staticmethod
    def _plot_hotelling(scores: np.ndarray, q: float, figure: tuple):
        fig, ax = figure

        pass


    def plot_vars(self, vars_array: np.ndarray, threshold: float=None, cumulative: bool=False, figure: tuple=None):
        rows = [f'PC{i}' for i in range(1, self.n_components+1)]
        if figure is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figure

        if threshold is not None:
            ax.axhline(threshold*100, color='red', linestyle='--')

        ax.set_title("Explained Variance")    
        if cumulative:
            ax.bar(rows, np.cumsum(vars_array*100), edgecolor=(0, 0, 0, 1), color=(0, 0, 0, 0.3))
            ax.set_ylabel("Cumulative Variance")
        else:
            ax.bar(rows, vars_array*100, edgecolor=(0, 0, 0, 1), color=(0, 0, 0, 0.3), alpha=0.3)
            ax.set_ylabel("% of total variance")

    def plot_ttests(self, ttests: TTest):
        
        ttest_figure_number = plt.gcf().number

        # if no figure currently exists, reset ttest_figure_number so plots start from
        # figure 1
        if ttest_figure_number == 1:
            ttest_figure_number = 0

        # initialise list of figure objects to be used when plotting in
        # LaTeX document
        figure_list = []

        # looping through each PC, which is a dictionary of key-value pairs in the form:
        # label: DataFrame of stats

        for num_pc, pc in enumerate(ttests.pcs_stats, 1): 

            # total number of plots
            num_plots = 0
            # get index of current plot
            current_plot = 0
            
            while num_plots < len(pc):
                ttest_figure_number += 1
                fig = plt.figure(ttest_figure_number)             
                
                for i in range(1, 13):
                    if current_plot < len(pc.keys()):
                        current_ax = fig.add_subplot(4, 3, i)

                        # get stats for current plot
                        feature_name = list(pc.keys())[current_plot]
                        stats = pc[feature_name]

                        current_ax.bar(
                            [self.original_control, self.original_case], [stats.loc[self.control, 'Mean'], stats.loc[self.case, 'Mean']], color=['green', 'blue'], 
                            edgecolor='black', alpha=0.5, yerr=[stats.loc[self.control, 'SEM'], stats.loc[self.case, 'SEM']], capsize=10
                        )

                        self._format_ttests(figure=(fig, current_ax), feature_name=feature_name, ttests=ttests)

                        num_plots += 1
                        current_plot += 1
                
                fig.tight_layout()
                figure_list.append((fig, num_pc))
        
        return figure_list
    
    @staticmethod
    def _format_ttests(figure: tuple[plt.Figure, plt.Axes], feature_name: str, ttests: TTest):
        fig, ax = figure
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(bottom=ymin, top=ymax*1.25)
        ax.set_ylabel('Spectral Integral (AU)', fontsize=8)
        ax.set_title(feature_name, fontsize=8)
        ax.tick_params(which='both', labelsize=8)
        ax.text(0.5, 0.90, ttests.p_values.loc[feature_name, 'Significance'], verticalalignment='top', horizontalalignment='center', fontsize=9, color='black', transform=ax.transAxes)

    @property
    def pca(self):
        return getattr(self, "_pca", None)
    
    @property
    def n_components(self):
        return getattr(self, "_n_components", None)

@dataclass
class TTest:
    p_values: pd.DataFrame
    pcs_stats: list[dict]


if __name__ == "__main__":
    test_data = PCAData.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv")

    # plots don't work without class_labels dict - cannot append strings into numpy array
    test_data.set_dataset_classes(control='RRMS', case='SPMS', class_labels={'control': -1, 'case': 1})

    loadings_matrix = test_data.get_loadings()
    scores_matrix = test_data.get_scores()
    test_data.get_vars(ratio=True)
    test_data.get_quantiles(test_data.loadings_matrix, q=0.95)
    test_data.rank_loadings()
    test_data.run_ttests()

    # test_data.plot_scores(test_data.scores_matrix, pcs=(1, 2))

    fig_list = test_data.plot_ttests(test_data.ttests)

    plt.show()
