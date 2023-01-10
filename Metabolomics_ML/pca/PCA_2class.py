# import libraries
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Data:
    entries: list
    labels: list

    @classmethod
    def new_from_csv(cls, fname: str) -> Data:
        """
        Creates a new Data class from test data.

        Parameters
        ----------
        fname: str
            File directory (file type must be .csv)
        """

        # parse data into a pandas DataFrame
        cls.test_data = pd.read_csv(fname)
        ids = cls.test_data.loc[:, 'ID'].to_numpy()
        classes = cls.test_data.loc[:, 'Class'].to_numpy()
        integs = [
            cls.test_data.iloc[i,2:].to_numpy() 
            for i in range(0, len(cls.test_data))
        ]
        
        labels = list(cls.test_data.columns.values)[2:]

        entries = [
            Entry(id_, class_, integ)
            for id_, class_, integ in zip(ids, classes, integs)
        ]

        return cls(entries, labels)
    
    def get_entry_from_id(self, id: int, show_class: bool=False):
        """
        Gets entry from id. By default, only integs are shown. If show_class is 
        True, then output is in the form [class_: str, integs: np.ndarray].
        """
        for entry in self.entries:
            if entry.id == id:
                if show_class:
                    return [entry.class_, entry.integs]
                else:
                    return entry.integs    
    ### raise exception if id is not in entry ids

    def set_dataset_classes(self, control: str, case: str, class_labels: dict=None, sort=False):
        """
        Sets control and case for the dataset (currently 1 control and 1 case supported).
        Takes optional input class_labels (dict, keys: 'control', 'case') 
        which turns string inputs for control/case into ints (e.g. -1, 1). Automatically 
        orders dataset so that lowest int comes first (i.e. set control as -1 if you would 
        like control to come before case in dataset).
        """
        self.original_control = control
        self.original_case = case

        if class_labels is not None:
            for entry in self.entries:
                if entry.class_ == self.original_control:
                    entry.class_ = class_labels['control']
                else:
                    entry.class_ = class_labels['case']
            
            for i in range(len(self.entries)):
                if self.test_data.loc[:,'Class'].iloc[i] == self.original_control:
                    self.test_data.loc[:,'Class'].iloc[i] = class_labels['control']
                else:
                    self.test_data.loc[:,'Class'].iloc[i] = class_labels['case']

            self.control = class_labels['control']
            self.case = class_labels['case']

            self.num_control = self.test_data['Class'].value_counts()[self.control]
            self.num_case = self.test_data['Class'].value_counts()[self.case]
            self.num_classes = len(class_labels)
        
        if sort:
            self.entries = sorted(self.entries, key=lambda entry: entry.class_)
    
    def _split_data(self, keep_id: bool=False):
        """
        Splits test data into integs and classes. Returns a tuple 
        (x_data, y_data). If keep_id is True, returns x_data as a list:
        [ids: np.ndarray, integs: np.ndarray]
        """
        if keep_id:
            x_data = (
                np.array([entry.id for entry in self.entries]), 
                np.array([entry.integs for entry in self.entries])
            )
        else:
            x_data = np.array([entry.integs for entry in self.entries])
        y_data = np.array([entry.class_ for entry in self.entries])

        return x_data, y_data

    def _scale_data(self):
        """
        Scales data: currently only standard scaling supported (zero mean and unit variance).
        Initialises scaled_test_data attribute, which presents scaled data as a pandas 
        DataFrame (same form as test_data).
        """
        if self.scaled_data is None:
            x_data, y_data = self._split_data()
            self._scaled_data = StandardScaler().fit_transform(x_data)
            scaled_test_data = pd.DataFrame(self._scaled_data, columns=self.labels)
            scaled_test_data.insert(0, column='Class', value=y_data)
            scaled_test_data.insert(0, column='ID', value=[entry.id for entry in self.entries])
            self.scaled_test_data = scaled_test_data

        return self.scaled_data
        ### allow option for scaling method
    
    @property
    def scaled_data(self):
        return getattr(self, "_scaled_data", None)
            
    def _get_pcs(self, n_components):
        if self.pca is None:
            self._pca = PCA(n_components=n_components, svd_solver='randomized')
            self._pca.fit_transform(self._scale_data())

        return self.pca

    def get_loadings(self, n_components: int=2):
        """
        Returns loadings matrix as a Pandas DataFrame. Takes in n_components as an input,
        set to 2 PCs by default
        """
        if self.n_components is None:
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

        x_data, y_data = self._split_data(keep_id=True)
        ids, integs = x_data
        scaled_integs = StandardScaler().fit_transform(integs)
        scores_matrix = scaled_integs @ np.array(self.get_loadings(self.n_components))

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
        
        #quantiles for PC1/PC2 currently
        pc1, pc2 = [j[0] for i, j in self.loadings], [j[1] for i, j in self.loadings]
        
        self.q = q
        self._upper_quantile_pc1 = np.quantile(pc1, q)
        self._lower_quantile_pc1 = np.quantile(pc1, 1-q)

        self.pc1_loadings = [
            (i, np.array([j[0], j[1]])) for i, j in self.loadings if j[0] > self._upper_quantile_pc1 or j[0] < self._lower_quantile_pc1
            ]
        self.pc2_loadings = [
            (i, np.array([j[0], j[1]])) for i, j in self.loadings if j[1] > np.quantile(pc2, q) or j[1] < np.quantile(pc2, 1-q)
            ]
        
        self.sig_loadings = self.pc1_loadings + self.pc2_loadings
        
        self._pc1_loadings_labels = [i[0] for i in self.pc1_loadings]
        self._pc2_loadings_labels = [i[0] for i in self.pc2_loadings]
        self._sig_loadings_labels = self._pc1_loadings_labels + self._pc2_loadings_labels

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
    
    def get_sig_data(self):
        """
        Returns (and initialises an attribute self.sig_data) a pandas DataFrame
        with all entries but only at significant peak frequencies given by the
        upper and lower quantiles. 
        """
        self.pc1_data = self.test_data.loc[:, ['ID', 'Class'] + [j[0] for j in self.pc1_loadings]]
        self.pc2_data = self.test_data.loc[:, ['ID', 'Class'] + [j[0] for j in self.pc2_loadings]]

        self.sig_data = self.test_data.loc[:,['ID', 'Class'] + self._sig_loadings_labels]

        return self.sig_data
    
    def run_ttests(self, sort_p_values: bool=False):
        """
        Takes pandas DataFrame of test data as input (e.g. pc1_data). Returns DataFrame of p_values and significance
        levels for each label (i.e. frequency range).
        """

        # get test_data from class
        data = self.test_data

        # initialise p_values DataFrame populated with zeros
        p_values = pd.DataFrame(np.zeros((len(data.columns[2:]), 3)), columns=['p-value', 'Significance', 'Bonferroni'], index=[data.columns[2:]])

        # initialise PC1/PC2 stats dictionaries
        pc1_stats = {}
        pc2_stats = {}

        ### iterating through pandas dataframe for t-test is slow - code to be optimised using numpy

        # query dataframe to give only control/case data
        self._control_data = data.query(f'Class == {self.control}')
        self._case_data = data.query(f'Class == {self.case}')
        bonf_factor = len(data.columns[2:])

        for i, col in enumerate(data.columns[2:]):

            # initialise DataFrame for mean, SD, and SEM for each column
            stat = pd.DataFrame(np.zeros((2, 3)), columns=['Mean', 'SD', 'SEM'], index=[self.control, self.case])

            # set mean, SD, and SEM for control and case for each column
            stat.loc[self.control, 'Mean'] = np.mean(self._control_data.loc[:, col])
            stat.loc[self.case, 'Mean'] = np.mean(self._case_data.loc[:, col])
            stat.loc[self.control, 'SD'] = np.std(self._control_data.loc[:, col])
            stat.loc[self.case, 'SD'] = np.std(self._case_data.loc[:, col])
            stat.loc[self.control, 'SEM'] = (stat.loc[self.control, 'SD'])/np.sqrt(self.num_control)
            stat.loc[self.case, 'SEM'] = (stat.loc[self.case, 'SD'])/np.sqrt(self.num_case)

            # store stats for top PC1/PC2 loadings - for plotting bar charts
            if col in self._pc1_loadings_labels:
                pc1_stats[col] = stat
            elif col in self._pc2_loadings_labels:
                pc2_stats[col] = stat
            
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
        
        if sort_p_values:
            p_values = p_values.sort_values(['p-value', 'Bonferroni'], key=lambda val: val.astype(float))

        self.ttests = TTest(p_values=p_values, pc1_stats=pc1_stats, pc2_stats=pc2_stats)
        
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
            for i in range(len(self.pc1_loadings)):
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
        
    def plot_scores(self, scores_matrix: pd.DataFrame, figure: tuple=None, fontsize: int=5):
        """
        Plots scores. If n_components is greater than 3, automatically plots only
        the first 3 PCs.
        """
        colors = []
        # get class values from numpy to speed up run time
        id_labels = scores_matrix.index
        np_scores = scores_matrix.to_numpy()

        for class_ in np_scores[:,0]:
            if class_ == self.control:
                colors.append('blue')
            else:
                colors.append('green')

        # create custom legend
        legend_elements = [
            Line2D([0], [0], label=self.original_control, color='blue', marker='o', markeredgecolor='black', alpha=0.5),
            Line2D([0], [0], label=self.original_case, color='green', marker='o', markeredgecolor='black', alpha=0.5)
            ]

        for handle in legend_elements:
            handle.set_linestyle("")

        if self.n_components == 2:
            self._plot_2d_scores(np_scores=np_scores, colors=colors, legend_elements=legend_elements, id_labels=id_labels, figure=figure, fontsize=fontsize)    
        elif self.n_components >= 3:
            self._plot_3d_scores(np_scores=np_scores, colors=colors, legend_elements=legend_elements, id_labels=id_labels, figure=figure, fontsize=fontsize)

    def _plot_2d_scores(self, np_scores, colors, legend_elements, id_labels, figure: tuple=None, fontsize: int=5):
        if figure is not None:
            fig, ax = figure
        else:  
            fig, ax = plt.subplots()

        ax.scatter(np_scores[:, 1], np_scores[:, 2], c=colors, s=25, edgecolors='black', alpha=0.5)
        for i, id in enumerate(id_labels):
            ax.annotate(id, (np_scores[i, 1], np_scores[i, 2]), fontsize=fontsize)
        self._add_labels_scores(ax, legend_elements)
        
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
    def _add_labels_scores(ax, legend_elements):
        ax.set_title('Scores Plot')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(linestyle='--')
        ax.legend(handles=legend_elements, loc='lower left', title='Classes', prop={'size': 8})
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel('PC3')
    
    @staticmethod
    def _add_labels_loadings(ax):
        ax.set_title('Loadings')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(linestyle='--')

    def plot_vars(self, vars_array: np.ndarray, threshold: float=None, cumulative: bool=False, figure: tuple=None):
        rows = [f'PC{i}' for i in range(1, self.n_components+1)]
        if figure is None:
            fig, ax = plt.subplots()
        else:
            fig, ax = figure

        if threshold is not None:
            ax.axhline(threshold, color='red', linestyle='--')

        ax.set_title("Explained Variance")    
        if cumulative:
            ax.bar(rows, np.cumsum(vars_array*100), edgecolor=(0, 0, 0, 1), color=(0, 0, 0, 0.3))
            ax.set_ylabel("Cumulative Variance")
        else:
            ax.bar(rows, vars_array*100, edgecolor=(0, 0, 0, 1), color=(0, 0, 0, 0.3), alpha=0.3)
            ax.set_ylabel("% of total variance")

    def plot_ttests(self, ttests: TTest):
        pc1_stats = ttests.pc1_stats
        pc2_stats = ttests.pc2_stats
        
        fig, ax = plt.subplots()

        for feature_name, stats in pc1_stats.items():
            ax.bar([self.original_control, self.original_case], [stats.loc[self.control, 'Mean'], stats.loc[self.case, 'Mean']], color=['green', 'blue'], yerr=[stats.loc[self.control, 'SEM'], stats.loc[self.case, 'SEM']], capsize=20)
            ax.set_title(feature_name)
            plt.show()

    @property
    def pca(self):
        return getattr(self, "_pca", None)
    
    @property
    def n_components(self):
        return getattr(self, "_n_components", None)

@dataclass
class Entry:
    id: int
    class_: str
    integs: np.ndarray

@dataclass
class TTest:
    p_values: pd.DataFrame
    pc1_stats: dict
    pc2_stats: dict


if __name__ == "__main__":
    test_data = Data.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\test_data.csv")

    # plots don't work without class_labels dict - cannot append strings into numpy array
    test_data.set_dataset_classes(control='RRMS', case='SPMS', class_labels={'control': -1, 'case': 1})

    ### where to put self.n_components? Currently a class property.
    loadings_matrix, scores_matrix, vars_array = test_data.get_loadings(n_components=20), test_data.get_scores(), test_data.get_vars(ratio=True)

    loadings_matrix = test_data.get_quantiles(loadings_matrix)

    #test_data.plot_loadings(loadings_matrix)

    # quantiles_matrix = test_data.get_quantiles(loadings_matrix)

    test_data.rank_loadings()
    test_data.get_sig_data()

    ttests = test_data.run_ttests()

    # test_data.plot_ttests(ttests)

    test_data.plot_vars(vars_array=vars_array, threshold=95, cumulative=True)
    # test_data.plot_loadings(quantiles_matrix, sig_labels=True)
    # test_data.plot_scores(scores_matrix)

    #print(scores_matrix.iloc[:,1:] @ loadings_matrix.iloc[:, :2].T)
    #print(test_data.scaled_test_data.iloc[:, 2:] - (scores_matrix.iloc[:,1:] @ loadings_matrix.iloc[:, :2].T))

    #x_data, y_data = test_data._split_data()
    #print(pd.DataFrame(StandardScaler().fit_transform(x_data)))

    ### TO DO
    # check out quantiles and ranked loadings
    # clean up all plots
    # create file with questions + next steps

    #### General rule - using get methods outputs pandas dataframes, but attributes use numpy arrays
    ### ttests in pandas dataframes due to use of strings e.g. significance level

    # plot spectrum
    #fig, ax = plt.subplots()
    #x_vals = np.linspace(0, 1, len(test_data.get_entry_from_id(1)))
    #ax.plot(x_vals, test_data.get_entry_from_id(20))
    #ax.plot(x_vals, test_data.get_entry_from_id(29))


    plt.show()