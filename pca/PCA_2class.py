# import libraries
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler
from pyopls import OPLS
import pickle
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Data:
    entries: list
    labels: list

    @classmethod
    def new_from_csv(cls, fname: str) -> Data:

        # parse data into a pandas DataFrame
        test_data = pd.read_csv(f'{fname}')
        ids = test_data.loc[:, 'ID'].to_numpy()
        classes = test_data.loc[:, 'Class'].to_numpy()
        integs = [
            test_data.iloc[i,2:].to_numpy() 
            for i in range(0, len(test_data))
        ]

        labels = list(test_data.columns.values)[2:]

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

    def set_dataset_classes(self, control: str, case: str, class_labels: dict=None):
        """
        Sets control and case for the dataset (currently 1 control and 1 case supported).
        Takes optional input class_labels (dict, keys: 'control', 'case') 
        which turns string inputs for control/case into ints (e.g. -1, 1). Automatically 
        orders dataset so that lowest int comes first (i.e. set control as -1 if you would 
        like control to come before case in dataset).
        """
        self.control = control
        self.case = case

        if class_labels is not None:
            for entry in self.entries:
                if entry.class_ == self.control:
                    entry.class_ = class_labels['control']
                else:
                    entry.class_ = class_labels['case']

            self.control = class_labels['control']
            self.case = class_labels['case']
        
        #self.entries = sorted(self.entries, key=lambda entry: entry.class_)
    
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
            
    def _get_pcs(self, n_components):
        if self.pca is None:
            x_data, y_data = self._split_data()
            scaled_data = StandardScaler().fit_transform(x_data)
            self._pca = PCA(n_components=n_components)
            self._pca.fit_transform(scaled_data)

        return self.pca
        ### allow option for scaling method

    def get_loadings(self, n_components: int=2):
        """
        Returns loadings matrix as a Pandas DataFrame. Takes in n_components as an input,
        set to 2 PCs by default
        """
        if self.n_components is None:
            self._n_components = n_components

        pca = self._get_pcs(n_components=self.n_components)
        cols = [f'PC{i}' for i in range(1, self.n_components+1)]
        loadings = pd.DataFrame(pca.components_.T, columns=cols, index=self.labels)
        self.loadings = [(a,b) for a, b in zip(self.labels, loadings.to_numpy())]

        return loadings
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
            return pd.DataFrame(scores_matrix, columns=cols, index=ids)
        else:
            return pd.DataFrame(scores_matrix, columns=cols, index=ids)
    
    def get_vars(self, n_components: int=2, ratio: bool=False):
        """
        Set ratio to True if desired output is the proportion of the total variance
        for each PC.
        """
        if self.n_components is None:
            self._n_components = n_components

        pca = self._get_pcs(n_components=self.n_components)

        if ratio:
            return pca.explained_variance_ratio_
        else:
            return pca.explained_variance_
    
    def get_quantiles(self, loadings_matrix: pd.DataFrame, q: float=0.95):
        
        #quantiles for PC1/PC2 currently
        pc1, pc2 = [j[0] for i, j in self.loadings], [j[1] for i, j in self.loadings]

        self.pc1_loadings = [
            (i, np.array([j[0], j[1]])) for i, j in self.loadings if j[0] > np.quantile(pc1, q) or j[0] < np.quantile(pc1, 1-q)
            ]
        self.pc2_loadings = [
            (i, np.array([j[0], j[1]])) for i, j in self.loadings if j[1] > np.quantile(pc2, q) or j[1] < np.quantile(pc2, 1-q)
            ]
        
        self.sig_loadings = self.pc1_loadings + self.pc2_loadings
        
        sig_loadings_labels = [i[0] for i in self.sig_loadings]
        labels_column = [
            label if label in sig_loadings_labels else "" for label in self.labels
            ]

        loadings_matrix['Labels'] = np.array(labels_column).T

        return loadings_matrix

    def plot_loadings(self, loadings_matrix: pd.DataFrame, sig_labels: bool=True):
        """
        Plot loadings (only 2D supported currently).
        """
        fig, ax = plt.subplots()

        alphas = []
        if sig_labels:
            for label in loadings_matrix.loc[:, 'Labels']:
                if label == "":
                    alphas.append(0.2)
                else:
                    alphas.append(1)
            ax.scatter(loadings_matrix['PC1'], loadings_matrix['PC2'], color='black', alpha=alphas)
        else:
            ax.scatter(loadings_matrix['PC1'], loadings_matrix['PC2'], color='black', alpha=0.5)

        self._add_labels_loadings(ax)

        ### add axes labels, add gridlines, add datapoint labels by id
    
    def plot_scores(self, scores_matrix: pd.DataFrame):
        """
        Plots scores. If n_components is greater than 3, automatically plots only
        the first 3 PCs.
        """
        colors = []
        # get class values from numpy to speed up run time
        np_scores = scores_matrix.to_numpy()

        for class_ in np_scores[:,0]:
            if class_ == self.control:
                colors.append('blue')
            else:
                colors.append('green')

        # create custom legend
        legend_elements = [
            Line2D([0], [0], label=self.control, color='blue', marker='o', markeredgecolor='black', alpha=0.5),
            Line2D([0], [0], label=self.case, color='green', marker='o', markeredgecolor='black', alpha=0.5)
            ]

        for handle in legend_elements:
            handle.set_linestyle("")

        if self.n_components == 2:
            self._plot_2d_scores(np_scores=np_scores, colors=colors, legend_elements=legend_elements)    
        elif self.n_components >= 3:
            self._plot_3d_scores(np_scores=np_scores, colors=colors, legend_elements=legend_elements)
    
    def _plot_2d_scores(self, np_scores, colors, legend_elements):
        fig, ax = plt.subplots()
        ax.scatter(np_scores[:, 1], np_scores[:, 2], c=colors, edgecolors='black', alpha=0.5)
        self._add_labels_scores(ax, legend_elements)
        
    def _plot_3d_scores(self, np_scores, colors, legend_elements):
        fig= plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(np_scores[:, 1], np_scores[:, 2], np_scores[:, 3], c=colors, edgecolors='black', alpha=0.5)
        self._add_labels_scores(ax, legend_elements)
    
    @staticmethod
    def _add_labels_scores(ax, legend_elements):
        ax.set_title('Scores Plot')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(linestyle='--')
        ax.legend(handles=legend_elements, loc='lower left', title='Classes')
        if hasattr(ax, "set_zlabel"):
            ax.set_zlabel('PC3')
    
    @staticmethod
    def _add_labels_loadings(ax):
        ax.set_title('Loadings')
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.grid(linestyle='--')

    def plot_vars(self, vars_array: np.ndarray, threshold: float=None, cumulative: bool=False):
        rows = [f'PC{i}' for i in range(1, self.n_components+1)]
        fig, ax = plt.subplots()
        if threshold is not None:
            ax.axhline(threshold, color='red', linestyle='--')
        
        if cumulative:
            ax.bar(rows, np.cumsum(vars_array))
            ax.set_title("Cumulative Variance")
        else:
            ax.bar(rows, vars_array)
            ax.set_title("Variance per Principle Component")

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

test_data = Data.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\R Scripts\PCA\test_data.csv")

# plots don't work without class_labels dict - cannot append strings into numpy array
test_data.set_dataset_classes(control='SPMS', case='RRMS', class_labels={'control': -1, 'case': 1})

### where to put self.n_components? Currently a class property.
loadings_matrix, scores_matrix, vars_array = test_data.get_loadings(n_components=2), test_data.get_scores(), test_data.get_vars(ratio=True)

quantiles_matrix = test_data.get_quantiles(loadings_matrix)


#test_data.plot_vars(vars_array=vars_array, threshold=0.95, cumulative=True)
test_data.plot_loadings(quantiles_matrix, sig_labels=True)
#test_data.plot_scores(scores_matrix)

#print(test_data.get_quantiles())

plt.show()
 

### TO DO
# check out quantiles and ranked loadings
# clean up all plots
# create file with questions + next steps

#### General rule - using get methods outputs pandas dataframes, but attributes use numpy arrays


# plot spectrum
#fig, ax = plt.subplots()
#x_vals = np.linspace(0, 1, len(test_data.get_entry_from_id(1)))
#ax.plot(x_vals, test_data.get_entry_from_id(12))
#ax.plot(x_vals, test_data.get_entry_from_id(50))
#plt.show()


