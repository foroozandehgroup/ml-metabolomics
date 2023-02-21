from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class Data:
    test_data: pd.DataFrame

    @classmethod
    def new_from_csv(cls, fname: str):
        """
        Creates a new Data class from test data.

        Parameters
        ----------
        fname: str
            File directory (file type must be .csv)
        """
        
        # get data as a pandas DataFrame
        test_data = pd.read_csv(fname)

        cls.labels, cls.entries = cls._get_entries_from_data(test_data)

        test_data = test_data.set_index('ID')

        return cls(test_data)

    @classmethod
    def new_from_df(cls, df: pd.DataFrame):
        
        cls.labels, cls.entries = cls._get_entries_from_data(df)

        df = df.set_index('ID')

        return cls(df)
    
    @staticmethod
    def _get_entries_from_data(test_data: pd.DataFrame):
        
        ids = test_data.loc[:, 'ID'].to_numpy()
        classes = test_data.loc[:, 'Class'].to_numpy()
        integs = [
            test_data.iloc[i,2:].to_numpy() 
            for i in range(0, len(test_data))
        ]

        labels = list(test_data.columns.values)[2:]
        entries = [
            Entry(id_, class_, integ)
            for id_, class_ , integ in zip(ids, classes, integs)
        ]

        return labels, entries

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

            # Standard scaling by default -- do not use scikit-learn StandardScaler as this uses unbiased
            # definition of standard deviation, as opposed to numpy/pandas which used the biased defintion.
            scaled_data = pd.DataFrame(x_data, columns=self.labels, index=[entry.id for entry in self.entries]) 
            scaled_data = (scaled_data - scaled_data.mean())/scaled_data.std()
            self._scaled_data = scaled_data.to_numpy()

            scaled_data.insert(0, column='Class', value=y_data)
            self.scaled_test_data = scaled_data

        return self.scaled_data
        ### allow option for scaling method

    @property
    def scaled_data(self):
        return getattr(self, "_scaled_data", None)

@dataclass
class Entry:
    id: int
    class_: str
    integs: np.ndarray

if __name__ == "__main__":
    pass
    