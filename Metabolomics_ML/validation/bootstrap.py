from sklearn.model_selection import BaseShuffleSplit
from sklearn.utils.validation import _num_samples
from sklearn.utils import resample
from sklearn.model_selection._split import _validate_shuffle_split
import numpy as np

class Bootstrap(BaseShuffleSplit):

    def __init__(self, n_splits: int, train_size: float=None, test_size: float=None, random_state: int=None):
        super().__init__(n_splits=n_splits, train_size=train_size, 
        random_state=random_state)

    def _iter_indices(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        indices = [i for i in range(n_samples)]

        for i in range(self.n_splits):
            # random partition with resampling            
            ind_train = resample(indices)
            ind_test = [i for i in indices if i not in ind_train]

            yield ind_train, ind_test



if __name__ == '__main__':
    array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    b = Bootstrap(n_splits=1)

    for train_index, test_index in b.split(array):
        print(train_index, test_index)