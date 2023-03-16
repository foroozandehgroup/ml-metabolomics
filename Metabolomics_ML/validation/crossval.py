# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from typing import Union
import random

from Metabolomics_ML.pls.oplsda import OPLSData
from Metabolomics_ML.pls.plsda import PLSData
from Metabolomics_ML.validation.metrics import Metrics, sensitivity_score, specificity_score

class CrossValidation:
    
    def __init__(self, estimator: Union[OPLSData, PLSData], folds: int=10, repetitions: int=100, metrics: list[Metrics]=None):
        # set random seed for reproducibility
        # np.random.seed(0)

        # initialises the scaled_test_data attribute for the estimator
        if not hasattr(estimator, "_scaled_data"):
            estimator._scale_data()
        
        self.estimator = estimator
        self.folds = folds
        self.repetitions = repetitions
        self.metrics = metrics
    
    @staticmethod
    def equalise_classes(estimator: Union[OPLSData, PLSData]):
        # get all id names for control and case
        # k fold with the difference between num control and num case
        
        if estimator.num_case > estimator.num_control:
            # random sample of cases to remove
            ids_to_remove = np.random.choice(estimator.case_ids, size=(estimator.num_case - estimator.num_control), replace=False)

            # check if random labels have been added for permutation test
            if hasattr(estimator, "_rand_scaled_data"):
                equal_data = estimator.rand_scaled_data.drop(index=ids_to_remove, inplace=False)
            
            else:
                # create copy of test data to replace for future iterations
                equal_data = estimator.scaled_test_data.drop(index=ids_to_remove, inplace=False)

            return equal_data

        elif estimator.num_control > estimator.num_case:
            ids_to_remove = np.random.choice(estimator.control_ids, size=(estimator.num_control - estimator.num_case), replace=False)

            if hasattr(estimator, "_rand_scaled_data"):
                equal_data = estimator.rand_scaled_data.drop(index=ids_to_remove, inplace=False)

            else:
                equal_data = estimator.scaled_test_data.drop(index=ids_to_remove, inplace=False)
            
            return equal_data

        # return untouched scaled test data in the case that class sizes are already equal
        return estimator.scaled_test_data

    @staticmethod
    def shuffle(data: pd.DataFrame):
        return data.sample(frac=1)

    def external_cv(self, data: pd.DataFrame, perm_test: bool=True):
        """
        Optional perm_test input - carries out permutation test. Adds results of each permutation test
        to Metrics object.
        """
        kf = KFold(n_splits=self.folds, shuffle=False)

        metrics = []

        for train_index, test_index in kf.split(data):
            
            # for each fold, split data into training and test set
            train_data_, test_data_ = data.iloc[train_index], data.iloc[test_index]

            if perm_test:
                # create Data objects from training and test data
                train_set, test_set = self._create_train_test(train_data_, test_data_, _from_perm_data=True)
            else:
                train_set, test_set = self._create_train_test(train_data_, test_data_, _from_perm_data=False)
                
            # build model on training set, and test using test set, storing the metrics 
            metrics.append(self.build_model(train_set, test_set, perm_test))

        return metrics

    def run_all(self, perm_test: bool=True):

        if self.metrics is None:
            self.metrics = []

        for i in range(self.repetitions):
            if perm_test:
                self.add_random_labels(self.estimator)
            equal_data = self.equalise_classes(self.estimator)
            equal_data = self.shuffle(equal_data)
            self.metrics.append(self.external_cv(equal_data, perm_test=perm_test))

            print(f"Repetition {i+1} complete.")
        
        return self.metrics
                    
    def _create_train_test(self, train_data_: pd.DataFrame, test_data_: pd.DataFrame, _from_perm_data: bool):
        """
        Creates and returns training and test Data objects (equivalent to estimator class) from folds
        of training and test data.
        """
        class_ = type(self.estimator)
        
        train_set = class_.new_from_df(train_data_, _from_perm_data=_from_perm_data)
        test_set = class_.new_from_df(test_data_, _from_perm_data=_from_perm_data)

        # relies on set_dataset_classes method being called on original estimator
        train_set._set_classes_validation(
            control=self.estimator.original_control,
            case=self.estimator.original_case,
            class_labels={'control': self.estimator.control, 'case': self.estimator.case}
            )
        test_set._set_classes_validation(
            control=self.estimator.original_control,
            case=self.estimator.original_case,
            class_labels={'control': self.estimator.control, 'case': self.estimator.case}
            )

        # initialise scaled_test_data attribute of train/test set
        if not hasattr(train_set, "_scaled_data") or not hasattr(test_set, "_scaled_data"):
            # using _scale_data method sets mean on truncated data
            train_set.scaled_test_data = train_set.test_data.copy()
            test_set.scaled_test_data = test_set.test_data.copy()

            train_set._scaled_data = train_set.scaled_test_data.iloc[:, 1:].to_numpy()
            test_set._scaled_data = test_set.scaled_test_data.iloc[:, 1:].to_numpy()

        if _from_perm_data:
            train_set._rand_scaled_data = train_data_
            test_set._rand_scaled_data = test_data_

        return train_set, test_set

    def build_model(self, train_set: Union[OPLSData, PLSData], test_set: Union[OPLSData, PLSData], perm_test: bool):
        
        if type(train_set) is OPLSData:
            # initialises opls attribute
            t_matrix, t_ortho_matrix = train_set.get_scores()
            
            y = train_set.opls.predict(test_set.scaled_data)
        
        if type(train_set) is PLSData:
            # initialise pls attribute
            x_scores = train_set.get_scores()

            y = train_set.pls.predict(test_set.scaled_data)

        y_true = np.array([entry.class_ for entry in test_set.entries])
        
        if test_set.control > test_set.case:
            y_pred = np.array([
                test_set.control if val > (test_set.control + test_set.case) / 2 else test_set.case for val in y
            ])
        else:
            y_pred = np.array([
                test_set.control if val < (test_set.control + test_set.case) / 2 else test_set.case for val in y
                ])

        # confusion matrix in the form:
        #     [ [TP, FP],
        #       [FN, TN] ]
        conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[test_set.case, test_set.control]).T

        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = sensitivity_score(conf_matrix)
        specificity = specificity_score(conf_matrix)

        if perm_test:
            y_rand = np.array(test_set.rand_scaled_data.loc[:, 'Random'])

            rand_conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_rand, labels=[test_set.case, test_set.control]).T
            rand_accuracy = accuracy_score(y_true, y_rand)
            rand_sensitivity = sensitivity_score(rand_conf_matrix)
            rand_specificity = specificity_score(rand_conf_matrix)

            return Metrics(
            conf_matrix=conf_matrix,
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity,
            rand_conf_matrix=rand_conf_matrix,
            rand_accuracy=rand_accuracy,
            rand_sensitivity=rand_sensitivity,
            rand_specificity=rand_specificity
            )
        
        return Metrics(
            conf_matrix=conf_matrix,
            accuracy=accuracy,
            sensitivity=sensitivity,
            specificity=specificity
        )

    def add_random_labels(self, estimator: Union[OPLSData, PLSData]):
        """
        Initialise a new attribute rand_scaled_data which has random class labels for the permutation test.
        """
        samples = [estimator.control, estimator.case]
        rand_classes = np.random.choice(samples, size=len(estimator.entries), p=[0.5, 0.5])

        rand_scaled_data = estimator.scaled_test_data.copy()
        rand_scaled_data.insert(1, column='Random', value=rand_classes)

        estimator._rand_scaled_data = rand_scaled_data

        return rand_scaled_data

if __name__ == "__main__":
    test_data = OPLSData.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv")
    test_data.set_dataset_classes(control="RRMS", case="SPMS", class_labels={'control': -1, 'case': 1}, sort=True)
    cv = CrossValidation(estimator=test_data, folds=8, repetitions=50)

    metrics = cv.run_all(perm_test=True)
    accs = []
    spec = []
    sens = []
    rand_accs = []
    for metric_list in metrics:
        for metric in metric_list:
            accs.append(metric.accuracy)
            spec.append(metric.specificity)
            sens.append(metric.sensitivity)
            rand_accs.append(metric.rand_accuracy)
    
    print(np.mean(accs))
    print(np.mean(rand_accs))
    # print([entry.class_ for entry in train_set.entries])

    # self.test_data['Class'].value_counts()[self.control]

    

    # print(test_data.pls.predict(test_data.test_data.iloc[:, 1:]))

    


 
