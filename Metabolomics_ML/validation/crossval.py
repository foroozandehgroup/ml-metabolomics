# import libraries
import numpy as np
from sklearn.model_selection import KFold
from typing import Union

from Metabolomics_ML.pls.oplsda import OPLSData
from Metabolomics_ML.pls.plsda import PLSData

class CrossValidation:
    
    def __init__(self, estimator: Union[OPLSData, PLSData], folds: int=10):
        self.estimator = estimator
        self.folds = folds
        
        np.random.seed(0)
    
    @staticmethod
    def equalise_classes(estimator: Union[OPLSData, PLSData]):
        # get all id names for control and case
        # k fold with the difference between num control and num case
        
        if estimator.num_case > estimator.num_control:
            # random sample of cases to remove
            ids_to_remove = np.random.choice(estimator.case_ids, size=(estimator.num_case - estimator.num_control), replace=False)
            print(ids_to_remove)

            # create copy of test data to replace for future iterations
            equal_data = estimator.test_data.drop(index=ids_to_remove, inplace=False)

            return equal_data

        elif estimator.num_control > estimator.num_case:
            ids_to_remove = np.random.choice(estimator.control_ids, size=(estimator.num_control - estimator.num_case), replace=False)
            print(ids_to_remove)

            equal_data = estimator.test_data.drop(index=ids_to_remove, inplace=False)
            
            return equal_data

if __name__ == "__main__":
    test_data = OPLSData.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv")
    test_data.set_dataset_classes(control='RRMS', case='SPMS', class_labels={'control': 0, 'case': 1}, sort=True)

    cv = CrossValidation(estimator=test_data)

    print(cv.equalise_classes(cv.estimator))



 
