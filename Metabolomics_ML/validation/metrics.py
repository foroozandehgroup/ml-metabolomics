from dataclasses import dataclass
import numpy as np

@dataclass
class Metrics:
    conf_matrix: np.ndarray
    accuracy: float
    sensitivity: float
    specificity: float
    q2: float=None
    r2x: float=None
    r2y: float=None
    n_comp: int=None
    rand_conf_matrix: np.ndarray=None
    rand_accuracy: float=None
    rand_sensitivity: float=None
    rand_specificity: float=None
    rand_q2: float=None
    rand_r2x: float=None
    rand_r2y: float=None
    mda: dict=None
    vip: list[tuple]=None


def sensitivity_score(conf_matrix: np.ndarray):
        """
        Calculates the sensitivity of the test given a confusion matrix in the form:

        [ [TP, FP],
          [FN, TN] ]

        """
        tp = conf_matrix[0, 0]
        fn = conf_matrix[1, 0]
        
        if tp == 0 and fn == 0:
            return None

        return tp / (tp + fn)

def specificity_score(conf_matrix: np.ndarray):
    """
    Calculates the specificity of the test given a confusion matrix in the form:

    [ [TP, FP],
      [FN, TN] ]

    """
    tn = conf_matrix[1, 1]
    fp = conf_matrix[0, 1]

    if tn == 0 and fp == 0:
         return None

    return tn / (tn + fp)