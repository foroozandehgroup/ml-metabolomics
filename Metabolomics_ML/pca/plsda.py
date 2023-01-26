# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, roc_curve, roc_auc_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict, train_test_split
from dataclasses import dataclass
from pyopls import OPLS

from Metabolomics_ML.pca.data import Data
from Metabolomics_ML.pca.PCA_2class import PCAData

@dataclass
class OPLSData(Data):
    pass


if __name__ == "__main__":
    test_data = OPLSData.new_from_csv(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv")
    test_data.set_dataset_classes(control='RRMS', case='SPMS', class_labels={'control': 1, 'case': 0})
    test_data._scale_data()

    y_data, x_data = test_data.scaled_test_data.iloc[:, 0], test_data.scaled_test_data.iloc[:, 1:]

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, shuffle=True)

    opls = OPLS(39)
    Z = opls.fit_transform(x_train, y_train)

    pls = PLSRegression(1)
    y_pred = cross_val_predict(pls, x_train, y_train, cv=KFold(10))
    q_squared = r2_score(y_train, y_pred)
    dq_squared = r2_score(y_train, np.clip(y_pred, -1, 1))

    processed_y_pred = cross_val_predict(pls, Z, y_train, cv=KFold(10))
    processed_q_squared = r2_score(y_train, processed_y_pred)
    processed_dq_squared = r2_score(y_train, np.clip(processed_y_pred, -1, 1))

    r2_x = opls.score(x_train)

    pls.fit(Z, y_train)
    df = pd.DataFrame(np.column_stack([y_train, pls.x_scores_, opls.T_ortho_[:, 0]]),
    index=x_train.index, columns=['Class', 't', 't_ortho'])

    pos_df = df[df['Class'] == 1]
    neg_df = df[df['Class'] == 0]

    fpr, tpr, thresholds = roc_curve(y_train, y_pred)
    roc_auc = roc_auc_score(y_train, y_pred)
    proc_fpr, proc_tpr, proc_thresholds = roc_curve(y_train, processed_y_pred)
    proc_roc_auc = roc_auc_score(y_train, processed_y_pred)


    ## now transform test data

    x_test_ortho = opls.transform(x_test)
    x_test_scores, y_test_scores = pls.transform(x_test, y_test)

    df_test = pd.DataFrame(np.column_stack([y_test, x_test_scores, x_test_ortho[:, 0]]), index=x_test.index, columns=['Class', 't', 't_ortho'])


    fig, axs = plt.subplots(1, 2)

    axs[0].scatter(neg_df['t'], neg_df['t_ortho'], c='blue')
    axs[0].scatter(pos_df['t'], pos_df['t_ortho'], c='red')
    
    # test set
    axs[0].scatter(df_test['t'], df_test['t_ortho'], c='green')

    axs[0].set_title('PLS Scores')
    axs[0].set_xlabel('t_ortho')
    axs[0].set_ylabel('t')
    axs[0].legend(loc='upper right')

    axs[1].plot(fpr, tpr, lw=2, color='blue', label=f'Unprocessed (AUC={roc_auc:.4f})')
    axs[1].plot(proc_fpr, proc_tpr, lw=2, color='red',
         label=f'39-component OPLS (AUC={proc_roc_auc:.4f})')
    axs[1].plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].set_title('ROC Curve')
    axs[1].legend(loc='lower right')




    plt.show()




