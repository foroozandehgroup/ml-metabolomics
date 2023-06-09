from Metabolomics_ML.gui.gui import GUIData
from Metabolomics_ML.pca.PCA_2class import PCAData
from Metabolomics_ML.tex_template.report_plot_funcs import PCAPlot
from Metabolomics_ML.tex_template.report_generator import generate
import os

test_1 = GUIData(
        name="DA",
        filepath=r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv",
        scaling="standard",
        q=0.95,
        control="RRMS",
        case="SPMS",
        control_colour="blue",
        case_colour="green",
        save_dir_path=r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\figures"
    )

test_2 = GUIData(
        name="DA",
        filepath=r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\tests\test_data.csv",
        scaling="standard",
        q=0.95,
        control="RRMS",
        case="SPMS",
        control_colour="blue",
        case_colour="green",
        save_dir_path=r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\figures"
    )

pcaplot = PCAPlot().run_all(test_1)

generate(pcaplot)

