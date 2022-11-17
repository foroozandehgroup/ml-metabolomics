import sys
import os
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import subprocess

# Import PCA_2class.py

sys.path.append(os.path.abspath(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\pca"))
from PCA_2class import *

# Loading the template
os.chdir(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\pca\tex_template")
with open("report.tex", "r") as fh:
    template = fh.read()

# Chucking stuff in

# Getting package versions
template = template.replace("<SKLEARN_V>", sklearn.__version__)

# Get scaling method
template = template.replace("<SCALING>", "Standard")

# Create PCA Summary figures (scores, loadings, variances)

loadings_matrix, scores_matrix, vars_array = test_data.get_loadings(n_components=2), test_data.get_scores(), test_data.get_vars(ratio=True)
loadings_matrix = test_data.get_quantiles(loadings_matrix, q=0.95)

fig, axs = plt.subplots(2, 2)

axs[0, 0] = test_data.plot_vars(vars_array, figure=(fig, axs[0, 0]))
axs[1, 0] = test_data.plot_scores(scores_matrix, figure=(fig, axs[1, 0]))
axs[1, 1] = test_data.plot_loadings(loadings_matrix, figure=(fig, axs[1, 1]))

fig.savefig("summary_figs.pdf")

template = template.replace("<SUMMARY_FIGS>", "summary_figs.pdf")

tab = np.arange(16).reshape(4, 4)
table_text = ""
for row in tab:
    table_text += "    " + " & ".join([str(x) for x in row]) + "\\\\\n"
table_text = table_text[:-3]

template = template.replace("<TABLE>", table_text)

# Save the completed template
with open("report_complete.tex", "w") as fh:
    fh.write(template)



subprocess.run(["pdflatex", "report_complete.tex"], shell=True)


# tempfile - to create a temporary directory for log files