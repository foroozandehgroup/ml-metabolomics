import sys
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import subprocess

import report_plot_funcs as plot

# Import PCA_2class.py

sys.path.append(os.path.abspath(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\pca"))
from PCA_2class import *

# Loading the template
os.chdir(r"C:\Users\mfgroup\Documents\Daniel Alimadadian\Metabolomics_ML\pca\tex_template")
with open("report.tex", "r") as fh:
    template = fh.read()

# Getting package versions
template = template.replace("<SKLEARN_V>", plot.sklearn_version)

# Get scaling method
template = template.replace("<SCALING>", "Standard")

# Create PCA Summary figures (scores, loadings, variances)
plot.summary()
template = template.replace("<SUMMARY_FIGS>", "summary_figs.pdf")

# Get number of loadings in upper and lower quantiles
template = template.replace("<PC1_LOADINGS>", str(len(plot.test_data.pc1_loadings)))
template = template.replace("<PC2_LOADINGS>", str(len(plot.test_data.pc2_loadings)))

# Plot scores with Hoteling's T2 Confidence Interval
# template = template.replace("<HOTELLINGS_SCORES>", "")
template = template.replace("<CONTROL>", str(plot.test_data.original_control))
template = template.replace("<CASE>", str(plot.test_data.original_case))
template = template.replace("<NUM_CONTROL>", str(plot.test_data.num_control))
template = template.replace("<NUM_CASE>", str(plot.test_data.num_case))

# Plot ranked loadings
plot.ranked_loadings()
template = template.replace("<UPPER_QUANTILE>", str(plot.upper_quantile))
template = template.replace("<LOWER_QUANTILE>", str(plot.lower_quantile))
template = template.replace("<RANKED_LOADINGS>", "ranked_loadings.pdf")

# tab = np.arange(16).reshape(4, 4)
# table_text = ""
# for row in tab:
#     table_text += "    " + " & ".join([str(x) for x in row]) + "\\\\\n"
# table_text = table_text[:-3]

# template = template.replace("<TABLE>", table_text)

# Save the completed template
with open("report_complete.tex", "w") as fh:
    fh.write(template)



subprocess.run(["pdflatex", "report_complete.tex"], shell=True)


# tempfile - to create a temporary directory for log files