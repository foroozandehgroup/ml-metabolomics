from sklearn import __version__ as sklearn_version
import subprocess
import os
from pathlib import Path
from Metabolomics_ML.tex_template.report_plot_funcs import PCAPlot

def generate(plot: PCAPlot):

    # Loading the template
    
    p = Path(__file__).with_name("report.tex")

    with p.open("r") as fh:
        template = fh.read()
    
    # Get final directory
    save_dir = plot.pcaplot.guidata.save_dir_path

    # Save graphics path
    template = template.replace("<GRAPHICS_PATH>", rf"{save_dir}/".replace("\\", "/"))

    # Get report author
    template = template.replace("<NAME>", plot.pcaplot.guidata.name)

    # Getting package versions
    template = template.replace("<SKLEARN_V>", sklearn_version)

    # Get scaling method
    template = template.replace("<SCALING>", "Standard")

    # Create PCA Summary figures (scores, loadings, variances)
    template = template.replace("<SUMMARY_FIGS>", "summary_figs.pdf")

    # Get number of loadings in upper and lower quantiles
    template = template.replace("<PC1_LOADINGS>", str(len(plot.pcaplot._sig_labels_list[0])))
    template = template.replace("<PC2_LOADINGS>", str(len(plot.pcaplot._sig_labels_list[1])))

    # Plot scores with Hoteling's T2 Confidence Interval
    # template = template.replace("<HOTELLINGS_SCORES>", "")
    template = template.replace("<CONTROL>", str(plot.pcaplot.original_control))
    template = template.replace("<CASE>", str(plot.pcaplot.original_case))
    template = template.replace("<NUM_CONTROL>", str(plot.pcaplot.num_control))
    template = template.replace("<NUM_CASE>", str(plot.pcaplot.num_case))

    # Plot ranked loadings
    template = template.replace("<UPPER_QUANTILE>", str(plot.pcaplot.upper_quantile))
    template = template.replace("<LOWER_QUANTILE>", str(plot.pcaplot.lower_quantile))
    template = template.replace("<RANKED_LOADINGS>", "ranked_loadings.pdf")

    # Plot p-values table
    template = template.replace("<SIG_P_VALUES>", plot.p_values_tables(pcaplot=plot.pcaplot, top_loadings=True))
    template = template.replace("<ALL_P_VALUES>", plot.p_values_tables(pcaplot=plot.pcaplot))

    # Plot ttest box plots
    ttest_boxplots = ""

    ttests = [file for file in os.listdir(save_dir) if file.startswith("ttest")]

    for fig in ttests:
        ttest_boxplots += rf"""
        \begin{{figure}}[h!]
        \begin{{center}}
            \includegraphics{{{fig}}}
        \end{{center}}
        \caption{{Bar plots from PC{fig[-5]} only. {str(plot.pcaplot.original_control)}: green. 
        {str(plot.pcaplot.original_case)}: blue. Student's t-test p-values \textless 0.05, 0.01, 0.001 
        are represented by *, **, and *** respectively. Plots show mean +/- SEM.}}
        \end{{figure}}

        \newpage
        """
    
    print(ttest_boxplots)

    template = template.replace("<TTEST_BOXPLOTS>", ttest_boxplots)

    # Save the completed template
    with open(rf"{save_dir}\report_complete.tex", "w") as fh:
        fh.write(template)

    subprocess.run(["pdflatex", "-output-directory", save_dir, rf"{save_dir}/report_complete.tex"], shell=True)

if __name__ == "__main__":
   pass


# tempfile - to create a temporary directory for log files