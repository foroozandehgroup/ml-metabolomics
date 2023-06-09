import Metabolomics_ML.gui.gui as gui
from Metabolomics_ML.tex_template.report_plot_funcs import PCAPlot
from Metabolomics_ML.tex_template.report_generator import generate

def main():
    app = gui.Window()
    app.mainloop()
    pcaplot = PCAPlot().run_all(app.guidata)
    generate(pcaplot)

if __name__ == "__main__":
    main()