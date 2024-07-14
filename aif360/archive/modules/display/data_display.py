import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


class DataDisplay:
    ''' class to display certain aspects of the input data provided. Can either
    output to the provided stdout (depending on how the provided
    on how the supplied protocol is configured).

    params:
        result: The dataframe to display
    '''

    def __init__(self, result: pd.DataFrame) -> None:
        self.df = result


    def plot_subplot(self, 
                    nrows: int,
                    ncols: int, 
                    suptitle: str,
                    plot_conf: dict,
                    sharex: bool = False,
                    sharey: bool = False,
                    figsize: tuple =(15, 6),
                    ) -> None:
        ''' Orchestration method to plot multiple subplots based on the provided plottingargs 
        
        params:
            nrows: Number of rows for the subplots
            ncols: Number of columns for the subplots
            suptitle: The general title for the whole image
            plot_conf: The content for the plots
            sharex: Defines if the x-axis is to be shared
            sharey: Defines if the x-axis is to be shared

        returns: None
        '''
        # Define the current options for plotting 
        plot_options = {'hist': sns.histplot, 'boxp': sns.boxplot, 'bar': sns.barplot}
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, 
                        sharex=sharex, sharey=sharey
                        )
        # Set the general title
        fig.suptitle(suptitle)

        cur_row, cur_col = 0, 0
        for ind, cfg in plot_conf.items():
            # Check which plot is to be plot
            to_plot = cfg['dia_type']
            cur_cfg = {k: v for k, v in cfg.items() if k != 'dia_type'}
            # Plot the current plot
            ax = plt.subplot(nrows, ncols, ind)
            # Define the details which can be set per subplot individually
            plt_details = {'title': ax.set_title, 'x_label': ax.set_xlabel, 'y_label': ax.set_ylabel,
                            'x_ticks': ax.set_xticks, 'legend': ax.legend}
            # Set additional suplot details, if any were provided
            for det in plt_details:
                if det in cfg:
                    plt_details[det](cfg[det])

            dia_kwargs = {k: i for k, i in cur_cfg.items() if k not in plt_details}
            # Plot the according diagram
            plot_options[to_plot](data=self.df, ax=ax, **dia_kwargs)
            # Determine the current pos & update the rows, if necessary
            if cur_col == ncols:
                cur_row += 1
                cur_col = 0     

        plt.tight_layout()
        plt.show()

        return
    

    def plot_confusion_matrix(self, predicted: pd.Series, actual: pd.Series) -> None:
        ''' Plots a simple confusion matrix
        
        params:
            predicted: The predictions
            actual: The actual targets
        
        returns:
            None
        '''

        re_cf = sklearn.metrics.confusion_matrix(actual, predicted)
        re_cf_displ = sklearn.metrics.ConfusionMatrixDisplay(re_cf)

        re_cf_displ.plot()
        plt.show()

        return