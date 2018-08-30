'''
    This file contents some functions for Data Visualization
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def add_subplot_axes(ax,rect,axisbg='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    subax = fig.add_axes([x,y,width,height],axisbg=axisbg)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return

def plotConfusionMatrix(predictions,
                        trgt,
                        class_labels,
                        ax,
                        annot=True,
                        normalize=True,
                        fontsize=15,
                        figsize = (10,10),
                        cbar_ax=None,
                        precision=2,
                        set_label = True):
    """Plots a confusion matrix from the network output

        Args:
            predictions (numpy.ndarray): Estimated target values
            trgt (numpy.ndarray) : Correct target values
            class_labels (dict): Mapping between target values and class names
            confusion matrix parameter. If None
            fontsize (int): Size of the annotations inside the matrix tiles
            figsize (tuple): A 2 item tuple, the first value with the horizontal size
            of the figure. Defaults to 15
            the second with the vertical size. Defaults to (10,6).
            precision (int): Decimal portion length of the tiles annotations.
            Defaults to 2
            set_label (bool): Whether to draw axis labels. Defaults to True
        """

    confusionMatrix = confusion_matrix(trgt, predictions)
    if normalize:
        cm = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]
    else:
        cm = confusionMatrix

    cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    sns.heatmap(cm, ax=ax, annot=annot, cbar_ax=cbar_ax, annot_kws={'fontsize': fontsize}, fmt=".%s%%" % precision,
                cmap="Greys")

    if set_label:
        ax.set_ylabel('True Label', fontweight='bold', fontsize=fontsize)
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=fontsize)


def plotMetrics(y,
                x,
                hue,
                data,
                markers,
                colors,
                x_label=None,
                y_label=None,
                dodge=True,
                figsize=(15, 8)):
    if not x_label:
        x_label = x
    if not y_label:
        y_label = y

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)

    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.numpoints'] = 1
    plt.rc('legend', **{'fontsize': 15})
    plt.rc('font', weight='bold')

    sns.pointplot(y=y, x=x, hue=hue, data=data, capsize=.02,
                  dodge=dodge, markers=markers)

    ax.set_xlabel(x_label, fontsize=18, weight='bold')
    ax.set_ylabel(y_label, fontsize=18, weight='bold')

    # SAVE THE FIGURE
    raise NotImplementedError


def plotLOFARgram(image,ax = None, filename = None):
    """Plot LOFARgram from an array of frequency spectre values

    Args:
    image (numpy.array): Numpy array with the frequency spectres along the second axis
    """
    if ax is None:
        fig = plt.figure(figsize=(20, 20))
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['font.size'] = 30
        plt.rcParams['xtick.labelsize'] = 30
        plt.rcParams['ytick.labelsize'] = 30

        plt.imshow(image,
                   cmap="jet", extent=[1, image.shape[1], image.shape[0], 1],
                   aspect="auto")

        plt.xlabel('Frequency bins', fontweight='bold')
        plt.ylabel('Time (seconds)', fontweight='bold')

        if not filename is None:
            plt.savefig(filename)
            plt.close()
            return

        return fig
    else:
        x = ax.imshow(image,
                   cmap="jet", extent=[1, 400, image.shape[0], 1],
                   aspect="auto")
        plt.colorbar(x, ax = ax)
        return

