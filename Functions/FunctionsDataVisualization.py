'''
    This file contents some functions for Data Visualization
'''

import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
import pandas as pd
import numpy as np

from itertools import cycle
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


def plotScores(scores_dataframe,
               class_labels,
               title,
               x_label="",
               y_label=("Classification Efficiency", "SP Index"),
               y_inf_lim = 0.80,
               figsize=(15,8)):
    molten_scores = scores_dataframe.melt(id_vars=['Class'])
    order_cats = molten_scores['variable'].unique()

    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)

    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    plt.rcParams['legend.numpoints'] = 1
    plt.rc('legend', **{'fontsize': 15})
    plt.rc('font', weight='bold')

    markers = ['^', 'o', '+', 's', 'p', 'o', '8', 'D', 'x']
    linestyles = ['-', '-', ':', '-.']
    colors = ['k', 'b', 'g', 'y', 'r', 'm', 'y', 'w']

    def cndCycler(cycler, std_marker, condition, data):
        return [std_marker if condition(var) else cycler.next() for var in data]

    sns.pointplot(y='value', x='variable', hue='Class',
                  order=order_cats,
                  data=molten_scores,
                  markers=cndCycler(cycle(markers[:-1]),
                                    markers[-1],
                                    lambda x: x in class_labels.values(),
                                    molten_scores['Class'].unique()),
                  linestyles=cndCycler(cycle(linestyles[:-1]), linestyles[-1],
                                       lambda x: x in class_labels.values(),
                                       molten_scores['Class'].unique()),
                  palette=cndCycler(cycle(colors[1:]), colors[0],
                                    lambda x: not x in class_labels.values(),
                                    molten_scores['Class'].unique()),
                  dodge=.5,
                  scale=1.7,
                  errwidth=2.2, capsize=.1, ax=ax)

    leg_handles = ax.get_legend_handles_labels()[0]

    ax.legend(handles=leg_handles,
              ncol=6, mode="expand", borderaxespad=0., loc=3)
    ax.set_xlabel(x_label, fontsize=1, weight='bold')
    ax.set_title(title, fontsize=25,
                 weight='bold')
    ax.set_ylabel(y_label[0], fontsize=20, weight='bold')
    ax.set_ylim([y_inf_lim, 1.0001])

    plt.xticks(rotation=25, ha='right')

    ax2 = ax.twinx()
    ax2.set_ylabel(y_label[1], fontsize=20, weight='bold')
    ax2.set_ylim([y_inf_lim, 1.0001])

    return fig


def plotLOFARgram(image,ax = None, filename = None, cmap = 'jet', colorbar=True):
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

        x= plt.imshow(image,
                   cmap=cmap, extent=[1, image.shape[1], image.shape[0], 1],
                   aspect="auto")

        plt.xlabel('Frequency bins', fontweight='bold')
        plt.ylabel('Time (seconds)', fontweight='bold')
        if colorbar:
            plt.colorbar(x)
        if not filename is None:
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            return

        return fig
    else:
        x = ax.imshow(image,
                   cmap=cmap, extent=[1, image.shape[1], image.shape[0], 1],
                   aspect="auto")
        if colorbar:
            plt.colorbar(x, ax=ax)
        return


def plotHBar(x, y, hue, err, data, height=0.20, order=None, hue_order=None,
             color=None, capsize=0, ax=None, title='', x_label='',
             hue_label='', y_label = '', x_margin=10):
    seaborn.set_style('white')

    if color is None:
        color = ['#069af3', 'IndianRed', '#76cd26']
    n_hue = np.unique(data['hue'].values)
    indices = np.arange(len(np.unique(data[y].values)))  # the x locations for the groups

    def autolabel(rects, ax, model_op, xpos='center', ypos = 'up'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        xpos: indicates which side to place the text w.r.t. the edge of
        the bar. It can be one of the following {'center', 'right', 'left'}.

        ypos: indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'top', 'bottom'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ypos = ypos.lower()
        va = {'center': 'center', 'top': 'bottom', 'bottom': 'top'}
        ha = {'center': 'center', 'left': 'right', 'right': 'left'}
        offset = {'center': 0.5, 'top': 0.57, 'bottom': 0.43}  # x_txt = x + w*off

        for rect, std in zip(rects, model_op['std'].values):
            width = rect.get_width()
            ax.text(1.01 * width, rect.get_y() + rect.get_height() * offset[ypos],
                    '{0:.2f}'.format(round(width,2)) + u'\u00b1' + '{0:.2f}'.format(round(std,2)),
                    va=va[ypos], ha=ha[xpos], rotation=0)

    fig, ax = plt.subplots()
    x_inf = max(data[x].values)
    for group_i, (group_name, group) in enumerate(data.groupby(by=hue)):
        pos = indices + (group_i - n_hue / 2.0) * (height)

        rect = ax.barh(pos, group[x].values[::-1], height, xerr=group[err].values[::],
                       color=color[group_i], label=group_name, linewidth=0,
                       ecolor='black', capsize=capsize, error_kw={'elinewidth': 2.2})

        autolabel(rect, ax, group, "right", "top")

        min_mean = min(group['mean'].values)
        min_mean_std = group.loc[group['mean'] == min_mean, 'std'].values

        x_margin = x_margin
        x_low = ((min_mean - min_mean_std) // 10) * 10 - x_margin
        x_inf = x_low if x_low < x_inf else x_inf

    ax.set_xlim(x_inf, ax.get_xlim()[1] + 10)
    ax.set_ylim(indices[0] - height * n_hue / 2 - 0.1,
                indices[-1] + height * n_hue / 2 + 0.55)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12, fontweight='semibold')
    ax.set_xlabel(x_label, fontsize=12, fontweight='semibold')
    ax.set_xticks(np.arange(x_inf, 101, 10))
    ax.set_yticks(indices)
    ax.set_yticklabels(np.unique(data[y].values)[::-1])

    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.xaxis.grid(True)
    ax.legend(title=hue_label, fontsize='medium', markerscale=0.7,
              frameon=True, fancybox=True, shadow=True,
              bbox_to_anchor=(0.0, 1.025), loc="upper left", ncol=3)
    fig.tight_layout()
    plt.show()
