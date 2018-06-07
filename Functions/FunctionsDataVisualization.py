'''
    This file contents some functions for Data Visualization
'''

import matplotlib.pyplot as plt
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


def plotConfusionMatrix(confusionMatrix,
                        class_labels,
                        filepath,
                        fontsize=15,
                        figsize=(10, 6),
                        cbar=True,
                        precision=2,
                        normalize=True):
    """Plots a confusion matrix as a heatmap using seaborn library functions.

    Args:
        confusion_matrix (numpy.ndarray): Resulting from
        sklearn.metrics.confusionMatrix or an array with similar shape
        class_labels (list): List containing the class labels in the order of the
        confusion matrix parameter
        figsize (tuple): A 2 item tuple, the first value with the horizontal size
        of the figure,
        the second with the vertical size. Defaults to (10,6).
        filepath (string): Saving folder for the resulting plot.
        fontsize (int): Font size for axes labels. Defaults to 15.
        cbar (bool): Whether to draw a colorbar, Defaults to True
        precision (int): Decimal portion length of confusion matrix tiles.
        Defaults to 1
    """

    if normalize:
        cm = confusionMatrix = confusionMatrix.astype('float') / confusionMatrix.sum(axis=1)[:, np.newaxis]
    else:
        cm = confusionMatrix

    cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_ylabel('True Label', fontweight='bold', fontsize=fontsize)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=fontsize)

    heatmap = sns.heatmap(cm, ax=ax, annot=True, fmt=".%s%%" % precision, cmap="Greys")

    plt.savefig('./Analysis/' + filepath)


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


def plotLOFARgram(image,filename = None):
    """Plot LOFARgram from an array of frequency spectre values

    Args:
    image (numpy.array): Numpy array with the frequency spectres along the second axis
    """

    fig = plt.subplots(figsize=(20, 20))
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.size'] = 30
    plt.rcParams['xtick.labelsize'] = 30
    plt.rcParams['ytick.labelsize'] = 30

    plt.imshow(image,
               cmap="jet", extent=[1, 400, image.shape[0], 1],
               aspect="auto")

    plt.xlabel('Frequency bins', fontweight='bold')
    plt.ylabel('Time (seconds)', fontweight='bold')

    if not filename is None:
        plt.savefig(filename)

    return plt