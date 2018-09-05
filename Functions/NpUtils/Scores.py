"""
This module contains metrics implemented using numpy and sklearn utilities
"""
import numpy as np
import sklearn


def spIndex(recall, n_classes):
    return np.sqrt(np.sum(recall) / n_classes *\
                   np.power(np.prod(recall), 1.0 / float(n_classes))
                   )

def effAcc(recall, n_classes):
    return np.sum(recall) / n_classes


def recall_score(trgt, model_output, average=None):
    recall = sklearn.metrics.recall_score(trgt, model_output, average=average)
    return recall


def confusionMatrix(y_true, y_pred, labels=None, sample_weight=None):
    return sklearn.metrics.confusion_matrix(y_true, y_pred, labels, sample_weight)


def trigger_score(y_predicted, y_true, novelty_class):
    known_events = y_predicted[y_true != novelty_class]
    return float(len(known_events))/len(y_predicted)

