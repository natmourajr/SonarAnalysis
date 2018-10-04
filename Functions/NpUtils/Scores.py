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

def recall_score_novelty(trgt, model_output, novelty_cls, class_labels, average=None):
    recall = list()
    keys = class_labels.keys()
    keys.sort()
    for cls in keys:
        if cls == novelty_cls:
            continue
        print trgt
        correct_trgt = trgt[trgt == cls]
        cls_out = model_output[trgt == cls]
        correct_outputs = cls_out == correct_trgt
        # print sum(correct_trgt)
        # if sum(correct_trgt) == 0:
        #     print trgt
        #     print correct_trgt
        #     raise NotImplementedError

        recall.append(sum(correct_outputs) / float(len(correct_trgt)))



    # recall = sklearn.metrics.recall_score(trgt, model_output, average=average)
    return recall


def confusionMatrix(y_true, y_pred, labels=None, sample_weight=None):
    return sklearn.metrics.confusion_matrix(y_true, y_pred, labels, sample_weight)


def trigger_score(trgt, predicted, novelty_class):
    known_events = predicted[trgt != novelty_class]
    known_right_events = known_events[known_events != novelty_class]

    if len(known_events) != 0:
    #known_events = trgt[predicted != novelty_class]
    #return float(len(known_events))/len(trgt)
        return float(len(known_right_events))/len(known_events)
    else:
        return 0

