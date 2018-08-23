import numpy as np
import sklearn
import keras.backend as K


def ind_SP(y_true, y_pred):
    num_classes = 4

    true_positives = K.sum(K.cast(y_true * K.one_hot(K.argmax(y_pred, axis=1), num_classes), dtype='float32'))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    sp_tf = K.sqrt(K.mean(recall) * K.prod(K.pow(recall, 1 / num_classes)))

    return sp_tf


def spIndex(recall, n_classes):
    return np.sqrt(np.sum(recall) / n_classes *\
                   np.power(np.prod(recall), 1.0 / float(n_classes))
                   )

# def tfSpIndex(trgt,model_output, average = None):
#     recall = sklearn.metrics.recall_score(trgt, model_output, average=average)
#     sp = np.sqrt(np.sum(recall) / n_classes * \
#                    np.power(np.prod(recall), 1.0 / float(n_classes))
#                    )
#
#     return K.variable(sp)


def effAcc(recall, n_classes):
    return np.sum(recall) / n_classes


def recall_score(trgt, model_output, average=None):
    return sklearn.metrics.recall_score(trgt, model_output, average=None)


def confusionMatrix(y_true, y_pred, labels=None, sample_weight=None):
    return sklearn.metrics.confusion_matrix(y_true, y_pred, labels, sample_weight)


def trigger(y_predicted, y_true, novelty_class):
    #novelty_events = y_predicted[y_true == novelty_class]
    known_events = y_predicted[y_true != novelty_class]

    #detected_novelties = novelty_events[novelty_events == novelty_class]
    #correct_events = known_events[known_events != novelty_class]
    return float(len(known_events))/len(y_predicted)

