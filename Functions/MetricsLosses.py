from keras import backend as K

def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.abs(K.sum(y_true * K.log(y_true / y_pred), axis=-1))
