import warnings
import numpy as np
import keras

class RestartTraining(keras.callbacks.Callback):
    def __init__(self, monitor, threshold, mode = 'auto'):
        super(RestartTraining, self).__init__()
        self.monitor = monitor
        self.threshold = threshold

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('RestartTraining mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if self.monitor not in ['acc', 'spIndex', 'effAcc']:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        self.mode = mode

    def on_train_end(self, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Restart conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return

        current_max = np.array(current).max()
        # if self.monitor_op(current_max, self.threshold):
        #
        # else:

#class ResumeTraining(keras.callbacks.Callback):
 #save hyperparameters
 #load if exists (saved with save_model)
 #get information of last epoch, maybe on recv_model name
 #resume on last epoch
