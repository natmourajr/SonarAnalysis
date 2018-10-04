import warnings
from copy import copy, deepcopy

import numpy as np
import keras
from keras.callbacks import ModelCheckpoint

class PersistentModelCheckpoint(ModelCheckpoint):
    def __init__(self, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, predef_thres=None):
        filepath = './' # Inherit BaseCheckpoint later
        super(PersistentModelCheckpoint, self).__init__(filepath, monitor='val_loss', verbose=0,
                                                     save_best_only=False, save_weights_only=False,
                                                     mode='auto', period=1)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    # Routine to consider n_inits procedure
                    if self.predef_thres is not None:
                        if self.monitor_op(current, self.predef_thres):
                            print('\nEpoch %05d: %s improved from predefined thresold: %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, current,
                                     self.predef_thres, filepath))
                        else:
                            print('\nEpoch %05d: %s did not improve from predefined threshold %0.5f' %
                                  (epoch + 1, self.monitor, self.predef_thres))
                            return
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.best_weights = self.model.get_weights()
                        else:
                            self.model.best = deepcopy(self.model)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.model_weights[epoch+1] = self.model.get_weights()
                else:
                    self.model.model_states[epoch+1] = self.model.get_weights()

# class ModelRestart(keras.callbacks.Callback):
#     def __init__(self, n_restarts=10):
#         super(ModelRestart, self).__init__()
#
#     #def on_epoch_end(self, epoch, logs=None):
#     def
#
#     def on_train_end(self, logs={}):
#         current = logs.get(self.monitor)
#         if current is None:
#             warnings.warn(
#                 'Restart conditioned on metric `%s` '
#                 'which is not available. Available metrics are: %s' %
#                 (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
#             )
#             return
#
#         current_max = np.array(current).max()
#         # if self.monitor_op(current_max, self.threshold):
#         #
#         # else:

#class ResumeTraining(keras.callbacks.Callback):
 #save hyperparameters
 #load if exists (saved with save_model)
 #get information of last epoch, maybe on recv_model name
 #resume on last epoch
