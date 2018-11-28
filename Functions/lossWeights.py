import numpy as np

def getGradientWeights(y_train):
    """Calculate values for weighting the loss function for each class.

       Args:
           y_train: array containing the target values that will be used
           for training
       Returns:
           dictionary mapping class indices to a weight for each class
    """
    cls_indices, event_count = np.unique(np.array(y_train), return_counts=True)
    min_class = min(event_count)
    return {cls_index: float(min_class) / cls_count for cls_index, cls_count in zip(cls_indices, event_count)}