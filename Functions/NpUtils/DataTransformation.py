"""
This module contains utilities to handle and transform lofar data
"""
import keras
import numpy as np


def trgt2categorical(trgt, n_classes):
    return keras.utils.to_categorical(trgt, num_classes=n_classes)


def lofar2image(all_data, all_trgt, index_info, class_labels, class_window, stride, verbose=False, run_split_info=None,
                debug = False,
                filepath=None,
                dtype=np.float64):
    if run_split_info is not None:
        fold_runs = np.concatenate([np.extract([np.isin(run, index_info).all() for run in cls_runs], cls_runs)
                                    for cls_runs in run_split_info.runs.values()])
        pruned_indexes = np.concatenate([range(run[0], run[-1] - class_window, stride) for run in fold_runs])
    else:
        # Extract the indexes of elements of a given eventclass
        i_class_extract = lambda class_type: np.where(all_trgt == class_type)[0]
        # Generate an iterator for sample extraction
        pruned_iterator = lambda class_type: range(i_class_extract(class_type)[0],
                                                   i_class_extract(class_type)[-1] - class_window,
                                                   stride)
        pruned_indexes = np.concatenate(map(lambda class_key: pruned_iterator(class_key), class_labels))

    data_shape = (pruned_indexes.shape[0],
                  class_window,
                  all_data.shape[1],
                  1)
    if not filepath is None:
        image_data = np.memmap(filename=filepath, shape=data_shape, mode='w+', dtype=dtype)
    else:
        image_data = np.zeros(shape=data_shape)

    trgt_image = np.zeros(shape=data_shape[0])

    for image_index, spectre_index in enumerate(pruned_indexes):
        new_data = all_data[spectre_index:spectre_index + class_window, :]
        new_data = np.array(new_data.reshape(new_data.shape[0], new_data.shape[1], 1), np.float64)
        image_data[image_index] = new_data
        trgt_image[image_index] = all_trgt[spectre_index]

    if verbose:
        print "Image dataset shape:"
        for class_key in class_labels:
            print "     %s (%i) samples: %i" % (class_labels[class_key], class_key, len(pruned_iterator(class_key)))
        print ""
        print "     Samples Total:   %s" % image_data.shape[0]
        print "     Sample length:   %s" % image_data.shape[1]
        print "     Sample width:    %s" % image_data.shape[2]
        print ""
        if image_data.nbytes < 10 ** 6:
            print "     File size:   Kb"
        elif image_data.nbytes < 10 ** 9:
            print "     File size:   %i Mb" % int(image_data.nbytes / 10 ** 6)
        else:
            print "     File size:   Gb"

    return [image_data, trgt_image]