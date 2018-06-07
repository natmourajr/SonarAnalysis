import joblib
import datetime
import os


def save(file, filepath):
    joblib.dump(file, filepath)


def load(filepath):
    return joblib.load(filepath)


def exists(filepath):
    return os.path.exists(filepath)


def remove(filepath):
    os.remove(filepath)


def mkdir(filepath, overwrite=False):
    if exists(filepath):
        if overwrite:
            os.remove(filepath)
        else:
            print "Folder already exists. To overwrite it, set overwrite to true"
    os.makedirs(filepath)


def filterPaths(folder_list):
    for folder in folder_list:
        if folder[0] != '.':
            yield folder


def listfolders(datapath):
    return filterPaths(os.listdir(datapath))