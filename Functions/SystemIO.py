import joblib
import datetime
import os
import shutil

def save(file, filepath):
    joblib.dump(file, filepath)


def load(filepath):
    return joblib.load(filepath)


def exists(filepath):
    return os.path.exists(filepath)


def remove(filepath, recursive = False):
    if recursive:
        shutil.rmtree(filepath)
        return
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
    return [folder for folder in os.listdir(datapath) if os.path.isdir(datapath + '/' + folder)]

def listfiles(datapath):
    return filterPaths(os.listdir(datapath))