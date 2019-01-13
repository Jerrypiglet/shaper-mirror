import os
import pickle


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def write_pkl(obj, fname):
    if not fname.endswith(".pkl"):
        fname += ".pkl"
    with open(fname, 'wb') as fid:
        pickle.dump(obj, fid, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(fname):
    with open(fname, 'rb') as fid:
        return pickle.load(fid)
