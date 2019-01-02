import os
import errno
import pickle


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def write_pkl(obj, fname):
    if not fname.endswith(".pkl"):
        fname += ".pkl"
    with open(fname, 'wb') as fid:
        pickle.dump(obj, fid, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(fname):
    with open(fname, 'rb') as fid:
        return pickle.load(fid)
