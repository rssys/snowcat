import os
import os.path as osp
import pickle
import fcntl

def try_cache_obj(save_filepath, obj):
    with open(save_filepath, "wb") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except:
            return
        pickle.dump(obj, f)

def find_obj_cache(save_filepath):
    try:
        with open(save_filepath, "rb") as f:
            return pickle.load(f)
    except:
        return None
