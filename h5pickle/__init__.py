"""
Pickle-able hdf5 files, groups, and datasets. A cache is provided to avoid
reopening files.
"""

import json
import h5py
from cachetools import LRUCache
from threading import RLock


class LRUFileCache(LRUCache):
    """An LRU-cache that tries to close open files on eviction."""
    def popitem(self):
        key, val = LRUCache.popitem(self)
        # Try to close file
        try:
            val.close()
        except AttributeError:  # catch close() not being defined
            pass
        return key, val


cache = LRUFileCache(1000)
lock = RLock()


def h5py_wrap_type(obj):
    """Produce our objects instead of h5py default objects."""
    if isinstance(obj, h5py.Dataset):
        return Dataset(obj)
    elif isinstance(obj, h5py.Group):
        return Group(obj.id)
    elif isinstance(obj, h5py.File):
        return File(obj.id)  # TODO: Warn that this messes with the cache
    elif isinstance(obj, h5py.Datatype):
        return obj  # Not supported for pickling yet
    else:
        return obj  # Just return, since we want to wrap h5py.Group.get too


class Dataset:
    """
    Wrap dataset to allow for pickling and reopening files
    closed by cache eviction.
    """

    def __init__(self, dataset):
        self.file_name = dataset.file.id.name.decode('utf-8')
        self.file_mode = dataset.file.mode
        self.dataset_name = dataset.name
        self.dataset = dataset

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['dataset']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self = File(self.file_name, mode=self.file_mode)[self.dataset_name]

    def __getattr__(self, name):
        self = File(self.file_name, mode=self.file_mode)[self.dataset_name]
        return getattr(self.dataset, name)

    def __getitem__(self, item):
        self = File(self.file_name, mode=self.file_mode)[self.dataset_name]
        return self.dataset[item]


class Group(h5py.Group):
    """Overwrite group to allow for pickling, and to create new groups and
    datasets of the right type (i.e. the ones defined in this module).
    """

    def __getstate__(self):
        """Save the current name and a reference to the root file object."""
        return {'name': self.name, 'file': self.file_info}

    def __setstate__(self, state):
        """File is reopened by pickle. Create a dataset and
        steal its identity"""
        self.__init__(state['file'][state['name']].id)

    def __getitem__(self, name):
        obj = h5py_wrap_type(h5py.Group.__getitem__(self, name))
        # If it is a group, copy the current file info
        if isinstance(obj, Group):
            obj.file_info = self.file_info
        return obj


def arghash(*args, **kwargs):
    return hash(json.dumps(args) + json.dumps(kwargs, sort_keys=True))


class File(h5py.File):
    """A subclass of h5py.File that implements a memoized cache and pickling.
    Use this if you are going to be creating h5py.Files of the same file often.

    Pickling is done not with __{get,set}state__ but with __getnewargs_ex__
    which produces the arguments to supply to the __new__ method. This is
    required to allow for memoization of unpickled values.
    """

    def __init__(self, *args, **kwargs):
        """We skip the init method, since it is called at object creation time
        by __new__. This is necessary to have both pickling and caching."""
        pass

    def __new__(cls, *args, **kwargs):
        """Create a new File object with the h5open function, which memoizes
        the file creation. Test if it is still valid and otherwise create a
        new one.
        """
        with lock:
            skip_cache = kwargs.pop('skip_cache', False)
            hsh = arghash(*args, **kwargs)
            if skip_cache or hsh not in cache:
                self = object.__new__(cls)
                h5py.File.__init__(self, *args, **kwargs)
                # Store args and kwargs for pickling
                self.init_args = args
                self.init_kwargs = kwargs
                self.skip_cache = skip_cache
                if not skip_cache:
                    cache[hsh] = self
            else:
                self = cache[hsh]
            self.hsh = hsh
            return self

    def __getitem__(self, name):
        obj = h5py_wrap_type(h5py.Group.__getitem__(self, name))
        # If it is a group, copy the current file info
        if isinstance(obj, Group):
            obj.file_info = self
        return obj

    def __getstate__(self):
        pass

    def __getnewargs_ex__(self):
        kwargs = self.init_kwargs.copy()
        kwargs['skip_cache'] = self.skip_cache
        return self.init_args, kwargs

    def close(self):
        """Override the close function to remove the file also from
        the cache."""
        with lock:
            h5py.File.close(self)
            for key in list(cache.keys()):
                if cache[key] == self.hsh:
                    del cache[key]
