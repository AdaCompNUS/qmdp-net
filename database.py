import numpy as np
import tables
import collections

try:
    import ipdb as pdb
except Exception:
    import pdb


class Datatable(collections.Sequence):
    """
    Wrapper around pytable node that supports non-unique indexing
    """
    def __init__(self, node):
        self.node = node

    def __len__(self):
        return len(self.node)

    def __getitem__(self, sliced):
        return self.node.__getitem__(sliced)

    def __repr__(self):
        return str(self.node)

    def unique_index(self, idx):
        i_unique, remap = np.unique(idx, return_inverse=True)
        if self.node.ndim > 1:
            vals_unique = self.node[i_unique,:]
        else:
            vals_unique = self.node[i_unique]
        return vals_unique[remap]


class Database(collections.defaultdict):
    """
    Wrapper around pytable database with cache
    """
    def __init__(self, filename=None, cache=None):
        super(Database, self).__init__()
        self.db = None
        self.filename = filename
        self.cache = ({} if cache is None else cache)

    def __getattr__(self, attr):
        return self.get(attr)

    def get(self, attr, **kwargs):
        try:
            return self.cache[str(attr)]
        except KeyError:
            return Datatable(self.db.get_node("/"+str(attr)))

    def get_all(self, attr):
        return self.db.get_node("/"+str(attr))[:]

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def __repr__(self):
        return str(self.db)

    def get_node(self, *args, **kwargs):
        return self.db.root.get_node(*args, **kwargs)

    def open(self, filename=None, mode='r'):
        self.close()
        if filename is not None:
            self.filename = filename
        self.db = tables.open_file(self.filename, mode=mode)

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def build_cache(self, cache_nodes):
        self.open()
        cache = {}
        for node in cache_nodes:
            node = str(node)
            try:
                cache[node] = self.get_all(node)
                print ("cached %s"%str(node))
            except tables.NoSuchNodeError:
                print ("cannot cache %s"%str(node))
        self.close()
        return cache
