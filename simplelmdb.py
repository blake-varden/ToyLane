import lmdb
import msgpack
import msgpack_numpy as m
m.patch()
import numpy as np
import random
import shutil
import os
class Simplelmdb(object):
    def __init__(self, path):
        self.path = path
        self.db = lmdb.open(path, map_size=10**13)

    def items(self):
        with self.db.begin() as txn:
            cur = txn.cursor()
            cur.set_range('')
            for k, v in cur:
                yield k, msgpack.unpackb(v)

    def keys(self):
        return (k for k,v in self.items())

    def values(self):
        return (v for k,v in self.items())

    def get(self, k, *args, **kwargs):
        with self.db.begin() as txn:
            val = txn.get(k)
            if val is None and len(args) > 0:
                return args[0]
            return msgpack.unpackb(val)

    def get_keys(self, keys, *args, **kwargs):
    	"""
    	second arg can be the default
    	"""
	with self.db.begin() as txn:
		for k in keys:
			val = txn.get(k)
			if val is None and len(args) > 0:
				yield args[0]
			yield msgpack.unpackb(val)


    def put(self, k, v):
        with self.db.begin(write=True) as txn:
            val = msgpack.packb(v)
            txn.put(k, val)

    def count(self):
        return self.db.stat()['entries']

    def take(self, n):
        items = self.items()
        for i in range(n):
            yield items.next()

    def first(self):
        return self.items().next()