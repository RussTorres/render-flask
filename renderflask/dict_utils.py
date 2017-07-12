from collections import Mapping
from itertools import chain
from operator import add
import collections
_FLAG_FIRST = object()

def flattenDict(d, join=add, lift=lambda x:x):
    results = []
    def visit(subdict, results, partialKey):
        for k,v in subdict.items():
            newKey = lift(k) if partialKey==_FLAG_FIRST else join(partialKey,lift(k))
            if isinstance(v,Mapping):
                visit(v, results, newKey)
            else:
                results.append((newKey,v))
    visit(d, results, _FLAG_FIRST)
    return results

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)