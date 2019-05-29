import math
import gzip
import time
import uuid
from contextlib import contextmanager
import os
import shutil
import unidecode
import warnings


Cache = {}


def mufidecode(string):
    retval = []

    for char in string:
        codepoint = ord(char)

        if codepoint < 0x80:  # Basic ASCII
            retval.append(str(char))
            continue

        if 0xd800 <= codepoint <= 0xdfff:
            warnings.warn("Surrogate character %r will be ignored. "
                          "You might be using a narrow Python build." % (char,),
                          RuntimeWarning, 2)

        section = codepoint >> 8  # Chop off the last two hex digits
        position = codepoint % 256  # Last two hex digits

        try:
            table = Cache[section]
        except KeyError:
            try:
                mod = __import__('unidecode.x%03x' % (section), globals(), locals(), ['data'])
            except ImportError:
                Cache[section] = None
                retval.append(char)
                continue
                
            Cache[section] = table = mod.data

        if table and len(table) > position:
            retval.append(table[position])
            continue

        # If not found
        retval.append(char)
        continue

    return ''.join(retval)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# What follows comes from the nice https://github.com/emanjavacas/pie/blob/master/pie/utils.py
@contextmanager
def tmpfile(parent='/tmp/'):
    fid = str(uuid.uuid1())
    tmppath = os.path.join(parent, fid)
    yield tmppath
    if os.path.isdir(tmppath):
        shutil.rmtree(tmppath)
    else:
        os.remove(tmppath)


def add_gzip_to_tar(string, subpath, tar):
    with tmpfile() as tmppath:
        with gzip.GzipFile(tmppath, 'w') as f:
            f.write(string.encode())
        tar.add(tmppath, arcname=subpath)


def get_gzip_from_tar(tar, fpath):
    return gzip.open(tar.extractfile(fpath)).read().decode().strip()


def ensure_ext(path, ext, infix=None):
    """
    Compute target path with eventual infix and extension
    >>> ensure_ext("model.pt", "pt", infix="0.87")
    'model-0.87.pt'
    >>> ensure_ext("model.test", "pt", infix="0.87")
    'model-0.87.test.pt'
    >>> ensure_ext("model.test", "test", infix="pie")
    'model-pie.test'
    """
    path, oldext = os.path.splitext(path)

    # normalize extension
    if ext.startswith("."):
        ext = ext[1:]
    if oldext.startswith("."):
        oldext = oldext[1:]

    # infix
    if infix is not None:
        path = "-".join([path, infix])

    # add old extension if not the same as the new one
    if oldext and oldext != ext:
        path = '.'.join([path, oldext])

    return '.'.join([path, ext])
