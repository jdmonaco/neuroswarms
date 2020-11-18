"""
Tools for datapath construction for HDF-5 files.
"""

import os
from os.path import splitdrive
import genericpath


def join(path, *paths):
    """
    Modified version of os.path.join that preserves '/' across OSes.
    """
    path = os.fspath(path)
    if isinstance(path, bytes):
        sep = b'/'
        seps = b'\\/'
        colon = b':'
    else:
        sep = '/'
        seps = '\\/'
        colon = ':'
    try:
        if not paths:
            path[:0] + sep  #23780: Ensure compatible data type even if p is null.
        result_drive, result_path = splitdrive(path)
        for p in map(os.fspath, paths):
            p_drive, p_path = splitdrive(p)
            if p_path and p_path[0] in seps:
                # Second path is absolute
                if p_drive or not result_drive:
                    result_drive = p_drive
                result_path = p_path
                continue
            elif p_drive and p_drive != result_drive:
                if p_drive.lower() != result_drive.lower():
                    # Different drives => ignore the first path entirely
                    result_drive = p_drive
                    result_path = p_path
                    continue
                # Same drive in different case
                result_drive = p_drive
            # Second path is relative to the first
            if result_path and result_path[-1] not in seps:
                result_path = result_path + sep
            result_path = result_path + p_path
        ## add separator between UNC and non-absolute path
        if (result_path and result_path[0] not in seps and
            result_drive and result_drive[-1:] != colon):
            return result_drive + sep + result_path
        return result_drive + result_path
    except (TypeError, AttributeError, BytesWarning):
        genericpath._check_arg_types('join', path, *paths)
        raise

def split(p):
    """
    Modified version of os.path.split that preserves '/' across OSes.
    """
    p = os.fspath(p)
    if isinstance(p, bytes):
        seps = b'\\/'
    else:
        seps = '\\/'
    # seps = _get_bothseps(p)
    d, p = splitdrive(p)
    # set i to index beyond p's last slash
    i = len(p)
    while i and p[i-1] not in seps:
        i -= 1
    head, tail = p[:i], p[i:]  # now tail has no slashes
    # remove trailing slashes from head, unless it's all slashes
    head = head.rstrip(seps) or head
    return d + head, tail
