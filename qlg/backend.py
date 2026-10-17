"""Small ctypes interface to the pinned sl2cfoam-next C ABI.

One native-library instance per worker process. MPI builds are rejected: each
Slurm task owns one process and uses the library's OpenMP parallelism.
"""

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .spins import vertex_shape


class Config(ct.Structure):
    _fields_ = [("verbosity", ct.c_int), ("accuracy", ct.c_int),
                ("max_two_spin", ct.c_int), ("max_MB_mem_per_thread", ct.c_size_t)]


class Tensor(ct.Structure):
    _fields_ = [("num_keys", ct.c_uint8), ("dims", ct.c_size_t * 5),
                ("strides", ct.c_ssize_t * 5), ("dim", ct.c_size_t),
                ("d", ct.POINTER(ct.c_double)), ("tag", ct.c_void_p)]


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def backend_identity(library):
    path = Path(library).expanduser().resolve(strict=True)
    manifest = path.parent.parent / "build-info.json"
    info = json.loads(manifest.read_text()) if manifest.exists() else {}
    return {"library_sha256": digest(path), "build": info}


class NativeBackend:
    def __init__(self, library, data, gamma, accuracy, max_two_spin, memory_mb):
        self.library = Path(library).expanduser().resolve(strict=True)
        self.data = Path(data).resolve()
        if not self.data.is_dir():
            raise ValueError(f"Missing native data directory: {self.data}")
        self.lib = ct.CDLL(str(self.library))
        lib = self.lib
        lib.sl2cfoam_is_MPI.argtypes = []
        lib.sl2cfoam_is_MPI.restype = ct.c_bool
        if lib.sl2cfoam_is_MPI():
            raise ValueError("Use an MPI=0 build with the single-process worker")
        lib.sl2cfoam_init_conf.argtypes = [ct.c_char_p, ct.c_double, ct.POINTER(Config)]
        lib.sl2cfoam_init_conf.restype = None
        lib.sl2cfoam_free.argtypes = []
        lib.sl2cfoam_free.restype = None
        lib.sl2cfoam_vertex_fullrange.argtypes = [ct.POINTER(ct.c_int), ct.c_int, ct.c_int]
        lib.sl2cfoam_vertex_fullrange.restype = ct.POINTER(Tensor)
        lib.sl2cfoam_vertex_free.argtypes = [ct.POINTER(Tensor)]
        lib.sl2cfoam_vertex_free.restype = None
        self.conf = Config(0, accuracy, max_two_spin, memory_mb)
        lib.sl2cfoam_init_conf(os.fsencode(self.data), gamma, ct.byref(self.conf))
        self.closed = False

    def vertex(self, spins, dl):
        shape = vertex_shape(spins)
        if 0 in shape:
            raise ValueError(f"Inadmissible vertex spins: {spins}")
        js = (ct.c_int * 10)(*spins)
        ptr = self.lib.sl2cfoam_vertex_fullrange(js, dl, 1)  # RETURN, no native vertex cache
        if not ptr:
            raise RuntimeError("sl2cfoam-next returned a null tensor")
        try:
            t = ptr.contents
            if t.num_keys != 5 or tuple(t.dims) != shape or t.dim != int(np.prod(shape)):
                raise RuntimeError("Unexpected native tensor layout; check ABI and spin ordering")
            if tuple(t.strides) != (1,) + tuple(np.cumprod(shape[:-1])):
                raise RuntimeError("Unexpected native tensor strides")
            # Native layout is column-major with axes (i5,i4,i3,i2,i1).
            result = np.ctypeslib.as_array(t.d, shape=(t.dim,)).copy().reshape(shape, order="F")
            if not np.isfinite(result).all():
                raise FloatingPointError("Non-finite native amplitude")
            return result
        finally:
            self.lib.sl2cfoam_vertex_free(ptr)

    def close(self):
        if not self.closed:
            self.lib.sl2cfoam_free()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
