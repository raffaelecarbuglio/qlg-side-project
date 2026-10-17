"""ABI checks use a C fixture; the separate optional test uses real EPRL."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from qlg.backend import NativeBackend
from qlg.jobs import prepare, worker
from qlg.analysis import collect


class BackendABITest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("gcc"), "gcc needed for C ABI fixture")
    def test_column_major_copy_and_lifetime(self):
        # Deliberately unequal dimensions reveal an incorrect order/reversal.
        code = r'''
        #include <stdint.h>
        #include <stdbool.h>
        #include <stddef.h>
        #include <stdlib.h>
        struct config {int verbosity, accuracy, max_two_spin; size_t memory;};
        struct tensor {uint8_t num_keys; size_t dims[5]; ptrdiff_t strides[5];
                       size_t dim; double *d; void *tag;};
        bool sl2cfoam_is_MPI(void) {return false;}
        void sl2cfoam_init_conf(char *path, double gamma, struct config *c) {
            if (gamma != 1.2 || c->accuracy != 2 || c->memory != 32) abort();
        }
        void sl2cfoam_free(void) {}
        struct tensor *sl2cfoam_vertex_fullrange(int *js, int dl, int result) {
            if (js[0] != 2 || js[9] != 3 || dl != 1 || result != 1) abort();
            size_t dims[] = {3,4,5,3,3};
            struct tensor *t = calloc(1, sizeof(*t));
            t->num_keys = 5; t->dim = 1;
            for(int k=0;k<5;k++) {
                t->dims[k]=dims[k]; t->strides[k]=t->dim; t->dim*=dims[k];
            }
            t->d = malloc(t->dim*sizeof(double));
            for(size_t i=0;i<t->dim;i++) t->d[i]=i+0.25;
            return t;
        }
        void sl2cfoam_vertex_free(struct tensor *t) {free(t->d); free(t);}
        '''
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "abi.c"
            source.write_text(code)
            library = root / "abi.so"
            subprocess.run(["gcc", "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
            spins = [2, 4, 6, 4, 6, 6, 2, 5, 5, 3]
            with NativeBackend(library, root, 1.2, 2, 100, 32) as native:
                tensor = native.vertex(spins, 1)
            self.assertEqual(tensor.shape, (3, 4, 5, 3, 3))
            self.assertEqual(tensor[2, 1, 3, 0, 2], 2 + 3*1 + 12*3 + 60*0 + 180*2 + 0.25)
            np.testing.assert_array_equal(tensor.ravel(order="F"), np.arange(540) + 0.25)


class NativeEPRLTest(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("SL2CFOAM_LIBRARY") and os.environ.get("SL2CFOAM_TABLES"),
                         "Real EPRL requires SL2CFOAM_LIBRARY and SL2CFOAM_TABLES")
    def test_real_delta4_against_published_dl0(self):
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            config = Path(tmp) / "native.json"
            config.write_text('{"model":"delta4","gamma":1.2,"two_j":2,"shells":[0]}')
            prepare(config, run, os.environ["SL2CFOAM_LIBRARY"], os.environ["SL2CFOAM_TABLES"])
            # Isolate native aborts/segfaults, just as the production local runner does.
            import sys
            result = subprocess.run([sys.executable, "-m", "qlg", "worker", str(run), "--dl", "0"],
                                    cwd=root, capture_output=True, text=True,
                                    env=dict(os.environ, OMP_NUM_THREADS="1"), timeout=300)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(collect(run, reference=True, rtol=1e-5)["reference_pass"])


if __name__ == "__main__":
    unittest.main()
