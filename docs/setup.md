# Native backend setup

The backend is the existing Lorentzian EPRL implementation in sl2cfoam-next.
Its C ABI is pinned to revision:

```text
052e4346028870bd76f69a3034e6cae8defb8f7f
```

Clone it on a machine with network access:

```bash
git clone https://github.com/qg-cpt-marseille/sl2cfoam-next.git
git -C sl2cfoam-next checkout 052e4346028870bd76f69a3034e6cae8defb8f7f
```

Use the upstream [Ubuntu instructions](https://github.com/qg-cpt-marseille/sl2cfoam-next/blob/052e4346028870bd76f69a3034e6cae8defb8f7f/QUICKSTART_Ubuntu.md)
for GMP, MPFR, MPC, WIGXJPF and FASTWIGXJ. The last two must be compiled under
`ext/wigxjpf` and `ext/fastwigxj`. The upstream guide documents a FASTWIGXJ 1.4.1
compiler issue; apply its dependency workaround if needed. Do not modify the
EPRL sources or Y-map as part of that workaround.

On INFN, first check available compiler, BLAS and multiprecision modules. Use
module-provided libraries or a user-local installation; no root installation
is assumed. Do not replace system libraries. For the `BLAS=system` option,
OpenBLAS development headers and libraries are needed. Set include/library
search paths using your cluster's modules as appropriate.

From this repository, build and record the native configuration:

```bash
bash scripts/build_backend.sh /absolute/path/to/sl2cfoam-next
```

The script rebuilds **only the upstream checkout's build products**, with MPI
disabled and OpenMP enabled. It records the revision, compiler, convention and
library hashes in `build-info.json`. Use `BLAS=mkl` with a configured `MKLROOT`
if you prefer the upstream MKL build. Compile on a compute-compatible CPU:
upstream uses `-march=native`.

Prepare the symbol lookup tables using the commands from the upstream guide:

```bash
mkdir -p /absolute/path/to/sl2cfoam-next/data_sl2cfoam
cd /absolute/path/to/sl2cfoam-next/data_sl2cfoam
../ext/fastwigxj/bin/hash_js --max-E-3j=50 /dev/null table_50.3j
../ext/fastwigxj/bin/hash_js --max-E-6j=40 /dev/null table_40.6j
```

Keep exactly one `.3j` and one `.6j` table in the selected table directory.
The library uses these tables during symbol evaluation. They are not a choice
of physical-spin cutoff.

Set these variables in each login/batch environment, using actual absolute paths:

```bash
export SL2CFOAM_ROOT=/absolute/path/to/sl2cfoam-next
export SL2CFOAM_LIBRARY="$SL2CFOAM_ROOT/lib/libsl2cfoam.so"
export SL2CFOAM_TABLES="$SL2CFOAM_ROOT/data_sl2cfoam"
```

For non-system shared libraries, load the same modules in batch jobs or export
the appropriate `LD_LIBRARY_PATH` before Python starts. `configs/backend.env.example`
shows the variables. The Slurm submission inherits the submitting environment.

Then activate this project's Python environment and run:

```bash
make test  # now includes the real native DL=0 benchmark
make smoke
```

The library may abort the process on an internal error. Production tasks run in
separate processes, preserving completed checkpoints. Native booster files are
isolated per cutoff task to avoid simultaneous writers. The project writes its
own atomic vertex checkpoints and does not reuse native vertex-cache files.

For larger calculations, benchmark higher `Delta l` and native accuracy settings
separately. The output-tensor estimate excludes potentially much larger native
workspace and lookup tables. `memory_mb_per_thread` is an upstream workspace
control, not a total process-memory guarantee.
