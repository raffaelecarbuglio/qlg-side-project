#!/usr/bin/env bash
# Source inside a Levante batch job after changing into the repository.
set -euo pipefail
source "${QLG_ENV_FILE:?Set QLG_ENV_FILE to the absolute private environment file}"
: "${QLG_VENV:?Set QLG_VENV}"
: "${QLG_BACKEND_ROOT:?Set QLG_BACKEND_ROOT}"
: "${SCRATCH_LOCAL:?Levante local scratch is required}"
source "$QLG_VENV/bin/activate"

# Keep native Wigner lookups and the library on the job's local NVMe. The
# cache and JSON reports stay at QLG_RUN on durable shared storage.
qlg_native_dir="$SCRATCH_LOCAL/qlg-native"
mkdir -p "$qlg_native_dir/lib" "$qlg_native_dir/data_sl2cfoam"
cp "$QLG_BACKEND_ROOT/lib/libsl2cfoam.so" "$qlg_native_dir/lib/"
cp "$QLG_BACKEND_ROOT/build-info.json" "$qlg_native_dir/"
cp "$QLG_BACKEND_ROOT"/data_sl2cfoam/*.3j "$qlg_native_dir/data_sl2cfoam/"
cp "$QLG_BACKEND_ROOT"/data_sl2cfoam/*.6j "$qlg_native_dir/data_sl2cfoam/"
export SL2CFOAM_LIBRARY="$qlg_native_dir/lib/libsl2cfoam.so"
export SL2CFOAM_TABLES="$qlg_native_dir/data_sl2cfoam"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE
export OMP_PLACES=cores OMP_PROC_BIND=close
