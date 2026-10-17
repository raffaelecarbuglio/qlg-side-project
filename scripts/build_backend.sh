#!/usr/bin/env bash
# Build an existing upstream checkout after installing its native dependencies.
set -euo pipefail
if [[ $# -ne 1 ]]; then
    echo "Usage: bash scripts/build_backend.sh /absolute/path/to/sl2cfoam-next" >&2
    exit 1
fi
root=$(cd "$1" && pwd)
pin=052e4346028870bd76f69a3034e6cae8defb8f7f
actual=$(git -C "$root" rev-parse HEAD)
if [[ "$actual" != "$pin" ]]; then
    echo "Expected upstream revision $pin; found $actual. See docs/setup.md." >&2
    exit 1
fi
if [[ -n $(git -C "$root" status --porcelain --untracked-files=no) ]]; then
    echo "Upstream checkout has modified tracked files; record/review them before building." >&2
    exit 1
fi
for header in "$root/ext/wigxjpf/inc/wigxjpf.h" "$root/ext/fastwigxj/inc/fastwigxj.h"; do
    if [[ ! -f "$header" ]]; then
        echo "Missing $header. Install the upstream dependencies described in docs/setup.md." >&2
        exit 1
    fi
done
# Refuse to silently inherit incompatible conventions or an MPI build.
make -C "$root" clean
make -C "$root" lib BLAS="${BLAS:-system}" MPI=0 OMP=1 IO=1 ADD_CFLAGS= -j "${BUILD_JOBS:-2}"
ROOT_PATH="$root" BLAS_VENDOR="${BLAS:-system}" python3 - <<'PY'
import hashlib, json, os, pathlib, subprocess
root = pathlib.Path(os.environ['ROOT_PATH'])
files = [root/'lib/libsl2cfoam.so', root/'ext/wigxjpf/lib/libwigxjpf.a',
         root/'ext/fastwigxj/lib/libfastwigxj.a']
info = {
    'repository': 'https://github.com/qg-cpt-marseille/sl2cfoam-next',
    'commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
    'rho_convention': 'gamma*(j+1)', 'mpi': False, 'openmp': True, 'io': True,
    'blas': os.environ['BLAS_VENDOR'],
    'compiler': subprocess.check_output(['gcc','--version'], text=True).splitlines()[0],
    'files_sha256': {str(p.relative_to(root)): hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
                     for p in files if p.exists()}
}
(root/'build-info.json').write_text(json.dumps(info, indent=2)+'\n')
PY
echo "Built $root/lib/libsl2cfoam.so"
