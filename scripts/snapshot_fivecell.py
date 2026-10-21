"""Export a validated, restartable five-cell vertex cache and partial report.

The archive contains complete native tensors and their strict cache identity.
Snapshots can be taken while a calculation is running because each tensor is
written atomically; temporary files are never included.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile

import numpy as np

from benchmarks.block_scan import cache_spec
from qlg.jobs import atomic_json, fingerprint


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--dl", type=int, required=True)
    args = parser.parse_args()
    run = args.run
    spec = cache_spec()
    cache_id = fingerprint(spec)
    files = sorted((run / f"dl_{args.dl:03d}").glob("*.npz"))
    entries = []
    for path in files:
        spins = tuple(map(int, path.stem.split("-")))
        expected = fingerprint(dict(cache_id=cache_id, dl=args.dl, spins=spins))
        with np.load(path, allow_pickle=False) as data:
            if str(data["identity"]) != expected or not np.isfinite(data["tensor"]).all():
                raise ValueError(f"Corrupt or incompatible checkpoint: {path}")
        entries.append(dict(path=str(path.relative_to(run)), bytes=path.stat().st_size,
                            sha256=sha256(path)))
    report_bytes = args.report.read_bytes()
    report = json.loads(report_bytes)
    index = dict(status=report["status"], created_utc=datetime.now(timezone.utc).isoformat(),
                 dl=args.dl, checkpoint_cache_id=cache_id, saved_vertices=len(entries),
                 completed_amplitudes=[dict(two_k=r["two_k"], ordering=r["ordering"])
                                       for r in report["results"] if r["dl"] == args.dl],
                 files=entries,
                 note="Partial tensors and finite cutoff amplitudes only; no continuum limit.")
    prefix = args.prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(prefix.with_name(prefix.name + "-checkpoint.json"), index)
    report_file = prefix.with_name(prefix.name + "-report.json")
    report_temp = report_file.with_name(report_file.name + ".tmp")
    report_temp.write_bytes(report_bytes)
    os.replace(report_temp, report_file)
    archive = prefix.with_name(prefix.name + "-checkpoint.tar.gz")
    temporary = archive.with_name(archive.name + ".tmp")
    backend = Path(os.environ["SL2CFOAM_LIBRARY"]).resolve()
    build_info = backend.parent.parent / "build-info.json"
    if not build_info.is_file():
        raise FileNotFoundError(build_info)
    try:
        with tarfile.open(temporary, "w:gz") as tar:
            for source, target in [(build_info, "build-info.json"),
                                   (report_file, "report.json")]:
                tar.add(source, arcname=target)
            # Save the cache specification exactly as used for identity checks.
            cache_file = run / "cache-spec.json"
            atomic_json(cache_file, spec)
            tar.add(cache_file, arcname="cache-spec.json")
            for entry in entries:
                tar.add(run / entry["path"], arcname=entry["path"])
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Saved {len(entries)} verified vertices in {archive}", flush=True)


if __name__ == "__main__":
    main()
