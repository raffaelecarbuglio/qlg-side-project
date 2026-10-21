"""Restore a five-cell checkpoint only when every numerical input matches."""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tarfile
import tempfile

import numpy as np

from benchmarks.block_scan import cache_spec
from qlg.jobs import fingerprint
from qlg.spins import vertex_shape


def read_member(tar, name):
    member = tar.getmember(name)
    if not member.isfile():
        raise ValueError(f"Expected a regular archive member: {name}")
    with tar.extractfile(member) as stream:
        return stream.read()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    spec = cache_spec()
    with tarfile.open(args.archive, "r:gz") as tar:
        archived = json.loads(read_member(tar, "cache-spec.json"))
        if archived != spec:
            raise ValueError("Backend, tables, wrapper or NumPy differ from the archived cache")
        index = json.loads(read_member(tar, "checkpoint-index.json"))
        if index["checkpoint_cache_id"] != fingerprint(spec):
            raise ValueError("Cache identity does not match the index")
        entries = index["files"]
        if len(entries) != index["saved_vertices"]:
            raise ValueError("Incomplete checkpoint index")
        prepared = []
        for entry in entries:
            name = entry["path"]
            path = Path(name)
            if (len(path.parts) != 2 or path.parent.name != f"dl_{index['dl']:03d}"
                    or path.suffix != ".npz" or not all(c.isdigit() or c == "-"
                                                        for c in path.stem)):
                raise ValueError(f"Invalid archive path: {name}")
            contents = read_member(tar, name)
            if len(contents) != entry["bytes"] or hashlib.sha256(contents).hexdigest() != entry["sha256"]:
                raise ValueError(f"Checkpoint digest mismatch: {name}")
            spins = tuple(map(int, path.stem.split("-")))
            expected = fingerprint(dict(cache_id=fingerprint(spec),
                                        dl=index["dl"], spins=spins))
            with np.load(BytesIO(contents), allow_pickle=False) as data:
                if (str(data["identity"]) != expected or
                        data["tensor"].shape != vertex_shape(spins) or
                        not np.isfinite(data["tensor"]).all()):
                    raise ValueError(f"Invalid tensor: {name}")
            prepared.append((path, contents, entry["sha256"]))
    for path, contents, digest in prepared:
        target = args.run / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError(f"Conflicting existing checkpoint: {target}")
            continue
        fd, temp = tempfile.mkstemp(dir=target.parent, prefix=".restore-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        finally:
            Path(temp).unlink(missing_ok=True)
    print(f"Restored {len(prepared)} verified tensors to {args.run}", flush=True)


if __name__ == "__main__":
    main()
