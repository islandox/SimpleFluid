#!/usr/bin/env python3
"""Keep split DWARF outside ccache and pack it as a solid zstd stream in CI."""

import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    temporary.replace(path)


def launch(store, command):
    # CMake supplies: ccache compiler [compiler arguments]. Non-split and IPO
    # compilations retain ordinary ccache behavior, including on macOS.
    arguments = command[2:]
    if "-gsplit-dwarf" not in arguments:
        return subprocess.call(command)
    if "-o" not in arguments or any(arg.startswith("@") for arg in arguments):
        raise ValueError("split DWARF launcher requires an explicit -o and expanded arguments")
    if any(arg.startswith(("-dumpdir", "-dumpbase", "-save-temps")) for arg in arguments):
        raise ValueError("split DWARF launcher requires .dwo output beside the object")
    output = Path(arguments[arguments.index("-o") + 1]).absolute()
    dwo = output.with_suffix(".dwo")
    # Never mistake symbols from an earlier compilation for a cache-miss output.
    dwo.unlink(missing_ok=True)
    env = os.environ.copy()
    # Hiding -gsplit-dwarf also hides ccache's output-name hashing. Retain that
    # protection explicitly: skeleton DWARF contains the output's .dwo pathname.
    identity = hashlib.sha256((str(Path.cwd()) + "\0" + str(output)).encode()).hexdigest()
    env["CCACHE_NAMESPACE"] = env.get("CCACHE_NAMESPACE", "") + ":dwo-bundle-v1:" + identity
    wrapped = command[:2]
    for argument in arguments:
        if argument == "-gsplit-dwarf":
            wrapped.append("--ccache-skip")
        wrapped.append(argument)
    status = subprocess.call(wrapped, env=env)
    if status:
        return status
    key = digest(output)
    record = store / key
    if not dwo.exists():
        # A record is either SHA256 + newline + raw DWO, or 'none' for a
        # successful compiler invocation that emits no DWO (e.g. Clang -g0).
        if record.is_file():
            data = record.read_bytes()
            checksum, separator, symbols = data.partition(b"\n")
            if data == b"none\n":
                record.touch()
                return 0
            if separator and hashlib.sha256(symbols).hexdigest().encode() == checksum:
                atomic_write(dwo, symbols)
                record.touch()
                return 0
        # An object-cache hit without matching symbols is incomplete. Recompile
        # through ccache so dependency files and diagnostics follow its contract.
        print("Split DWARF bundle miss; regenerating " + str(dwo), file=sys.stderr)
        env["CCACHE_RECACHE"] = "1"
        status = subprocess.call(wrapped, env=env)
        if status:
            return status
        key = digest(output)
        record = store / key
    if dwo.exists():
        symbols = dwo.read_bytes()
        atomic_write(record, hashlib.sha256(symbols).hexdigest().encode() + b"\n" + symbols)
    else:
        atomic_write(record, b"none\n")
    return 0


def pack(store, bundle):
    store.mkdir(parents=True, exist_ok=True)
    bundle.parent.mkdir(parents=True, exist_ok=True)
    temporary = bundle.with_suffix(bundle.suffix + ".tmp")
    try:
        with temporary.open("wb") as output:
            process = subprocess.Popen(
                ["zstd", "-T2", "-9", "--long=27", "-c"], stdin=subprocess.PIPE, stdout=output
            )
            try:
                with tarfile.open(fileobj=process.stdin, mode="w|") as archive:
                    for record in sorted(store.iterdir()):
                        if not re.fullmatch(r"[0-9a-f]{64}", record.name) or not record.is_file():
                            continue
                        if record.stat().st_mtime < time.time() - 7 * 86400:
                            record.unlink()
                            continue
                        archive.add(record, arcname=record.name, recursive=False)
            finally:
                process.stdin.close()
                status = process.wait()
            if status:
                raise RuntimeError("zstd compression failed")
        temporary.replace(bundle)
    finally:
        temporary.unlink(missing_ok=True)


def unpack(store, bundle):
    # A missing or damaged companion bundle must cause regeneration, never use
    # leftovers from a different restored archive. Only flat regular records
    # are accepted; never extract arbitrary paths or links from a cached tar.
    if store.exists():
        shutil.rmtree(store)
    store.parent.mkdir(parents=True, exist_ok=True)
    if not bundle.exists():
        store.mkdir()
        return
    with tempfile.TemporaryDirectory(dir=store.parent, prefix="dwo-unpack-") as directory:
        staging = Path(directory) / "records"
        staging.mkdir()
        process = subprocess.Popen(["zstd", "-d", "--long=27", "-c", str(bundle)], stdout=subprocess.PIPE)
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    if not member.isfile() or not re.fullmatch(r"[0-9a-f]{64}", member.name):
                        raise ValueError("invalid split DWARF bundle member: " + member.name)
                    with archive.extractfile(member) as source, (staging / member.name).open("wb") as target:
                        shutil.copyfileobj(source, target)
                    os.utime(staging / member.name, (member.mtime, member.mtime))
        finally:
            process.stdout.close()
            status = process.wait()
        if status:
            raise RuntimeError("zstd decompression failed")
        staging.replace(store)


def main():
    store = Path(os.environ["SIMPLEFLUID_DWO_CACHE_DIR"]).absolute()
    if sys.argv[1] == "pack":
        pack(store, Path(sys.argv[2]))
    elif sys.argv[1] == "unpack":
        unpack(store, Path(sys.argv[2]))
    else:
        return launch(store, sys.argv[1:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
