"""Copy immutable expert datasets to local storage, leaving checkpoints on persistent storage."""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path


def signature(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_dataset(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        raise ValueError("Local storage must differ from the source dataset.")
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "staging.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    started = time.perf_counter()
    copied = 0
    for filename in ("data.hdf5", "metadata.json"):
        src, dst = source / filename, target / filename
        before = signature(src)
        cached = manifest.get(filename, {})
        if dst.exists():
            if (
                cached.get("source") == str(src)
                and cached.get("source_stat") == before
                and cached.get("target_stat") == signature(dst)
            ):
                continue
            # Do not overwrite an existing dataset, including a different experiment's data.
            if dst.stat().st_size != before["bytes"] or sha256(dst) != sha256(src):
                raise FileExistsError(f"Existing staged file differs from source: {dst}")
            digest = sha256(dst)
        else:
            partial = dst.with_suffix(dst.suffix + ".part")
            reclaimable = partial.stat().st_size if partial.exists() else 0
            if shutil.disk_usage(target).free + reclaimable < before["bytes"]:
                raise OSError(f"Insufficient local space for {src}: need {before['bytes'] / 1024**3:.1f} GiB")
            digest = hashlib.sha256()
            last_print = time.perf_counter()
            written = 0
            print(f"Storage | copying {src} -> {dst} | {before['bytes'] / 1024**3:.2f} GiB", flush=True)
            with src.open("rb") as reader, partial.open("wb") as writer:
                for block in iter(lambda: reader.read(8 * 1024**2), b""):
                    writer.write(block)
                    digest.update(block)
                    written += len(block)
                    if time.perf_counter() - last_print >= 30:
                        print(f"Storage | {filename} | {written / before['bytes']:.0%}", flush=True)
                        last_print = time.perf_counter()
            digest = digest.hexdigest()
            if signature(src) != before or sha256(partial) != digest:
                raise RuntimeError(f"Source changed or copy verification failed: {src}")
            partial.replace(dst)
            copied += before["bytes"]
        if signature(src) != before:
            raise RuntimeError(f"Dataset changed while staging: {src}")
        manifest[filename] = {
            "source": str(src),
            "source_stat": before,
            "target_stat": signature(dst),
            "sha256": digest,
        }
        temporary = manifest_path.with_suffix(".json.part")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)
    result = {"dataset": source.name, "copied_bytes": copied, "seconds": time.perf_counter() - started}
    print(
        f"Storage | ready={target} | copied={copied / 1024**3:.2f} GiB | elapsed={result['seconds']:.1f}s", flush=True
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--target", type=Path, required=True, help="An actual local SSD mount; /tmp is not always SSD-backed."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["cartpole_balance_sparse", "reacher_easy", "ball_in_cup_catch"]
    )
    args = parser.parse_args()
    for name in args.datasets:
        if Path(name).name != name:
            parser.error("Dataset names must be directory names, not paths")
        stage_dataset(args.source / name, args.target / name)


if __name__ == "__main__":
    main()
