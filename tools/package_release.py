#!/usr/bin/env python3
"""Create or verify a source ZIP from the exact allowlisted export."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import zipfile

from check_release import check_release, contains_private, export_path, scan_paths

ARCHIVE_ROOT = "surgecount"
NEUTRAL_DATE = (1980, 1, 1, 0, 0, 0)
NEUTRAL_MODE = 0o100644 << 16


def source_files(root: Path):
    for path, relative in sorted(scan_paths(root), key=lambda entry: entry[1].as_posix()):
        if path.is_file() and not path.is_symlink() and export_path(relative):
            yield path


def verify_archive(archive_path: Path, root: Path, deny_terms=()) -> list[str]:
    """Check exact membership, bytes, portable names and neutral ZIP metadata.

    This verifies against the current source tree. It does not extract files or
    accept arbitrary archive contents as a new source of trusted files.
    """
    root = root.resolve()
    issues = []
    expected = {ARCHIVE_ROOT + "/" + path.relative_to(root).as_posix(): path for path in source_files(root)}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                issues.append("archive contains duplicate member names")
            if set(names) != set(expected):
                issues.append("archive member set differs from the allowlisted source export")
            if archive.comment:
                issues.append("archive contains a nonneutral ZIP comment")
            for item in archive.infolist():
                name = PurePosixPath(item.filename)
                if name.is_absolute() or ".." in name.parts or "\\" in item.filename or "\x00" in item.filename:
                    issues.append("archive contains an unsafe member path")
                if (
                    item.date_time != NEUTRAL_DATE
                    or item.external_attr != NEUTRAL_MODE
                    or item.comment
                    or item.extra
                    or item.create_system != 3
                    or item.internal_attr != 0
                ):
                    issues.append(f"{item.filename}: nonneutral member metadata")
                data = archive.read(item)
                if contains_private(data, Path(item.filename), deny_terms):
                    issues.append(f"{item.filename}: supplied private identifier")
                if item.filename in expected and data != expected[item.filename].read_bytes():
                    issues.append(f"{item.filename}: archived bytes differ from checked source")
            if archive.testzip() is not None:
                issues.append("archive CRC integrity validation failed")
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as error:
        issues.append(f"archive could not be verified: {type(error).__name__}")
    return sorted(set(issues))


def build_archive(output: Path, root: Path, deny_terms=()) -> int:
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Choose a ZIP destination outside the source tree")
    issues = check_release(root, deny_terms)
    if issues:
        raise ValueError("Release scan failed:\n" + "\n".join(issues))
    files = list(source_files(root))
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.comment = b""
        for path in files:
            info = zipfile.ZipInfo(ARCHIVE_ROOT + "/" + path.relative_to(root).as_posix(), date_time=NEUTRAL_DATE)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = NEUTRAL_MODE
            info.internal_attr = 0
            info.extra = b""
            info.comment = b""
            archive.writestr(info, path.read_bytes())
    issues = verify_archive(output, root, deny_terms)
    if issues:
        raise ValueError("Created ZIP failed verification:\n" + "\n".join(issues))
    return len(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="ZIP path outside the source tree")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--verify", action="store_true", help="Audit an existing ZIP against this source tree; do not write"
    )
    parser.add_argument("--deny-term", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        if args.verify:
            issues = check_release(args.root, args.deny_term) + verify_archive(args.output, args.root, args.deny_term)
            if issues:
                raise ValueError("Archive verification failed:\n" + "\n".join(issues))
            print(f"Verified {args.output.name}: exact source members, bytes and neutral metadata.")
        else:
            count = build_archive(args.output, args.root, args.deny_term)
            print(
                f"Created {args.output.name}: {count} files, {args.output.stat().st_size:,} bytes; verification passed."
            )
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
