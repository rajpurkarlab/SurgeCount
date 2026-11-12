#!/usr/bin/env python3
"""Audit a source release without network, git, hardware, or ML imports.

Pass private identifiers with --deny-term; do not store an identity deny list in
this repository. Automatic checks do not replace visual review of media.
"""

from __future__ import annotations

import argparse
import ipaddress
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

SKIP_DIRS = {".git", ".venv", ".cache", ".pytest_cache", ".ruff_cache", "__pycache__", "build", "dist"}
FORBIDDEN_DIRS = {"outputs", "datasets", "models", "checkpoints", "calibration", "training_state"}
TEXT_SUFFIXES = {
    ".md",
    ".mdx",
    ".rst",
    ".lock",
    ".svg",
    ".proto",
    ".in",
    ".py",
    ".json",
    ".toml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".txt",
    ".yaml",
    ".yml",
    ".cff",
    ".sh",
    ".cfg",
    ".ini",
    ".xml",
}
FORBIDDEN_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".parquet", ".pem", ".key", ".jsonl"}
ROOT_FILES = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY.md",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "MANIFEST.in",
    "pyproject.toml",
    "uv.lock",
    ".gitignore",
    ".env.example",
}
SOURCE_DIRS = {"src", "configs", "docs", "tests", "tools", "vendor", "website", ".github"}
REQUIRED = {
    "README.md",
    "LICENSE",
    "NOTICE",
    "THIRD_PARTY.md",
    "CITATION.cff",
    "pyproject.toml",
    "uv.lock",
    "configs/train.json",
    "configs/record.json",
    "configs/deploy.json",
    "configs/robot.example.json",
    "vendor/lerobot/LICENSE",
    "vendor/lerobot/SURGECOUNT_CHANGES.md",
}
# Public dependency providers, workflow actions, and attribution repositories.
# Project publication links are reviewed separately below.
UPSTREAM_REPOSITORIES = {
    ("huggingface", "lerobot"),
    ("jgthms", "bulma"),
    ("nerfies", "nerfies.github.io"),
    ("physical-intelligence", "openpi"),
    ("actions", "checkout"),
    ("actions", "configure-pages"),
    ("actions", "upload-pages-artifact"),
    ("actions", "deploy-pages"),
    ("astral-sh", "setup-uv"),
}
PUBLIC_PROJECT_REPOSITORIES = {("rajpurkarlab", "surgecount")}
PUBLIC_PROJECT_URLS = {
    ("github.com", "/rajpurkarlab/SurgeCount"),
    ("rajpurkarlab.github.io", "/SurgeCount"),
}
HF_NAMESPACES = {"local", "lerobot", "google"}
LOCAL_NAMESPACES = SOURCE_DIRS | {
    "data",
    "meta",
    "outputs",
    "models",
    "checkpoints",
    "path",
    "absolute",
    "dev",
    "images",
    "static",
    "videos",
    "action",
    "observation",
    "stats",
    "common",
    "policies",
    "video",
    "audio",
    "image",
    "text",
    "application",
}
PUBLIC_HOSTS = {
    "apache.org",
    "www.apache.org",
    "creativecommons.org",
    "opensource.org",
    "pypi.org",
    "files.pythonhosted.org",
    "docs.python.org",
    "docs.astral.sh",
    "docs.pytest.org",
    "docs.github.com",
    "download.pytorch.org",
    "pytorch.org",
    "numpy.org",
    "pandas.pydata.org",
    "arrow.apache.org",
    "docs.scipy.org",
    "docs.opencv.org",
    "ai.google.dev",
    "cloud.google.com",
    "bulma.io",
    "www.w3.org",
}
GENERIC_USERS = {"your_username", "user", "username", "path", "example"}
SECRETS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:hf_[A-Za-z0-9]{25,}|AIza[A-Za-z0-9_-]{30,}|sk-[A-Za-z0-9_-]{25,})\b"),
]
PERSONAL_PATH = re.compile(r"/(?:Users|home)/([^/\s\"'<>]+)/", re.I)
WINDOWS_PATH = re.compile(r"[A-Z]:[\\/]+Users[\\/]+([^\\/\s\"'<>]+)[\\/]", re.I)
SERIAL = re.compile(r"/dev/tty\.usbmodem[A-Za-z0-9]{6,}")
URL = re.compile(r"https?://[^\s<>\"'`\\]+", re.I)
QUOTED_REPO = re.compile(r"[\"'`]([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)[\"'`]")
HF_CONTEXT_REPO = re.compile(
    r"(?:\bhf\s+download\s+(?:--repo-type\s+\w+\s+)?|(?:dataset_repo_id|repo_id|base_model|model_repo_id)\s*(?::[^=\n]+)?\s*[=:]\s*[\"']?)([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)"
)
BARE_REPO = re.compile(r"(?m)^\s*(?:[-*]\s+)?([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\s*$")
EMAIL = re.compile(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
COMMIT = re.compile(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{40}(?![A-Fa-f0-9])")
HISTORICAL_SLUG = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*_\d{2,}_(?:clean|mixed|singlecam)(?:_[A-Za-z0-9]+)*\b", re.I)


def os_metadata(relative: Path) -> bool:
    return any(
        part in {".DS_Store", "__MACOSX", "Thumbs.db", "desktop.ini"} or part.startswith("._")
        for part in relative.parts
    )


def excluded(relative: Path) -> bool:
    return any(
        part in SKIP_DIRS or part.endswith(".egg-info") for part in relative.parts
    ) or relative.suffix.lower() in {".pyc", ".pyo"}


def export_path(relative: Path) -> bool:
    return (
        not excluded(relative)
        and not os_metadata(relative)
        and (relative.parts[0] in SOURCE_DIRS or relative.as_posix() in ROOT_FILES)
    )


def scan_paths(root: Path):
    """Prune excluded directories without ever traversing a symlink."""
    pending = [root]
    while pending:
        directory = pending.pop()
        for path in sorted(directory.iterdir()):
            relative = path.relative_to(root)
            if excluded(relative):
                continue
            yield path, relative
            if path.is_dir() and not path.is_symlink():
                pending.append(path)


def decoded_text(data: bytes):
    # Byte searches cover all formats. Decoding also enables Unicode casefold
    # matches, UTF-16 metadata and plain-text extensions without a suffix list.
    yield data.decode("utf-8", errors="ignore")
    if b"\x00" in data:
        for encoding in ("utf-16-le", "utf-16-be"):
            yield data.decode(encoding, errors="ignore")


def text_container(data: bytes) -> bool:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return True
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def contains_private(data: bytes, relative: Path, terms) -> bool:
    texts = None
    for term in terms:
        if not term:
            continue
        if term.casefold() in relative.as_posix().casefold():
            return True
        for encoding in ("utf-8", "utf-16-le", "utf-16-be"):
            if term.encode(encoding).lower() in data.lower():
                return True
        if texts is None:
            texts = list(decoded_text(data))
        if any(term.casefold() in text.casefold() for text in texts):
            return True
    return False


def approved_ip(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return True  # E.g. dotted package versions, not an address.
    if address.is_loopback or address.is_unspecified:
        return True
    return any(
        address in block
        for block in (
            ipaddress.ip_network("192.0.2.0/24"),
            ipaddress.ip_network("198.51.100.0/24"),
            ipaddress.ip_network("203.0.113.0/24"),
            ipaddress.ip_network("2001:db8::/32"),
        )
        if address.version == block.version
    )


def approved_url(value: str) -> bool:
    try:
        parsed = urlsplit(value.rstrip(".,);]}"))
        host = (parsed.hostname or "").lower()
        if parsed.username or parsed.password:
            return False
        if (
            parsed.scheme == "https"
            and parsed.port is None
            and not parsed.query
            and not parsed.fragment
            and (host, parsed.path.rstrip("/")) in PUBLIC_PROJECT_URLS
        ):
            return True
        if host in {"localhost", "127.0.0.1", "::1", "example.com", "example.org", "example.net"} or host.endswith(
            ".invalid"
        ):
            return True
        try:
            ipaddress.ip_address(host)
            return approved_ip(host)
        except ValueError:
            pass
        parts = [part for part in unquote(parsed.path).split("/") if part]
        if host in {"huggingface.co", "hf.co", "www.huggingface.co"}:
            if not parts or parts[0] in {"docs", "terms-of-service", "privacy"}:
                return True
            if parts[0] in {"datasets", "spaces"}:
                parts = parts[1:]
            return bool(parts) and parts[0].lower() in HF_NAMESPACES - {"local"}
        if host in {"github.com", "www.github.com", "raw.githubusercontent.com", "codeload.github.com"}:
            return (
                len(parts) >= 2 and (parts[0].lower(), parts[1].removesuffix(".git").lower()) in UPSTREAM_REPOSITORIES
            )
        return host in PUBLIC_HOSTS
    except ValueError:
        return False


def automatic_issues(text: str, relative: Path) -> list[str]:
    reasons = []
    if any(pattern.search(text) for pattern in SECRETS):
        reasons.append("possible credential/private key")
    paths = list(PERSONAL_PATH.finditer(text)) + list(WINDOWS_PATH.finditer(text))
    if any(match.group(1).casefold() not in GENERIC_USERS for match in paths):
        reasons.append("personal machine path")
    # Vendor source carries essential upstream authors, account URLs and
    # revision links. Supplied private terms and credential/path checks apply
    # there too; first-party additions receive the stricter checks below.
    if HISTORICAL_SLUG.search(text) or HISTORICAL_SLUG.search(relative.as_posix()):
        reasons.append("historical dataset/checkpoint identifier")
    upstream = relative.parts[0] == "vendor" and relative.name != "SURGECOUNT_CHANGES.md"
    if upstream:
        return reasons
    if SERIAL.search(text):
        reasons.append("device-specific serial port")
    if EMAIL.search(text) and relative.name not in {"LICENSE", "NOTICE", "THIRD_PARTY.md"}:
        reasons.append("author/contact email")
    urls = list(URL.finditer(text))
    if any(not approved_url(match.group()) for match in urls):
        reasons.append("unapproved external account or URL")
    without_urls = URL.sub("", text)
    for match in (
        list(QUOTED_REPO.finditer(without_urls))
        + list(HF_CONTEXT_REPO.finditer(without_urls))
        + list(BARE_REPO.finditer(without_urls))
    ):
        owner, repo = match.group(1).lower(), match.group(2)
        if (
            owner not in HF_NAMESPACES | LOCAL_NAMESPACES
            and not re.fullmatch(r"[0-9.]+", owner)
            and (owner, repo.lower()) not in UPSTREAM_REPOSITORIES | PUBLIC_PROJECT_REPOSITORIES
        ):
            reasons.append("unapproved dataset/model or account identifier")
            break
    network_text = re.sub(r"\b(?:version|specifier)\s*=\s*[\"'][^\"']*[\"']", "", without_urls)
    if any(not approved_ip(match.group()) for match in IPV4.finditer(network_text)):
        reasons.append("nonexample IP address")
    # Lockfiles contain integrity hashes. Revisions inside reviewed upstream
    # URLs are allowed; a bare checkout fingerprint in first-party prose is not.
    if relative.suffix.lower() != ".lock" and COMMIT.search(without_urls):
        reasons.append("bare repository revision fingerprint")
    return reasons


def check_release(root: Path, deny_terms=(), tracked_paths=()) -> list[str]:
    root = root.resolve()
    issues = []
    for name in sorted(REQUIRED):
        if not (root / name).is_file():
            issues.append(f"missing release file: {name}")
    for value in tracked_paths:
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts or excluded(relative) or os_metadata(relative):
            issues.append(f"{value}: unsafe tracked artifact (excluded from source export)")
    for path, relative in scan_paths(root):
        if os_metadata(relative):
            issues.append(f"{relative}: operating-system metadata artifact")
        if path.is_symlink():
            issues.append(f"{relative}: symlink outside the self-contained source artifact")
            continue
        if not path.is_file():
            continue
        if (
            relative.parts[0] in FORBIDDEN_DIRS
            or "training_state" in relative.parts
            or path.suffix.lower() in FORBIDDEN_SUFFIXES
        ):
            issues.append(f"{relative}: runtime data or private artifact")
        if path.name.startswith(".env") and path.name != ".env.example":
            issues.append(f"{relative}: environment file")
        if path.stat().st_size > 25 * 1024 * 1024:
            issues.append(f"{relative}: unexpectedly large file")
        data = path.read_bytes()
        if contains_private(data, relative, deny_terms):
            issues.append(f"{relative}: supplied private identifier")
        # Compressed media produces accidental email/repository-shaped bytes.
        # Scan credentials in every container, but semantic rules only in text.
        # Supplied identifiers above always scan raw bytes and UTF-16 variants.
        reasons = set()
        texts = list(decoded_text(data))
        if text_container(data) or path.suffix.lower() in TEXT_SUFFIXES:
            for text in texts:
                reasons.update(automatic_issues(text, relative))
        elif any(pattern.search(text) for pattern in SECRETS for text in texts):
            reasons.add("possible credential/private key")
        issues.extend(f"{relative}: {reason}" for reason in sorted(reasons))
    return sorted(set(issues))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--deny-term", action="append", default=[])
    parser.add_argument("--tracked-path-list", type=Path, help="Optional external newline-separated git tracked paths")
    args = parser.parse_args(argv)
    tracked = args.tracked_path_list.read_text().splitlines() if args.tracked_path_list else ()
    issues = check_release(args.root, args.deny_term, tracked)
    if issues:
        print("Release check failed:\n" + "\n".join(f"- {item}" for item in issues))
        return 1
    print("Source release check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
