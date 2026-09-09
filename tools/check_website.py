#!/usr/bin/env python3
"""Validate the anonymous static website without making network requests."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
from urllib.parse import unquote, urlsplit


TEXT_SUFFIXES = {".html", ".css", ".js", ".json", ".svg", ".txt", ".md"}
DOCUMENT_SUFFIXES = {".pdf", ".docx", ".tex", ".zip", ".bib"}
NETWORK_URL = re.compile(r"(?:https?:)?//[^\s\"'<>]+", re.IGNORECASE)
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
IP_ADDRESS = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
OLD_VENUE = re.compile(r"\bCoRL\b|Conference on Robot Learning", re.IGNORECASE)


class Page(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.references: list[str] = []
        self.meta: dict[str, str] = {}
        self.issues: list[str] = []

    def handle_starttag(self, tag: str, attributes: list[tuple[str, str | None]]) -> None:
        attrs = dict(attributes)
        if attrs.get("id"):
            if attrs["id"] in self.ids:
                self.issues.append(f"duplicate id: {attrs['id']}")
            self.ids.add(attrs["id"])
        for key in ("src", "href", "poster", "action", "data"):
            if attrs.get(key) is not None:
                self.references.append(attrs[key])
        if attrs.get("srcset"):
            self.references.extend(item.strip().split()[0] for item in attrs["srcset"].split(","))
        if tag == "meta":
            name = attrs.get("name") or attrs.get("property") or attrs.get("http-equiv")
            if name:
                self.meta[name.lower()] = attrs.get("content") or ""
                if name in {"og:image", "twitter:image", "citation_pdf_url"} and attrs.get("content"):
                    self.references.append(attrs["content"])
        if tag == "video" and "controls" not in attrs:
            self.issues.append("video is missing native playback controls")
        if tag == "img" and not attrs.get("alt"):
            self.issues.append("image is missing alt text")
        if any(key.lower().startswith("on") for key in attrs):
            self.issues.append("inline event handler bypasses the self-only script policy")

    handle_startendtag = handle_starttag


def check_image_metadata(path: Path) -> list[str]:
    """Inspect metadata containers without external image libraries."""
    data = path.read_bytes()
    issues = []
    if path.suffix.lower() == ".png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ["invalid PNG signature"]
        offset = 8
        while offset + 12 <= len(data):
            length = struct.unpack(">I", data[offset : offset + 4])[0]
            kind = data[offset + 4 : offset + 8]
            if kind in {b"eXIf", b"tEXt", b"iTXt", b"zTXt", b"tIME"}:
                issues.append(f"PNG metadata chunk {kind.decode()}")
            offset += 12 + length
            if kind == b"IEND":
                break
    elif path.suffix.lower() in {".jpg", ".jpeg"}:
        if not data.startswith(b"\xff\xd8"):
            return ["invalid JPEG signature"]
        offset = 2
        while offset + 4 <= len(data) and data[offset] == 0xFF:
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            marker = data[offset]
            offset += 1
            if marker in {0xDA, 0xD9}:
                break
            if marker in {0xE1, 0xED, 0xFE}:
                issues.append(f"JPEG descriptive metadata marker 0x{marker:02x}")
            length = struct.unpack(">H", data[offset : offset + 2])[0]
            if length < 2:
                return issues + ["invalid JPEG marker length"]
            offset += length
    return issues


def check_video_metadata(path: Path, ffprobe: str) -> list[str]:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_format", "-show_streams", "-show_chapters", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        return ["ffprobe could not read the video"]
    try:
        probe = json.loads(result.stdout)
    except ValueError:
        return ["ffprobe returned invalid JSON"]
    issues = []
    streams = probe.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "video":
        issues.append("must contain exactly one video stream, with no audio or data/timecode tracks")
    if probe.get("chapters"):
        issues.append("contains chapter metadata")
    if set(probe.get("format", {}).get("tags", {})) - {"major_brand", "minor_version", "compatible_brands"}:
        issues.append("contains descriptive container metadata")
    for stream in streams:
        tags = stream.get("tags", {})
        if set(tags) - {"language", "handler_name"}:
            issues.append("contains descriptive stream metadata")
        if tags.get("language", "und") != "und" or tags.get("handler_name", "VideoHandler") != "VideoHandler":
            issues.append("contains a non-generic stream label")
    return issues


def check_site(root: Path, *, media: bool = False, deny_terms: tuple[str, ...] = ()) -> list[str]:
    root = root.resolve()
    issues: list[str] = []
    if not (root / "index.html").is_file():
        return ["index.html is missing"]
    pages: dict[Path, Page] = {}
    texts: dict[Path, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            issues.append(f"{relative}: symlinks are not allowed in the public artifact")
            continue
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in relative.parts):
            issues.append(f"{relative}: hidden file in public artifact")
        if path.suffix.lower() in DOCUMENT_SUFFIXES:
            issues.append(f"{relative}: unreviewed downloadable document/archive")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        source = path.read_text(encoding="utf-8")
        texts[path] = source
        scan = source.replace("http://www.w3.org/2000/svg", "")
        if NETWORK_URL.search(scan):
            issues.append(f"{relative}: external URL in source or comments")
        if EMAIL.search(scan) or "mailto:" in scan.lower():
            issues.append(f"{relative}: contact address")
        if IP_ADDRESS.search(scan):
            issues.append(f"{relative}: server address")
        if OLD_VENUE.search(scan):
            issues.append(f"{relative}: old venue label")
        for term in deny_terms:
            if term.casefold() in scan.casefold() or term.casefold() in str(relative).casefold():
                issues.append(f"{relative}: contains a supplied private identifier")
        if path.suffix.lower() == ".html":
            page = Page()
            page.feed(source)
            pages[path] = page
            issues.extend(f"{relative}: {issue}" for issue in page.issues)

    def check_reference(source_path: Path, reference: str) -> None:
        relative = source_path.relative_to(root)
        parsed = urlsplit(reference)
        if parsed.scheme or parsed.netloc:
            issues.append(f"{relative}: nonlocal asset/link target")
            return
        if not parsed.path and not parsed.fragment:
            issues.append(f"{relative}: empty placeholder link")
            return
        target = (
            (root / unquote(parsed.path).lstrip("/"))
            if parsed.path.startswith("/")
            else (source_path.parent / unquote(parsed.path))
        )
        target = target.resolve() if parsed.path else source_path
        if not target.is_relative_to(root):
            issues.append(f"{relative}: asset/link escapes website directory")
        elif not target.is_file():
            issues.append(f"{relative}: missing local target {parsed.path}")
        elif parsed.fragment and target in pages and unquote(parsed.fragment) not in pages[target].ids:
            issues.append(f"{relative}: missing fragment {parsed.fragment}")

    for path, page in pages.items():
        for reference in page.references:
            check_reference(path, reference)
    for path, source in texts.items():
        if path.suffix == ".css":
            for reference in re.findall(r"url\(\s*['\"]?([^)'\"]+)", source):
                if not reference.startswith("data:"):
                    check_reference(path, reference.strip())
    home = pages[root / "index.html"]
    for key in ("author", "citation_author"):
        if "anonymous" not in home.meta.get(key, "").lower():
            issues.append(f"index.html: {key} metadata is not anonymous")
    if "ML4H" not in home.meta.get("citation_conference_title", ""):
        issues.append("index.html: venue metadata is not ML4H")
    if "Anonymous submission · ML4H 2026" not in texts[root / "index.html"]:
        issues.append("index.html: anonymous ML4H 2026 submission label is missing")
    if "noindex" not in home.meta.get("robots", ""):
        issues.append("index.html: noindex metadata is missing")
    if home.meta.get("referrer") != "no-referrer":
        issues.append("index.html: no-referrer metadata is missing")
    if "connect-src 'none'" not in home.meta.get("content-security-policy", ""):
        issues.append("index.html: network connection restriction is missing")
    if media:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            issues.append("--media requires ffprobe from an FFmpeg installation")
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(root)
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                issues.extend(f"{relative}: {issue}" for issue in check_image_metadata(path))
            elif path.suffix.lower() == ".mp4" and ffprobe:
                issues.extend(f"{relative}: {issue}" for issue in check_video_metadata(path, ffprobe))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "website")
    parser.add_argument("--media", action="store_true", help="inspect image metadata and probe all MP4s")
    parser.add_argument("--deny-term", action="append", default=[], help="additional private identifier to reject")
    args = parser.parse_args()
    issues = check_site(args.root, media=args.media, deny_terms=tuple(args.deny_term))
    if issues:
        for issue in issues:
            print(f"FAIL: {issue}")
        return 1
    print("PASS: anonymous website source and local targets" + (", image/video metadata" if args.media else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
