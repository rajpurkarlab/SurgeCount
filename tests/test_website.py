"""Regression checks for disclosure paths in the static review artifact."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zlib


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("check_website", ROOT / "tools" / "check_website.py")
checker = module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class WebsiteChecks(unittest.TestCase):
    def test_actual_website(self):
        self.assertEqual(checker.check_site(ROOT / "website"), [])

    def fixture(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        shutil.copyfile(ROOT / "website" / "index.html", root / "index.html")
        return root

    def test_identifier_in_comment_is_rejected(self):
        root = self.fixture()
        with (root / "index.html").open("a") as stream:
            stream.write("<!-- private-lab-example -->")
        self.assertTrue(
            any("private identifier" in item for item in checker.check_site(root, deny_terms=("private-lab-example",)))
        )

    def test_remote_source_is_rejected(self):
        root = self.fixture()
        with (root / "index.html").open("a") as stream:
            stream.write('<script src="https://example.invalid/tracker.js"></script>')
        issues = checker.check_site(root)
        self.assertTrue(any("external URL" in item for item in issues))
        self.assertTrue(any("nonlocal" in item for item in issues))

    def test_missing_asset_is_rejected(self):
        self.assertTrue(any("missing local target" in item for item in checker.check_site(self.fixture())))

    def test_download_and_hidden_file_are_rejected(self):
        root = self.fixture()
        (root / "draft.pdf").write_bytes(b"example")
        (root / ".env").write_text("example")
        issues = checker.check_site(root)
        self.assertTrue(any("downloadable" in item for item in issues))
        self.assertTrue(any("hidden file" in item for item in issues))

    def test_png_text_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            payload = b"Author\0private-lab-example"
            chunk = b"tEXt" + payload
            path.write_bytes(
                b"\x89PNG\r\n\x1a\n" + struct.pack(">I", len(payload)) + chunk + struct.pack(">I", zlib.crc32(chunk))
            )
            self.assertIn("PNG metadata chunk tEXt", checker.check_image_metadata(path))


class VideoMetadataChecks(unittest.TestCase):
    def probe(self, stream_tags=None, container_tags=None, extra_streams=()):
        stream = {"codec_type": "video", "tags": {"language": "und", "handler_name": "VideoHandler"}}
        stream["tags"].update(stream_tags or {})
        tags = {"major_brand": "isom", "minor_version": "512", "compatible_brands": "isomiso2avc1mp41"}
        tags.update(container_tags or {})
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({"streams": [stream, *extra_streams], "chapters": [], "format": {"tags": tags}}),
        )
        with patch.object(checker.subprocess, "run", return_value=result):
            return checker.check_video_metadata(Path("clip.mp4"), "ffprobe")

    def test_newer_ffprobe_without_vendor_tag_is_accepted(self):
        self.assertEqual(self.probe(), [])

    def test_ffmpeg_six_zero_vendor_tag_is_accepted(self):
        self.assertEqual(self.probe(stream_tags={"vendor_id": "[0][0][0][0]"}), [])

    def test_nonzero_or_empty_vendor_tag_is_rejected(self):
        for value in ("appl", "[0][0][0][1]", ""):
            with self.subTest(value=value):
                issues = self.probe(stream_tags={"vendor_id": value})
                self.assertIn("contains descriptive stream metadata: vendor_id", issues)

    def test_neutral_vendor_does_not_allow_descriptive_stream_tags(self):
        issues = self.probe(
            stream_tags={
                "vendor_id": "[0][0][0][0]",
                "encoder": "private encoder value",
                "creation_time": "private time value",
            }
        )
        self.assertIn("contains descriptive stream metadata: creation_time, encoder", issues)
        self.assertNotIn("private encoder value", " ".join(issues))
        self.assertNotIn("private time value", " ".join(issues))

    def test_container_diagnostics_name_keys_without_values(self):
        issues = self.probe(container_tags={"author": "private author value"})
        self.assertIn("contains descriptive container metadata: author", issues)
        self.assertNotIn("private author value", " ".join(issues))

    def test_audio_and_data_streams_remain_rejected(self):
        for kind in ("audio", "data"):
            with self.subTest(kind=kind):
                issues = self.probe(extra_streams=[{"codec_type": kind}])
                self.assertTrue(any("exactly one video stream" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
