"""Regression checks for disclosure paths in the static review artifact."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
