"""Regression checks for public and anonymous static website validation."""

from importlib.util import module_from_spec, spec_from_file_location
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
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
RELEASE_SPEC = spec_from_file_location("website_release_check", ROOT / "tools" / "check_release.py")
release_checker = module_from_spec(RELEASE_SPEC)
RELEASE_SPEC.loader.exec_module(release_checker)


class WebsiteChecks(unittest.TestCase):
    def test_actual_website(self):
        self.assertEqual(checker.check_site(ROOT / "website", public=True), [])

    def fixture(self, *, public=True):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        authors = checker.PUBLIC_AUTHORS if public else ("Anonymous Authors",)
        citations = "".join(f'<meta name="citation_author" content="{author}">' for author in authors)
        label = "ML4H 2026" if public else "Anonymous submission · ML4H 2026"
        robots = "index, follow" if public else "noindex, nofollow"
        (root / "index.html").write_text(
            "<!doctype html><html><head>"
            f'<meta name="author" content="{"; ".join(authors)}">{citations}'
            '<meta name="citation_conference_title" content="ML4H 2026">'
            f'<meta name="robots" content="{robots}">'
            '<meta name="referrer" content="no-referrer">'
            '<meta http-equiv="Content-Security-Policy" content="connect-src \'none\'">'
            f"</head><body><p>{label}</p></body></html>"
        )
        return root

    def test_public_mode_accepts_repeated_author_tags_and_indexing(self):
        self.assertEqual(checker.check_site(self.fixture(), public=True), [])

    def test_anonymous_mode_still_accepts_prepared_copy(self):
        self.assertEqual(checker.check_site(self.fixture(public=False)), [])

    def test_public_page_is_not_an_anonymous_artifact(self):
        issues = checker.check_site(self.fixture())
        self.assertTrue(any("metadata is not anonymous" in issue for issue in issues))
        self.assertTrue(any("noindex" in issue for issue in issues))

    def test_public_mode_requires_every_named_citation_author(self):
        root = self.fixture()
        page = root / "index.html"
        page.write_text(
            page.read_text().replace(f'<meta name="citation_author" content="{checker.PUBLIC_AUTHORS[0]}">', "")
        )
        self.assertTrue(
            any(
                "citation_author metadata must include all four" in issue
                for issue in checker.check_site(root, public=True)
            )
        )

    def test_anonymous_mode_checks_every_repeated_author_tag(self):
        root = self.fixture(public=False)
        page = root / "index.html"
        page.write_text(
            page.read_text().replace(
                "</head>", f'<meta name="citation_author" content="{checker.PUBLIC_AUTHORS[0]}"></head>'
            )
        )
        self.assertTrue(any("citation_author metadata is not anonymous" in issue for issue in checker.check_site(root)))

    def test_public_label_must_be_visible_and_current(self):
        root = self.fixture()
        page = root / "index.html"
        original = page.read_text()
        page.write_text(original.replace("<p>ML4H 2026</p>", "<p>Project website</p>"))
        self.assertTrue(
            any("public ML4H 2026 label is missing" in issue for issue in checker.check_site(root, public=True))
        )
        page.write_text(original.replace("<p>ML4H 2026</p>", "<p>Anonymous submission · ML4H 2026</p>"))
        self.assertTrue(any("retains an anonymous" in issue for issue in checker.check_site(root, public=True)))

    def test_public_cli_mode_reports_public_check(self):
        with redirect_stdout(io.StringIO()) as output:
            result = checker.main(["--root", str(self.fixture()), "--public"])
        self.assertEqual(result, 0)
        self.assertIn("PASS: public website", output.getvalue())

    def test_identifier_in_comment_is_rejected(self):
        root = self.fixture()
        with (root / "index.html").open("a") as stream:
            stream.write("<!-- private-lab-example -->")
        self.assertTrue(
            any(
                "private identifier" in item
                for item in checker.check_site(root, public=True, deny_terms=("private-lab-example",))
            )
        )

    def test_remote_source_is_rejected(self):
        root = self.fixture()
        with (root / "index.html").open("a") as stream:
            stream.write('<script src="https://example.invalid/tracker.js"></script>')
        issues = checker.check_site(root, public=True)
        self.assertTrue(any("external URL" in item for item in issues))
        self.assertTrue(any("nonlocal" in item for item in issues))

    def test_missing_asset_is_rejected(self):
        root = self.fixture()
        with (root / "index.html").open("a") as stream:
            stream.write('<img src="missing.png" alt="Missing image">')
        self.assertTrue(any("missing local target" in item for item in checker.check_site(root, public=True)))

    def test_public_mode_keeps_path_contact_and_inline_script_guards(self):
        root = self.fixture()
        contact = "private-person" + "@" + "example.invalid"
        with (root / "index.html").open("a") as stream:
            stream.write(f'<img src="../outside.png" alt="Outside image"><p onclick="alert(1)">{contact}</p>')
        issues = checker.check_site(root, public=True)
        self.assertTrue(any("escapes website directory" in item for item in issues))
        self.assertTrue(any("contact address" in item for item in issues))
        self.assertTrue(any("inline event handler" in item for item in issues))

    def test_download_and_hidden_file_are_rejected(self):
        root = self.fixture()
        (root / "draft.pdf").write_bytes(b"example")
        (root / ".env").write_text("example")
        issues = checker.check_site(root, public=True)
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


class ReviewedPublicationLinks(unittest.TestCase):
    def test_only_the_reviewed_project_url_is_allowed(self):
        self.assertTrue(release_checker.approved_url("https://github.com/rajpurkarlab/SurgeCount"))
        owner = "rajpurkarlab"
        for url in (
            "https://" + "github.com/" + owner + "/other-project",
            "https://" + "github.com/" + owner + "/SurgeCount/blob/main/private.txt",
            "https://" + owner + ".github.io/SurgeCount/",
            "https://" + owner + ".github.io/other-project/",
            "https://" + owner + ".github.io/SurgeCount/private/",
        ):
            with self.subTest(url=url):
                self.assertFalse(release_checker.approved_url(url))

    def test_unused_pages_actions_are_not_allowlisted(self):
        self.assertTrue(release_checker.approved_url("https://github.com/actions/checkout"))
        for action in ("configure-pages", "upload-pages-artifact", "deploy-pages", "unreviewed-action"):
            with self.subTest(action=action):
                self.assertFalse(release_checker.approved_url("https://" + "github.com/" + "actions/" + action))

    def test_all_eight_reviewed_video_uploads_are_allowed(self):
        assets = (
            "b9a7de95-ba37-46f3-bf13-eddd0e51d600",
            "4c6ca2b3-2bc2-4d38-8cc3-c59e398dc8f8",
            "0ce8313a-d1e7-41d6-9a67-e2e36433a5f5",
            "e914b74e-81a2-4f69-a3b5-4160686d0074",
            "65a21152-db8e-48ad-853b-d72a0c97b7c6",
            "aaf2ae76-1c7b-457f-9624-ae0af8925cd7",
            "ac7c6868-0d74-452a-a7a0-4aa8aaf3b987",
            "dd56678a-9e3a-4071-a1fc-c00b21ec7c90",
        )
        self.assertEqual(len(release_checker.PUBLIC_MEDIA_ASSETS), len(assets))
        for asset in assets:
            with self.subTest(asset=asset):
                url = "https://" + "github.com/user-attachments/assets/" + asset
                self.assertTrue(release_checker.approved_url(url))
                self.assertEqual(release_checker.automatic_issues(url, Path("README.md")), [])

    def test_reviewed_video_upload_permission_is_exact(self):
        path = "/user-attachments/assets/" + "b9a7de95-ba37-46f3-bf13-eddd0e51d600"
        url = "https://" + "github.com" + path
        for rejected in (
            "https://" + "github.com/user-attachments/assets/" + "00000000-0000-0000-0000-000000000000",
            url + "/",
            url + "/private",
            url + "?download=1",
            url + "#fragment",
            "http://" + "github.com" + path,
            "https://" + "github.com:443" + path,
            "https://" + "www.github.com" + path,
            "https://" + "raw.githubusercontent.com" + path,
            "https://" + "user" + "@" + "github.com" + path,
        ):
            with self.subTest(url=rejected):
                self.assertFalse(release_checker.approved_url(rejected))
                self.assertIn(
                    "unapproved external account or URL",
                    release_checker.automatic_issues(rejected, Path("README.md")),
                )


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
