"""Anonymous release audits cover source, opaque files and the exact ZIP export."""

import importlib.util
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_release", ROOT / "tools/check_release.py")
release_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_check)
sys.modules["check_release"] = release_check
spec = importlib.util.spec_from_file_location("package_release", ROOT / "tools/package_release.py")
package_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_release)


def write(root, name, value):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(value.encode() if isinstance(value, str) else value)
    return target


def minimal_release(root):
    root.mkdir()
    for name in release_check.REQUIRED:
        write(root, name, "")
    return root


def test_release_tree_is_clean():
    assert release_check.check_release(ROOT) == []


def test_data_package_is_source_but_runtime_datasets_are_rejected(tmp_path):
    write(tmp_path, "vendor/lerobot/src/lerobot/datasets/reader.py", "# dataset implementation\n")
    write(tmp_path, "datasets/raw/frame.parquet", b"raw recording")
    issues = release_check.check_release(tmp_path)
    assert any("frame.parquet: runtime" in issue for issue in issues)
    assert not any("reader.py" in issue for issue in issues)


def test_secrets_rejected_even_in_opaque_vendor_file(tmp_path):
    write(tmp_path, "vendor/settings.dat", b"\xff\x00" + ("hf_" + "a" * 30).encode())
    assert any("credential" in issue for issue in release_check.check_release(tmp_path))


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
def test_supplied_terms_scan_binary_bytes_and_case_insensitively(tmp_path, encoding):
    term = "outside" + "-owner"
    write(tmp_path, "website/image.bin", b"\xff\x00" + term.upper().encode(encoding) + b"\x00\xff")
    issues = release_check.check_release(tmp_path, (term,))
    assert any("image.bin: supplied private identifier" in issue for issue in issues)


def test_supplied_terms_scan_filenames_with_unrecognized_extension(tmp_path):
    term = "outside" + "-owner"
    write(tmp_path, "docs/" + term + ".opaque", b"no text")
    assert any("supplied private identifier" in issue for issue in release_check.check_release(tmp_path, (term,)))


@pytest.mark.parametrize("extension", ["json", "md", "mdx", "rst", "lock", "svg", "proto", "in"])
def test_nonlocal_hf_identifiers_detected_in_every_text_format(tmp_path, extension):
    account, repository = "unlisted-account", "hidden-demonstrations"
    write(tmp_path, "docs/reference." + extension, 'dataset_repo_id = "' + account + "/" + repository + '"')
    assert any("account identifier" in issue for issue in release_check.check_release(tmp_path))


def test_unquoted_repository_identifier_in_prose_is_rejected(tmp_path):
    account, repository = "unlisted-account", "hidden-demonstrations"
    write(tmp_path, "docs/reference.md", account + "/" + repository + "\n")
    assert any("account identifier" in issue for issue in release_check.check_release(tmp_path))


def test_unknown_hf_url_and_github_repository_rejected(tmp_path):
    account, repository = "unlisted-account", "hidden-demonstrations"
    for host, prefix in (("huggingface.co", "datasets/"), ("github.com", "")):
        write(tmp_path, "docs/link.md", "https://" + host + "/" + prefix + account + "/" + repository)
        assert any("external account" in issue for issue in release_check.check_release(tmp_path))


def test_upstream_base_and_dependency_links_are_allowed(tmp_path):
    write(
        tmp_path,
        "docs/deps.md",
        "`lerobot/pi0_base`\nhttps://huggingface.co/lerobot/pi0_base\n"
        "https://github.com/huggingface/lerobot\nhttps://files.pythonhosted.org/package.whl\n",
    )
    assert not any("deps.md" in issue for issue in release_check.check_release(tmp_path))


def test_machine_paths_and_direct_contacts_rejected_in_first_party(tmp_path):
    user_path = "/" + "home" + "/" + "named-person" + "/" + "project"
    contact = "researcher" + "@" + "institution.example"
    write(tmp_path, "configs/private.json", user_path)
    write(tmp_path, "docs/contact.rst", contact)
    issues = release_check.check_release(tmp_path)
    assert any("personal machine path" in issue for issue in issues)
    assert any("author/contact email" in issue for issue in issues)


def test_vendor_attribution_is_preserved_but_supplied_terms_still_apply(tmp_path):
    contact = "upstream-person" + "@" + "institution.example"
    write(tmp_path, "vendor/lerobot/LICENSE", "Copyright upstream contributor " + contact)
    assert not any("LICENSE:" in issue for issue in release_check.check_release(tmp_path))
    assert any("supplied private identifier" in issue for issue in release_check.check_release(tmp_path, (contact,)))


def test_bare_revision_rejected_but_upstream_commit_link_allowed(tmp_path):
    revision = "a1" * 20
    write(tmp_path, "docs/provenance.md", "Source revision: " + revision)
    assert any("revision fingerprint" in issue for issue in release_check.check_release(tmp_path))
    write(tmp_path, "docs/provenance.md", "https://github.com/huggingface/lerobot/commit/" + revision)
    assert not any("provenance.md" in issue for issue in release_check.check_release(tmp_path))


def test_version_numbers_in_lock_do_not_hide_real_network_address(tmp_path):
    version = ".".join(("1", "2", "3", "4"))
    address = ".".join(("10", "23", "45", "67"))
    write(tmp_path, "uv.lock", 'version = "' + version + '"\n')
    assert not any("uv.lock:" in issue for issue in release_check.check_release(tmp_path))
    write(tmp_path, "uv.lock", 'endpoint = "' + address + '"\n')
    assert any("IP address" in issue for issue in release_check.check_release(tmp_path))


def test_historical_resource_slug_is_rejected_without_identity_denylist(tmp_path):
    slug = "policy_" + "321" + "_clean_singlecam"
    write(tmp_path, "configs/deploy.json", 'checkpoint = "' + slug + '"')
    assert any("historical dataset/checkpoint identifier" in issue for issue in release_check.check_release(tmp_path))


@pytest.mark.parametrize("name", [".DS_Store", "._image.png", "__MACOSX" + "/item", "Thumbs.db", "desktop.ini"])
def test_os_metadata_rejected_and_never_exported(tmp_path, name):
    path = write(tmp_path, "vendor/" + name, b"metadata")
    assert any("operating-system metadata" in issue for issue in release_check.check_release(tmp_path))
    assert path not in set(package_release.source_files(tmp_path))


def test_excluded_caches_not_exported_and_unsafe_tracked_paths_reported(tmp_path):
    tracked = ".venv" + "/private.dat"
    write(tmp_path, tracked, b"local cache")
    assert not list(package_release.source_files(tmp_path))
    assert any(
        "unsafe tracked artifact" in issue for issue in release_check.check_release(tmp_path, tracked_paths=[tracked])
    )


def test_export_never_follows_symlinks(tmp_path):
    root = minimal_release(tmp_path / "source")
    other = write(tmp_path, "outside.txt", "private note")
    (root / "docs").mkdir()
    (root / "docs/reference").symlink_to(other)
    assert any("symlink" in issue for issue in release_check.check_release(root))
    assert root / "docs/reference" not in set(package_release.source_files(root))


def test_exact_archive_uses_neutral_names_dates_and_bytes(tmp_path):
    root = minimal_release(tmp_path / "source")
    write(root, "website/static/image.dat", b"\xff\x01\x02binary")
    write(root, "local-note.txt", "not in allowlisted source set")
    output = tmp_path / "anonymous-code.zip"
    count = package_release.build_archive(output, root)
    assert package_release.verify_archive(output, root) == []
    with zipfile.ZipFile(output) as archive:
        assert len(archive.namelist()) == count
        assert not any("local-note" in name for name in archive.namelist())
        assert not archive.comment
        assert all(item.date_time == (1980, 1, 1, 0, 0, 0) for item in archive.infolist())
        assert all(name.startswith("surgecount/") for name in archive.namelist())
    write(root, "README.md", "changed after packaging")
    assert any("bytes differ" in issue for issue in package_release.verify_archive(output, root))


def test_archive_verifier_rejects_added_unsafe_members_and_comments(tmp_path):
    root = minimal_release(tmp_path / "source")
    output = tmp_path / "anonymous-code.zip"
    package_release.build_archive(output, root)
    with zipfile.ZipFile(output, "a") as archive:
        archive.comment = b"identifying build note"
        archive.writestr("../outside.txt", "unexpected")
    issues = package_release.verify_archive(output, root)
    assert any("member set" in issue for issue in issues)
    assert any("unsafe member path" in issue for issue in issues)
    assert any("ZIP comment" in issue for issue in issues)


def test_archive_verifier_scans_opaque_bytes_for_external_deny_terms(tmp_path):
    root = minimal_release(tmp_path / "source")
    term = "outside" + "-owner"
    write(root, "website/picture.dat", b"\xff\x00" + term.encode("utf-16-le"))
    output = tmp_path / "anonymous-code.zip"
    package_release.build_archive(output, root)
    assert any(
        "supplied private identifier" in issue for issue in package_release.verify_archive(output, root, (term,))
    )
