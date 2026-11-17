# Public source and anonymous review copies

The repository README is the public project page and identifies SurgeCount
and its authors: Sungeun Kim, Xiaoman Zhang, Xucheng Wang, and Pranav Rajpurkar.
Its visual and video gallery uses repository media, including the full MP4s
in `website/static/videos/`. The optional static HTML copy is described in
[website.md](website.md).

**Datasets and checkpoints will be released upon publication.** Training
examples continue to use local artifact placeholders. Credentials, private
machine paths, device identifiers, and runtime data stay outside the source
release. Upstream dependencies retain their copyright and license credits.

## Public release checks

Run these checks from the repository root:

```bash
python3 tools/check_release.py
python3 tools/check_website.py --public --media
```

These checks preserve credential, private-path, runtime-artifact, and
media-container guards. The canonical project destination is
[rajpurkarlab/SurgeCount](https://github.com/rajpurkarlab/SurgeCount).
A passing source check validates release contents; anonymous review
requires the separate preparation below.

## Prepare an anonymous review copy

Use a separate source copy. Replace project and author identifiers throughout
its source, documentation, tests, metadata, and media, including the current
public validator fixtures and reviewed-link constants. Preserve required
upstream attribution. The anonymous website needs anonymous author/citation
metadata, the anonymous ML4H 2026 label, and `noindex` metadata.

Validate that prepared copy explicitly:

```bash
python3 tools/check_release.py /path/to/anonymous-copy --deny-term "identifier-to-remove"
python3 tools/check_website.py --root /path/to/anonymous-copy/website --media \
  --deny-term "identifier-to-remove"
```

Omitting `--public` selects anonymous website checks. Supply private search
terms at invocation time rather than recording a private deny list in source.
Inspect the rendered README, optional static page, and replacement media as
well as the scan results.

The source ZIP utility exports a prepared tree with relative paths and
neutral archive metadata. It excludes Git history, environments, runtime
outputs, caches, and operating-system metadata. It packages the supplied
content, including its author names, project links, and images.

```bash
python3 tools/package_release.py /path/to/review-copy.zip --root /path/to/anonymous-copy
```

Review the resulting archive and the review hosting interface for identifying
content before sharing. Keep real hardware configuration, datasets,
checkpoints, and generated run manifests outside the review copy.
