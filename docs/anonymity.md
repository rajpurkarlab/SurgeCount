# Anonymous review release

**Datasets and checkpoints will be released upon publication.** Project
accounts, dataset/checkpoint repository names and links, personal paths,
machine addresses, private source-history identifiers, and author contact
details are omitted from this release. Examples use local placeholders.
Upstream dependencies retain their required copyright and license credits.

## Exporting the repository

Run the checks from the repository root, then create a clean source ZIP:

```bash
python3 tools/check_release.py
python3 tools/check_website.py --media
python3 tools/package_release.py /path/to/SurgeCount_anonymous.zip
```

The archive includes the intended source, documentation, configuration,
tests, vendored dependencies, and website. It excludes Git history, local
environments, runtime outputs, caches, operating-system metadata, and
generated package metadata. The export uses relative member paths and
neutral ZIP metadata. Private terms can be supplied to the release checker
with `--deny-term` without adding them to this public repository.

The clean source can be hosted in a personal repository and shared for
review through an anonymizing repository viewer. Share the review viewer's
link with reviewers and inspect that rendered copy for account details,
links, and metadata. A repository owned by an anonymous account is another
option. Keep real hardware configuration, local datasets, checkpoints,
and generated run manifests outside the public repository.

The website uses local assets and has been checked for identifying text
and media metadata. Visual review covered six frames per clip, all posters,
and the architecture figure; see [website notes](website.md).
Review replacement media and rerun
the release checks before updating the public artifact.
