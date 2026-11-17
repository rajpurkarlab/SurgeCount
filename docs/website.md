# Repository front page and demo media

The [repository README](../README.md) is SurgeCount's main project page for
**ML4H 2026**, with the system architecture, measured results, and a visual
and video gallery. The authors are **Sungeun Kim, Xiaoman Zhang, Xucheng Wang,
and Pranav Rajpurkar**.

The canonical project URL is
[github.com/rajpurkarlab/SurgeCount](https://github.com/rajpurkarlab/SurgeCount).
The gallery uses repository images and GitHub-hosted video attachments that
play inline in the README. The attachments are copies of all eight full MP4 files in
[`website/static/videos/`](../website/static/videos/). Poster images remain
in [`website/static/images/posters/`](../website/static/images/posters/).

Datasets and checkpoints will be released upon publication.

## Optional offline page

`website/index.html` provides an optional static copy of the project
presentation. Its styles, JavaScript, images, and videos are local, and it
uses system fonts. Preview it from the repository root:

```bash
python3 tools/check_website.py --public --media
python3 -m http.server 8000 --bind 127.0.0.1 --directory website
```

Open `http://127.0.0.1:8000/`. This preview serves only the `website/` directory.

## Content and media

- The README and static page identify the four authors and ML4H 2026 venue.
- Static-page resources and links resolve locally. Template credit and
  license attribution appear in the local notice; repository media terms
  are recorded in [THIRD_PARTY.md](../THIRD_PARTY.md).
- Each MP4 contains a single video stream with neutral container metadata.
  Remuxing preserves the encoded video and removes audio, data/timecode
  tracks, chapters, and descriptive metadata. Images have EXIF, XMP, and
  text metadata removed.
- Visual review sampled six frames distributed across each clip and covered
  the posters and architecture figure, including its training inset.

The system overview describes SurgeCount's language-conditioned policy with
twelve robot actions, a learned stop signal, and Gemini composition. The
end-to-end pi0.5 baseline uses a trained pi0-based subtask predictor to
condition twelve-dimensional actions at chunk boundaries.

## Validation

`tools/check_website.py --public` requires the four named authors in both
author and citation metadata and a visible ML4H 2026 label. It checks local
HTML targets and fragment IDs and rejects external resource URLs, personal
contact links, host addresses, old venue labels, hidden files, and document
downloads. Referrer and content-security policies remain required. Text
inside comments is included in the scan. Optional `--deny-term` arguments
check additional private identifiers supplied at invocation time.

CI validates the optional static page in public mode. The default anonymous mode
remains available for a separately prepared anonymous copy; see
[anonymous review copies](anonymity.md). The current named page is a public
artifact.

`--media` also validates the MP4 streams and metadata using `ffprobe` and
checks PNG/JPEG metadata using Python's standard library. It requires an
installed FFmpeg distribution that provides `ffprobe`; the static page and
ordinary link checks need only Python 3. Recreating the sanitized media from
original sources additionally requires `ffmpeg` and Pillow.

```bash
python3 -m unittest discover -s tests -p 'test_website.py'
```

Visually inspect replacement media and rerun the checks before adding it to
the repository gallery or optional static page.
