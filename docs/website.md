# Anonymous project page

The static page in `website/` is prepared for **Anonymous submission · ML4H
2026**. The title follows the revised working draft. It does not claim
acceptance. The supplied layout, results, and eight short demonstration videos are
retained. The architecture figure is retained following visual review. The page is self-contained: it
uses local styles, JavaScript, images, and videos, with system fonts and no
analytics, embedded remote players, or third-party requests.

Preview from the release root:

```bash
python3 tools/check_website.py --media
python3 -m http.server 8000 --bind 127.0.0.1 --directory website
```

Open `http://127.0.0.1:8000/`. Serving the `website/` directory alone prevents
the surrounding repository and working documents from being part of the
preview. Only this directory is a website artifact. No hosting credentials,
hostnames, server addresses, or deployment configuration are included.

## Content and anonymization

- Anonymous author and venue labels appear in the page and metadata. Old
  venue mentions and the unused citation block are removed.
- Placeholder Paper, arXiv, and Code buttons are removed. The working draft
  is not included or linked. No author, institution, personal account,
  dataset, checkpoint, repository, or contact links are published.
- HTML comments, unused template scripts, unused images, and the original
  favicon are omitted. Template credit is retained in a local notice.
- The eight existing short MP4s are remuxed without audio, data/timecode
  tracks, chapters, or descriptive metadata. Encoded video is copied without
  re-encoding. Images are saved without EXIF, XMP, or text metadata.
- Six frames distributed across each clip were visually reviewed, along
  with the posters and architecture figure. The architecture figure is
  retained following review of its training inset, which includes a partial
  operator profile. This is a sampled visual review; metadata validation
  cannot certify every visible frame.

The written system overview and page prose use "subtask-conditioned VLA."
This cleanup does not revise manuscript results or experimental claims.

## Validation

`tools/check_website.py` checks all local HTML asset/link targets and fragment
IDs, rejects external resource URLs, personal contact links, host addresses,
old venue labels, missing anonymous metadata, hidden files, and document
downloads. It scans text inside comments as well as visible text. Optional
`--deny-term` arguments allow reviewers to check additional private
identifiers without saving those identifiers in this anonymous release.

`--media` also validates the MP4 streams and metadata using `ffprobe` and
checks PNG/JPEG metadata using Python's standard library. It requires an
installed FFmpeg distribution that provides `ffprobe`; the static page and
ordinary link checks need only Python 3. Recreating the sanitized media from
original sources additionally requires `ffmpeg` and Pillow.

```bash
python3 -m unittest discover -s tests -p 'test_website.py'
```

Before replacing any media, visually inspect it and rerun the checks.
Robots metadata and `robots.txt` request exclusion from indexing; they do
not provide access control.
