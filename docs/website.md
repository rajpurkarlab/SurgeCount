# Anonymous project page

The static page in `website/` presents SurgeCount as an **Anonymous submission
· ML4H 2026**, with the system architecture, measured results, and eight
demonstration videos. The page uses local styles, JavaScript, images, and
videos with system fonts.

Datasets and checkpoints will be released upon publication.

Preview from the release root:

```bash
python3 tools/check_website.py --media
python3 -m http.server 8000 --bind 127.0.0.1 --directory website
```

Open `http://127.0.0.1:8000/`. Serve `website/` as the complete web artifact.

## Content and anonymization

- Anonymous author and ML4H 2026 venue labels appear in the page and metadata.
- All page resources and links resolve locally. Template credit and license
  attribution appear in the local notice.
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

`tools/check_website.py` checks local HTML asset/link targets and fragment
IDs and rejects external resource URLs, personal contact links, host addresses,
old venue labels, missing anonymous metadata, hidden files, and document
downloads. It scans text inside comments as well as visible text. Optional
`--deny-term` arguments allow reviewers to check additional private
identifiers supplied at invocation time.

`--media` also validates the MP4 streams and metadata using `ffprobe` and
checks PNG/JPEG metadata using Python's standard library. It requires an
installed FFmpeg distribution that provides `ffprobe`; the static page and
ordinary link checks need only Python 3. Recreating the sanitized media from
original sources additionally requires `ffmpeg` and Pillow.

```bash
python3 -m unittest discover -s tests -p 'test_website.py'
```

Visually inspect replacement media and rerun the checks before publishing.
Robots metadata and `robots.txt` request exclusion from indexing. Configure
access control at the hosting layer when restricting who can view the page.
