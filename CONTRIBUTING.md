# Contributing

Use Python 3.12 and the vendored LeRobot source. Run `uv sync --frozen
--no-editable --python 3.12 --extra data --extra dev` for data/configuration
work, or add the `train`, `robot`, and
`planner` extras for those components. Do not install an unmodified PyPI
LeRobot over the vendored package.

```bash
uv run --no-sync pytest
uv run --no-sync ruff check src tests tools
uv run --no-sync python tools/check_release.py
uv run --no-sync python tools/check_website.py --media
```

Keep tests independent of physical devices, cloud API calls, downloaded
checkpoints, and training data. New data preparation must write a new
destination and leave its source untouched. Keep task strings, task IDs,
camera masks, joint units, and stop semantics consistent across collection,
training, and deployment. Never commit credentials, local hardware
configuration, calibration, raw recordings, or model artifacts.

Datasets and checkpoints will be released upon publication. During review,
keep project accounts, artifact links, personal paths, and identifying commit
metadata out of contributions. See [anonymous release notes](docs/anonymity.md).

When modifying the vendor package, retain original license notices and
document the change in `docs/provenance.md`. For robot changes, record the
offline tests performed and separately report physical validation; passing
configuration tests does not establish hardware behavior.
