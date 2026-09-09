"""Collect language-labeled bimanual SO-101 demonstrations."""

from surgecount.runtime import cli


def main() -> int:
    return cli(deployment=False)


if __name__ == "__main__":
    raise SystemExit(main())
