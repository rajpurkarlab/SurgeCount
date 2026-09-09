"""Run a local 13-dimensional skill checkpoint with manual or Gemini selection."""

from surgecount.runtime import cli


def main() -> int:
    return cli(deployment=True)


if __name__ == "__main__":
    raise SystemExit(main())
