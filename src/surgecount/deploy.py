"""Run a local 13-dimensional pi0.5 checkpoint with manual or Gemini skills."""

from surgecount.runtime import cli


def main() -> int:
    return cli(deployment=True)


if __name__ == "__main__":
    raise SystemExit(main())
