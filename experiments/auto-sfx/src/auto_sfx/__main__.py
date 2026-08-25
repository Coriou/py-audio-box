"""``python -m auto_sfx`` entry point (the Docker ENTRYPOINT target)."""

import sys

from auto_sfx.cli import main

if __name__ == "__main__":
    sys.exit(main())
