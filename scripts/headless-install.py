#!/usr/bin/env python3
"""Distribution entry point; imports only from this installed distribution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.package_install import main
if __name__ == '__main__':
    main()
