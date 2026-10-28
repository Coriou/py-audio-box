#!/usr/bin/env python3
"""Native Kokoro CPU setup and installed public service commands."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from service.native_cpu_bootstrap import main
if __name__=='__main__':raise SystemExit(main())
