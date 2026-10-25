"""Explicit installed native control entry; imports this source distribution."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from service.native_launch import main
if __name__ == '__main__':
    main()
