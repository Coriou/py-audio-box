"""Separate source-only CPU archive; native-qwen-v1 admission is unchanged."""
import json
from pathlib import Path
from service import native_package as common
from worker.transport import Refused

VARIANT='native-kokoro-cpu-v1'
LAUNCHER=b"import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom service.native_cpu_bootstrap import main\nraise SystemExit(main())\n"


def members():
    value=json.loads((common.ROOT/'native-cpu-dist-manifest.json').read_bytes())
    if value['variant']!=VARIANT or value['qualification']!='unqualified':raise Refused('native_cpu_package_variant')
    files=value['files']
    if len(set(files))!=len(files) or any(Path(n).is_absolute() or '..' in Path(n).parts for n in files):raise Refused('native_cpu_package_members')
    return files


def archive(destination):return common._archive(destination,members(),LAUNCHER)


def verify_archive(path):return common._verify_archive(path,members(),LAUNCHER)
