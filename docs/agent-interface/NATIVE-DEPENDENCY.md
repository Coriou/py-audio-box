# Native Qwen import dependency patch

This fixes a demonstrated dependency import / containment conflict, not synthesis.
The installed native spawn guard is unchanged. No process API is monkeypatched
to accommodate a dependency, no fake module or successful SoX probe is supplied,
and no broad upstream code tree is vendored into application imports.

## Choice and upstream provenance

The official [Qwen-TTS 0.1.1 wheel](https://pypi.org/project/qwen-tts/0.1.1/)
eagerly imports the 25 Hz model from `qwen_tts/core/__init__.py`. Its
`inference/qwen3_tts_tokenizer.py` also imports/registers that model for every
tokenizer request. The unused 25 Hz family imports PySox, whose pinned 1.4.1
initializer executes `os.popen('sox -h')`. The native guard correctly refuses
that nested launch before Qwen model loading. An import-only dependency check
outside the guard missed this conflict.

Upstream sources: [core imports](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/core/__init__.py),
[tokenizer dispatch](https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_tokenizer.py),
[Apache-2.0 license](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE),
[PySox 1.4.1](https://pypi.org/project/sox/1.4.1/).
The exact input below, rather than mutable GitHub main, defines patched bytes.
PySox 1.5.0 was also inspected and retains the import-time shell probe; it is not
a demonstrated fix and neither PySox nor other dependency versions are upgraded.

Build `1native12hz` changes only two upstream Python files: core exposes the
25 Hz model through lazy `__getattr__`; tokenizer dispatch reads the requested
configuration first, then imports/registers the 25 Hz model only when explicitly
selected. The 12 Hz model and all its real dependencies still import normally.
The 25 Hz configuration class remains available. An explicit 25 Hz request keeps
upstream behavior; 25 Hz is not supported by the native execution contract.
This adds one configuration lookup using the same loader arguments/cache; native
requests use an already verified local snapshot and offline flags.

The artifact retains distribution version 0.1.1 and uses a distinct wheel build
tag, WHEEL Build header, marked modified files and NOTICE. Original METADATA,
Requires-Dist, copyright and Apache-2.0 LICENSE are preserved. Application source
remains AGPL-3.0-only; dependency/model licenses are separate.

## Reproduction and install closure

- Public upstream wheel: `qwen_tts-0.1.1-py3-none-any.whl`, SHA256
  `11a290d8dabc7ef91a90c54478c8ab19b3edb1d85c0882313721892bdc4af15d`.
- Native artifact: `qwen_tts-0.1.1-1native12hz-py3-none-any.whl`, SHA256
  `96e5d3dfdcf2a86bcb59d9a68de7ef51bd6f591a63cdae816a7c1da4cad31606`.

The small upstream wheel is retained under `requirements/upstream/` for offline
audit/reproduction; it is not selected for installation. The native wheel is
under `requirements/wheels/`. Neither contains model weights. From checkout root:

```sh
python3 scripts/build-native-qwen-wheel.py \
  --upstream requirements/upstream/qwen_tts-0.1.1-py3-none-any.whl \
  --output /absolute/new-dir/qwen_tts-0.1.1-1native12hz-py3-none-any.whl
python3.11 -m pip install --require-hashes --only-binary=:all: \
  --index-url https://pypi.org/simple -r requirements/qwen-macos-arm64.lock
```

The builder has no network/install/model operations. It verifies the whole input
wheel and both original file hashes, checks exact replacement contexts, updates
RECORD hashes/sizes and writes sorted entries with fixed metadata using stored
ZIP members. Output must not already exist. Native `.in` and `.lock` select the
same local artifact; all other lock versions/hashes are unchanged. Commands must
run from checkout root because the wheel reference is root-relative. Linux
locks continue selecting the unmodified upstream wheel.

The native worker ZIP remains a source-only archive without wheels or an
installer. Its existing `lib/tts.py` carries the shared bootstrap used by the
real loader and import regression; this provenance document is also included.
For runtime manifests the operator must retain the exact native wheel as its
separate dependency artifact. Metadata version alone cannot distinguish builds.
Native wheel identity/tag/Requires-Dist checks and archive source closure are
tested; archive construction does not establish an installed deployment.

## Upgrade and qualification gate

Do not blindly regenerate the lock from PyPI: that would restore the upstream
eager import. A Qwen upgrade must re-evaluate both import sites and dispatch/API
semantics, the pinned native closure, license notices and build recipe. Changed
upstream bytes fail closed. Remove the patch when an official compatible release
fixes both sites, after the same guard regression passes. Do not accept a newly
computed hash without reviewing the source delta.

The branch-only Apple CI runs `scripts/ci-native-import.py`: a fresh isolated
interpreter installs the product permanent guard, calls the real loader's shared
bootstrap, imports actual 12 Hz classes and drives config-only 12 Hz dispatch to
the expected missing-weight error. It compares installed dependency bytes with
the locked wheel, checks PySox/25 Hz modeling stayed unloaded and verifies an
explicit spawn is still refused. Missing dependencies/import errors retain their
original exception and traceback; they are not successful skips. Parent deadline
is 60 seconds, bounded log storage and exact child reap remain enforced. Reports
state weights not loaded and runtime unqualified. No model acquisition or
inference is authorized by this gate. Independent Opus source review and a new
explicit runtime authorization remain required before further speech testing.
