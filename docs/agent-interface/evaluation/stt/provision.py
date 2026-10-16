"""Fetch only public pinned assets into a dedicated evaluation root."""
import argparse, csv, hashlib, json, tarfile, urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf

DATASET_REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
MODELS = {
    "small": ("Systran/faster-whisper-small", "536b0662742c02347bc0e980a01041f333bce120", ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]),
    "turbo": ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf", ["config.json", "model.bin", "tokenizer.json", "vocabulary.json", "preprocessor_config.json"]),
}

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def fetch(url, path):
    if path.exists(): return
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, stage.open("wb") as output:
        while block := response.read(1024 * 1024): output.write(block)
    stage.rename(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/assets"))
    args = parser.parse_args()
    root = args.root
    root.mkdir(parents=True, exist_ok=True)
    receipts, samples = [], []
    for language, config in [("fr", "fr_fr"), ("en", "en_us")]:
        base = f"https://huggingface.co/datasets/google/fleurs/resolve/{DATASET_REV}/data/{config}"
        tsv, archive = root / "sources" / f"{config}-test.tsv", root / "sources" / f"{config}-test.tar.gz"
        fetch(base + "/test.tsv", tsv)
        rows = list(csv.reader(tsv.read_text().splitlines(), delimiter="\t"))
        chosen, ids = [], set()
        for row in rows:
            if row[0] in ids: continue
            chosen.append(row); ids.add(row[0])
            if len(chosen) == 3: break
        fetch(base + "/audio/test.tar.gz", archive)
        wanted = {row[1]: row for row in chosen}
        with tarfile.open(archive, "r:gz") as tar:
            for item in tar:
                basename = Path(item.name).name
                if basename not in wanted or not item.isfile(): continue
                dest = root / "corpus" / f"{language}-{basename}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(item).read())
                audio, rate = sf.read(dest, dtype="float32")
                row = wanted.pop(basename)
                samples.append(dict(id=f"{language}-{row[0]}", language=language, file=str(dest), text=row[2], dataset_text_normalized=row[3], source_row_id=row[0], source_filename=basename, duration=len(audio)/rate, sha256=digest(dest), kind="natural_read_speech"))
                if not wanted: break
        if wanted: raise RuntimeError(f"Missing public samples: {list(wanted)}")
        receipts += [dict(url=base + "/test.tsv", path=str(tsv), bytes=tsv.stat().st_size, sha256=digest(tsv)), dict(url=base + "/audio/test.tar.gz", path=str(archive), bytes=archive.stat().st_size, sha256=digest(archive))]
    samples.sort(key=lambda x: (x["language"], x["id"]))
    rate = 16000
    for name, audio in [("silence", np.zeros(4*rate, dtype=np.float32)), ("noise", np.random.default_rng(1729).normal(0, .02, 4*rate).astype(np.float32))]:
        dest = root / "corpus" / f"{name}.wav"; sf.write(dest, audio, rate, subtype="PCM_16")
        samples.append(dict(id=name, language=None, file=str(dest), text="", duration=4, sha256=digest(dest), kind="synthetic_negative"))
    left = next(x for x in samples if x["language"] == "fr")
    right = next(x for x in samples if x["language"] == "en")
    a, ar = sf.read(left["file"], dtype="float32"); b, br = sf.read(right["file"], dtype="float32")
    if ar != rate or br != rate: raise RuntimeError("Unexpected FLEURS rate")
    dest = root / "corpus" / "constructed-fr-en.wav"
    sf.write(dest, np.concatenate([a, np.zeros(rate, dtype=np.float32), b]), rate, subtype="PCM_16")
    samples.append(dict(id="constructed-fr-en", language=None, file=str(dest), text=left["text"] + " " + right["text"], duration=len(a)/rate+1+len(b)/rate, sha256=digest(dest), kind="constructed_concat_not_natural_code_switch", source_samples=[left["id"], right["id"]]))
    for label, (repo, revision, files) in MODELS.items():
        for filename in files:
            dest = root / "models" / label / filename
            url = f"https://huggingface.co/{repo}/resolve/{revision}/{filename}"
            print(f"fetch {label}/{filename}", flush=True); fetch(url, dest)
            receipts.append(dict(url=url, path=str(dest), bytes=dest.stat().st_size, sha256=digest(dest)))
    (root / "manifest.json").write_text(json.dumps(dict(dataset="google/fleurs", dataset_revision=DATASET_REV, license="CC-BY-4.0", dataset_citation="Conneau et al., FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech (2022)", models={k: dict(repo=v[0], revision=v[1]) for k,v in MODELS.items()}, samples=samples, sources=receipts, total_download_bytes=sum(x["bytes"] for x in receipts)), indent=2, ensure_ascii=False))
    print(json.dumps(dict(samples=len(samples), download_bytes=sum(x["bytes"] for x in receipts))), flush=True)

if __name__ == "__main__": main()
