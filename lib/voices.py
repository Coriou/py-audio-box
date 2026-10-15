"""
lib/voices.py — shared named voice registry for all py-audio-box apps.

Each named voice lives at:

    /cache/voices/<slug>/
        voice.json                            registry metadata
        source_clip.wav                       raw best clip (from voice-split or user)
        ref.wav                               24 kHz processed segment (from voice-clone)
        prompts/
            <model_tag>_<mode>_v<N>.pkl
            <model_tag>_<mode>_v<N>.meta.json

``voice.json`` schema (all fields except ``slug`` and ``created_at`` are mutable)::

    {
        "slug":         "david-attenborough",
        "display_name": "David Attenborough",
        "description":  "...",
        "created_at":   "<iso8601>",
        "updated_at":   "<iso8601>",
        "source": {
            "type":      "file" | "youtube" | "designed",
            "path":      "...",        // original file (if type=file)
            "url":       "...",        // YouTube URL (if type=youtube)
            "video_id":  "...",        // YouTube video ID (if type=youtube)
            "instruct":  "...",        // voice description (if type=designed)
            "ref_text":  "..."         // design ref text (if type=designed)
        },
        "ref": {                       // null until voice-clone prepare-ref is run
            "hash":            "...",
            "segment_start":   0.0,
            "segment_end":     8.5,
            "duration_sec":    8.5,
            "transcript":      "...",
            "transcript_conf": -0.4,
            "language":        "English",
            "language_prob":   0.99
        },
        "prompts": {                   // stem -> relative path; index only
            "<stem>": "prompts/<stem>.pkl"
        },
        "tones": {                     // tone label -> stem; set by --tone on voice-clone synth
            "neutral": "<stem1>",        // e.g. "neutral", "sad", "excited"
            "sad":     "<stem2>"
        },
        "engine": "clone_prompt" | "custom_voice" | "designed_clone",
        "custom_voice": {             // present when engine=custom_voice
            "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "speaker": "Ryan",
            "instruct_default": "Warm, calm delivery",
            "language_default": "English",
            "tones": {                // tone label -> instruction preset
                "neutral": "Calm, clear, broadcast tone",
                "excited": "Energetic, upbeat promo read"
            }
        },
        "generation_defaults": {      // optional per-voice sampling defaults
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.05,
            "max_new_tokens_policy": "balanced"
        }
    }

Import from an app::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "lib"))
    from voices import VoiceRegistry, validate_slug
"""

import json
import os
import tempfile
import threading
import fcntl
import stat
import zipfile
from functools import wraps
from contextlib import contextmanager
from pathlib import PurePosixPath
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Slug rules: lowercase letters, digits, hyphens; must start and end with alnum;
# min 1 char, max 64 chars.
SLUG_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,62}[a-z0-9])?$")
VALID_ENGINES = {"clone_prompt", "custom_voice", "designed_clone"}


def infer_engine(data: dict[str, Any]) -> str:
    """
    Infer synthesis engine mode for a voice record.

    Backward-compatibility rules:
      - explicit ``engine`` wins when valid
      - ``custom_voice.speaker`` implies ``custom_voice``
      - source.type == "designed" implies ``designed_clone``
      - otherwise default to ``clone_prompt``
    """
    explicit = str(data.get("engine", "")).strip()
    if explicit in VALID_ENGINES:
        return explicit

    profile = data.get("custom_voice")
    if isinstance(profile, dict) and str(profile.get("speaker", "")).strip():
        return "custom_voice"

    source = data.get("source") or {}
    if isinstance(source, dict) and source.get("type") == "designed":
        return "designed_clone"

    return "clone_prompt"


def validate_slug(slug: str) -> str:
    """
    Normalise *slug* to lowercase and validate it.
    Returns the normalised slug or raises ``ValueError``.
    """
    if not isinstance(slug, str):
        raise ValueError("Voice name must be a string")
    slug = slug.strip().lower()
    if not SLUG_RE.match(slug):
        raise ValueError(
            f"Invalid voice name {slug!r}. "
            "Use only lowercase letters, digits, and hyphens "
            "(e.g. 'david-attenborough', 'my-narrator-v2')."
        )
    return slug


def _locked(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock():
            return method(self, *args, **kwargs)
    return wrapped


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".json-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class VoiceRegistry:
    """
    Thin wrapper around ``/cache/voices/`` that provides create/read/update
    operations for named voices.

    Mutations hold a process lock across read/modify/write and replace JSON atomically.
    """

    def __init__(self, cache: Path) -> None:
        self.root = Path(cache) / "voices"
        self._mutex = threading.RLock()
        self._lock_depth = 0

    @contextmanager
    def _lock(self):
        with self._mutex:
            self.root.mkdir(parents=True, exist_ok=True)
            if self._lock_depth:
                yield
                return
            with open(self.root / ".registry.lock", "a") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                self._lock_depth += 1
                try:
                    yield
                finally:
                    self._lock_depth -= 1
                    fcntl.flock(fh, fcntl.LOCK_UN)

    # ── path helpers ────────────────────────────────────────────────────────────

    def voice_dir(self, slug: str) -> Path:
        return self.root / validate_slug(slug)

    def voice_json(self, slug: str) -> Path:
        return self.root / validate_slug(slug) / "voice.json"

    def source_clip(self, slug: str) -> Path:
        return self.root / validate_slug(slug) / "source_clip.wav"

    def ref_wav(self, slug: str) -> Path:
        return self.root / validate_slug(slug) / "ref.wav"

    def prompts_dir(self, slug: str) -> Path:
        return self.root / validate_slug(slug) / "prompts"

    # ── read ────────────────────────────────────────────────────────────────────

    def exists(self, slug: str) -> bool:
        return self.voice_json(slug).exists()

    def _normalise_record(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Normalise optional/additive fields for backward compatibility.

        Does not write to disk; callers may persist explicitly after mutation.
        """
        if not isinstance(data, dict):
            raise ValueError("Voice metadata must be a JSON object")
        if data.get("schema_version", 1) not in (1, 2):
            raise ValueError("Unsupported voice registry schema")
        if not isinstance(data.get("prompts"), dict):
            data["prompts"] = {}

        if "tones" in data and not isinstance(data.get("tones"), dict):
            data["tones"] = {}
        data.setdefault("tones", {})

        source = data.get("source")
        if source is None or not isinstance(source, dict):
            data["source"] = {}

        engine = infer_engine(data)
        data["engine"] = engine

        custom = data.get("custom_voice")
        if custom is None or not isinstance(custom, dict):
            custom = {}
        tones = custom.get("tones")
        if tones is None or not isinstance(tones, dict):
            custom["tones"] = {}
        data["custom_voice"] = custom

        if "generation_defaults" in data and not isinstance(data.get("generation_defaults"), dict):
            data["generation_defaults"] = {}

        return data

    def load(self, slug: str) -> dict[str, Any]:
        vj = self.voice_json(slug)
        if not vj.exists():
            raise KeyError(
                f"Voice '{slug}' not found in registry. "
                f"Looked in: {vj}"
            )
        with open(vj) as fh:
            data = json.load(fh)
        return self._normalise_record(data)

    def get_ref(self, slug: str) -> Path | None:
        """
        Return the best available reference WAV for *slug*, in preference order:
        1. ``ref.wav``        — processed 24 kHz segment (best for cloning)
        2. ``source_clip.wav`` — raw clip (needs voice-clone processing)
        Returns ``None`` if neither exists.
        """
        ref = self.ref_wav(slug)
        if ref.exists():
            return ref
        src = self.source_clip(slug)
        if src.exists():
            return src
        return None

    def best_prompt(self, slug: str, **compatibility) -> Path | None:
        """
        Return the most recently written ``.pkl`` in this voice's prompts/,
        or ``None`` if no prompts exist yet.
        """
        pd = self.prompts_dir(slug)
        if not pd.exists():
            return None
        pkls = sorted(pd.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return next((p for p in pkls if not compatibility or self.compatible_prompt(p, **compatibility)), None)

    def prompt_reference(self, path: Path) -> Path | None:
        """Return the immutable per-prompt reference, confined to its cache directory."""
        path = Path(path)
        try:
            meta = json.loads(path.with_suffix(".meta.json").read_text())
            name = meta.get("reference_wav")
            if name != path.stem + ".ref.wav":
                return None
            reference = path.parent / name
            if reference.is_symlink() or not reference.is_file():
                return None
            return reference
        except (OSError, ValueError, TypeError):
            return None

    def compatible_prompt(self, path: Path, *, checkpoint: str, revision: str | None = None,
                          reference_hash: str | None = None, schema_version: int | None = None,
                          mode: str | None = None) -> bool:
        """Check identity without deserialising executable local cache bytes."""
        import hashlib
        path = Path(path)
        try:
            if path.is_symlink() or not path.is_file():
                return False
            meta = json.loads(path.with_suffix(".meta.json").read_text())
            if not isinstance(meta, dict) or meta.get("engine") != "qwen3":
                return False
            if revision is None or meta.get("checkpoint") != checkpoint or meta.get("revision") != revision:
                return False
            if not isinstance(meta.get("schema_version"), int) or meta.get("mode") not in ("full", "xvec"):
                return False
            if schema_version is not None and meta["schema_version"] != schema_version:
                return False
            if mode is not None and meta["mode"] != mode:
                return False
            if meta.get("cache_identity_version") != 1 or not meta.get("reference_sha256"):
                return False
            if reference_hash is not None and meta.get("reference_sha256") != reference_hash:
                return False
            return meta.get("pickle_sha256") == hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, ValueError, TypeError):
            return False

    def matching_prompt(self, slug: str, *, checkpoint: str, revision: str | None,
                        schema_version: int, tone: str | None = None) -> Path | None:
        """Select an existing fully validated model variant without loading pickle."""
        import hashlib
        current_tones = self.list_tones(slug)
        candidates = sorted(self.prompts_dir(slug).glob("*.pkl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates:
            try:
                meta = json.loads(path.with_suffix(".meta.json").read_text())
                if not isinstance(meta, dict):
                    continue
                if tone is not None and not (meta.get("tone") == tone or current_tones.get(tone) == path.stem):
                    continue
                reference = self.prompt_reference(path)
                if reference is None:
                    continue
                reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
                if self.compatible_prompt(path, checkpoint=checkpoint, revision=revision,
                        schema_version=schema_version, reference_hash=reference_hash):
                    return path
            except (OSError, ValueError, TypeError):
                continue
        return None

    def list_voices(self) -> list[dict[str, Any]]:
        """
        Return all named voices, sorted by creation date (newest first).
        Each entry is the voice.json dict with two extra keys:
          ``_prompt_count``  — number of .pkl files in prompts/
          ``_ready``         — True if a prompt exists (can synthesise immediately)
        """
        if not self.root.exists():
            return []
        voices: list[dict[str, Any]] = []
        for vj in sorted(self.root.glob("*/voice.json")):
            try:
                with open(vj) as fh:
                    d = self._normalise_record(json.load(fh))
            except Exception:
                continue
            pd = vj.parent / "prompts"
            count = len(list(pd.glob("*.pkl"))) if pd.exists() else 0
            d["_prompt_count"] = count
            d["_engine"] = infer_engine(d)
            if d["_engine"] == "custom_voice":
                profile = d.get("custom_voice") if isinstance(d.get("custom_voice"), dict) else {}
                source = d.get("source") if isinstance(d.get("source"), dict) else {}
                speaker = str(profile.get("speaker", "")).strip()
                # Model may be stored in either custom_voice.model (new schema)
                # or source.model (older/mixed records).
                model = str(profile.get("model", "")).strip() or str(source.get("model", "")).strip()
                d["_ready"] = bool(speaker and model)
            else:
                d["_ready"] = count > 0
            d["_has_ref"] = (vj.parent / "ref.wav").exists()
            d["_has_source"] = (vj.parent / "source_clip.wav").exists()
            voices.append(d)
        voices.sort(key=lambda v: v.get("created_at", ""), reverse=True)
        return voices

    # ── write ───────────────────────────────────────────────────────────────────

    def _save(self, slug: str, data: dict[str, Any]) -> None:
        vd = self.voice_dir(slug)
        vd.mkdir(parents=True, exist_ok=True)
        data["updated_at"] = datetime.now(timezone.utc).isoformat()
        data["schema_version"] = 2
        data["slug"] = validate_slug(slug)
        atomic_json(vd / "voice.json", data)

    @_locked
    def create(
        self,
        slug: str,
        display_name: str,
        source: dict[str, Any],
        description: str = "",
    ) -> dict[str, Any]:
        """
        Create a new voice entry. If the slug already exists, returns
        the existing record unchanged (idempotent).
        """
        if self.exists(slug):
            return self.load(slug)
        now = datetime.now(timezone.utc).isoformat()
        data: dict[str, Any] = {
            "slug":         slug,
            "display_name": display_name,
            "description":  description,
            "created_at":   now,
            "source":       source,
            "ref":          None,
            "prompts":      {},
            "tones":        {},
            "engine":       "designed_clone" if source.get("type") == "designed" else "clone_prompt",
        }
        self._save(slug, data)
        return data

    @_locked
    def set_source_clip(self, slug: str, wav_src: Path) -> Path:
        """
        Copy *wav_src* into the voice dir as ``source_clip.wav``.
        Returns the destination path.
        """
        dest = self.source_clip(slug)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if str(wav_src.resolve()) != str(dest.resolve()):
            shutil.copyfile(wav_src, dest)
        return dest

    @_locked
    def update_ref(self, slug: str, segment_wav: Path, ref_meta: dict[str, Any]) -> Path:
        """
        Record reference-segment metadata and copy *segment_wav* into the
        voice dir as ``ref.wav``.  Updates ``voice.json``.
        Returns the destination path.
        """
        dest = self.ref_wav(slug)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if str(segment_wav.resolve()) != str(dest.resolve()):
            shutil.copyfile(segment_wav, dest)
        data = self.load(slug)
        data["ref"] = {**ref_meta, "wav": "ref.wav"}
        self._save(slug, data)
        return dest

    @_locked
    def register_prompt(
        self,
        slug: str,
        stem: str,
        pkl_src: Path,
        meta: dict[str, Any],
        tone: str | None = None,
    ) -> Path:
        """
        Copy *pkl_src* (and its ``.meta.json`` sibling) into the voice's
        ``prompts/`` directory and update ``voice.json``.

        If *tone* is given (e.g. ``"neutral"``, ``"sad"``), it is indexed in
        the ``"tones"`` mapping so ``prompt_for_tone`` can retrieve it later.

        Returns the destination ``.pkl`` path.
        """
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", stem) or stem in (".", ".."):
            raise ValueError("Invalid prompt stem")
        pd = self.prompts_dir(slug)
        pd.mkdir(parents=True, exist_ok=True)

        dest_pkl  = pd / f"{stem}.pkl"
        dest_meta = pd / f"{stem}.meta.json"

        if Path(pkl_src).resolve() != dest_pkl.resolve():
            shutil.copyfile(pkl_src, dest_pkl)
        meta = dict(meta)
        if tone is not None:
            meta["tone"] = tone
        reference = self.ref_wav(slug)
        if reference.is_file():
            import hashlib
            with open(reference, "rb") as fh:
                actual_hash = hashlib.file_digest(fh, "sha256").hexdigest()
            if meta.get("reference_sha256") != actual_hash:
                raise ValueError("Prompt reference does not match current voice reference")
            prompt_reference = pd / f"{stem}.ref.wav"
            shutil.copyfile(reference, prompt_reference)
            meta["reference_wav"] = prompt_reference.name
        atomic_json(dest_meta, meta)

        data = self.load(slug)
        data["prompts"][stem] = f"prompts/{stem}.pkl"
        if tone:
            data.setdefault("tones", {})[tone] = stem
        self._save(slug, data)
        return dest_pkl

    @_locked
    def finalize_clone(
        self, slug: str, *, segment_wav: Path, ref_meta: dict[str, Any],
        stem: str, pkl_src: Path, meta: dict[str, Any], source: dict[str, Any],
        tone: str | None = None, display_name: str | None = None,
        source_clip: Path | None = None,
        generation_defaults: dict[str, Any] | None = None,
    ) -> Path:
        """Stage a complete clone under the registry lock; preserve old data on failure."""
        slug = validate_slug(slug)
        if not isinstance(meta, dict) or not meta:
            raise ValueError("Prompt metadata is required for clone registration")
        dest = self.voice_dir(slug)
        workspace = Path(tempfile.mkdtemp(prefix=".clone-", dir=self.root))
        backup = workspace / "previous"
        try:
            staged = VoiceRegistry(workspace)
            if dest.exists():
                shutil.copytree(dest, staged.voice_dir(slug))
            else:
                staged.create(slug, display_name=display_name or slug, source=source)
            if source_clip is not None:
                staged.set_source_clip(slug, source_clip)
            staged.update_ref(slug, segment_wav, ref_meta)
            staged.register_prompt(slug, stem, pkl_src, meta, tone=tone)
            if generation_defaults is not None:
                staged.update_generation_defaults(slug, **generation_defaults)
            if dest.exists():
                os.replace(dest, backup)
            try:
                os.replace(staged.voice_dir(slug), dest)
            except BaseException:
                if backup.exists():
                    os.replace(backup, dest)
                raise
        finally:
            # Retain recovery bytes if rollback itself failed.
            if not backup.exists() or dest.exists():
                shutil.rmtree(workspace)
        return self.prompts_dir(slug) / f"{stem}.pkl"

    def custom_voice_profile(self, slug: str) -> dict[str, Any] | None:
        """
        Return normalised custom-voice profile for *slug*, or ``None``
        when this voice is not configured for ``engine=custom_voice``.
        """
        data = self.load(slug)
        if infer_engine(data) != "custom_voice":
            return None
        profile = data.get("custom_voice") or {}
        if not isinstance(profile, dict):
            return None
        speaker = str(profile.get("speaker", "")).strip()
        if not speaker:
            return None
        tones = profile.get("tones") or {}
        if not isinstance(tones, dict):
            tones = {}
        return {
            "model": profile.get("model"),
            "speaker": speaker,
            "instruct_default": profile.get("instruct_default"),
            "language_default": profile.get("language_default"),
            "tones": dict(tones),
        }

    @_locked
    def register_custom_voice(
        self,
        slug: str,
        *,
        model: str,
        speaker: str,
        instruct_default: str | None = None,
        language_default: str | None = None,
        tone: str | None = None,
        tone_instruct: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """
        Create or update a named built-in CustomVoice profile.

        Safety policy:
          - Existing clone/designed voices are not overwritten.
          - Existing custom_voice entries are updated in place.
        """
        if tone and (tone_instruct is None or not tone_instruct.strip()):
            raise ValueError("Tone preset requires non-empty tone instruction.")
        if tone_instruct and not tone:
            raise ValueError("Tone instruction requires --tone.")

        if self.exists(slug):
            data = self.load(slug)
            engine = infer_engine(data)
            if engine != "custom_voice":
                has_clone_data = bool(data.get("prompts")) or data.get("ref") is not None
                source_type = str((data.get("source") or {}).get("type", "")).strip()
                if has_clone_data or source_type not in ("", "builtin"):
                    raise ValueError(
                        f"Voice '{slug}' already exists as a non-CustomVoice profile. "
                        "Use a different --voice-name."
                    )
        else:
            now = datetime.now(timezone.utc).isoformat()
            data = {
                "slug": slug,
                "display_name": display_name or slug,
                "description": description or "",
                "created_at": now,
                "source": {"type": "builtin"},
                "ref": None,
                "prompts": {},
                "tones": {},
                "engine": "custom_voice",
                "custom_voice": {},
            }

        data = self._normalise_record(data)
        data["engine"] = "custom_voice"

        if display_name is not None:
            data["display_name"] = display_name
        elif not data.get("display_name"):
            data["display_name"] = slug
        if description is not None:
            data["description"] = description

        source = data.setdefault("source", {})
        source["type"] = "builtin"
        source["speaker"] = speaker
        source["model"] = model

        profile = data.setdefault("custom_voice", {})
        profile["model"] = model
        profile["speaker"] = speaker
        if instruct_default is not None:
            profile["instruct_default"] = instruct_default
        elif "instruct_default" not in profile:
            profile["instruct_default"] = None
        if language_default is not None:
            profile["language_default"] = language_default
        elif "language_default" not in profile:
            profile["language_default"] = "English"

        tone_map = profile.setdefault("tones", {})
        if tone is not None:
            tone_map[tone] = (tone_instruct or "").strip()

        self._save(slug, data)
        return self.load(slug)

    def prompt_for_tone(self, slug: str, tone: str, **compatibility) -> Path | None:
        """
        Return the ``.pkl`` path for *tone* (e.g. ``"sad"``), or ``None``
        if that tone has not been registered for this voice.
        """
        data  = self.load(slug)
        tones = data.get("tones") or {}
        stem  = tones.get(tone)
        if not stem:
            return None
        pkl = self.prompts_dir(slug) / f"{stem}.pkl"
        return pkl if pkl.exists() and (not compatibility or self.compatible_prompt(pkl, **compatibility)) else None

    def list_tones(self, slug: str) -> dict[str, str]:
        """Return the ``{tone_name: stem}`` mapping for this voice (may be empty)."""
        data = self.load(slug)
        return dict(data.get("tones") or {})

    def list_custom_tones(self, slug: str) -> dict[str, str]:
        """
        Return ``{tone_name: instruct}`` for custom-voice profiles.
        Returns an empty dict for non-custom voices.
        """
        profile = self.custom_voice_profile(slug)
        if not profile:
            return {}
        tones = profile.get("tones") or {}
        if not isinstance(tones, dict):
            return {}
        return {
            str(name): str(instruct)
            for name, instruct in tones.items()
        }

    def generation_defaults(self, slug: str) -> dict[str, Any]:
        """
        Return optional per-voice generation defaults.
        """
        data = self.load(slug)
        defaults = data.get("generation_defaults")
        if not isinstance(defaults, dict):
            return {}
        return dict(defaults)

    @_locked
    def update_generation_defaults(
        self,
        slug: str,
        *,
        profile: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        max_new_tokens_policy: str | None = None,
    ) -> dict[str, Any]:
        """
        Upsert per-voice generation defaults in ``voice.json``.
        """
        data = self.load(slug)
        defaults = data.setdefault("generation_defaults", {})
        if not isinstance(defaults, dict):
            defaults = {}
            data["generation_defaults"] = defaults

        if profile is not None:
            defaults["profile"] = str(profile)
        if temperature is not None:
            defaults["temperature"] = float(temperature)
        if top_p is not None:
            defaults["top_p"] = float(top_p)
        if repetition_penalty is not None:
            defaults["repetition_penalty"] = float(repetition_penalty)
        if max_new_tokens_policy is not None:
            defaults["max_new_tokens_policy"] = str(max_new_tokens_policy)

        self._save(slug, data)
        return self.generation_defaults(slug)

    # ── management ──────────────────────────────────────────────────────────────

    @_locked
    def rename(self, old_slug: str, new_slug: str) -> dict[str, Any]:
        """
        Rename a voice: move its directory and update voice.json's slug field.
        Returns the updated voice record.
        """
        if not self.exists(old_slug):
            raise KeyError(f"Voice '{old_slug}' not found.")
        new_slug = validate_slug(new_slug)
        if self.exists(new_slug):
            raise ValueError(f"Voice '{new_slug}' already exists.")
        old_dir = self.voice_dir(old_slug)
        new_dir = self.voice_dir(new_slug)
        old_dir.rename(new_dir)
        # Update slug (and display_name when it mirrored the slug)
        data = self.load(new_slug)
        data["slug"] = new_slug
        if data.get("display_name") == old_slug:
            data["display_name"] = new_slug
        self._save(new_slug, data)
        return data

    @_locked
    def delete(self, slug: str) -> None:
        """Permanently delete a named voice and all its files."""
        vd = self.voice_dir(slug)
        if not vd.exists():
            raise KeyError(f"Voice '{slug}' not found.")
        shutil.rmtree(vd)

    @_locked
    def export_zip(self, slug: str, dest: Path) -> Path:
        """
        Pack the voice directory into a zip archive at *dest*.
        The archive preserves paths relative to ``self.root`` so that
        ``import_zip`` can unpack it correctly on any machine.
        Returns *dest*.
        """
        slug = validate_slug(slug)
        vd = self.voice_dir(slug)
        data = self.load(slug)
        data["prompts"] = {}
        data["tones"] = {}
        data["schema_version"] = 2
        data["exchange_format"] = "wav-json-v1"
        source = data.get("source", {})
        data["source"] = {k: v for k, v in source.items() if k not in ("path", "url", "video_id")}
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix=".voice-export-", dir=dest.parent)
        os.close(fd)
        try:
            with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{slug}/voice.json", json.dumps(data, indent=2, allow_nan=False))
                for filename in ("ref.wav", "source_clip.wav"):
                    path = vd / filename
                    if path.is_symlink():
                        raise ValueError("Cannot export symlinked voice files")
                    if path.is_file():
                        zf.write(path, f"{slug}/{filename}")
            os.replace(name, dest)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return dest

    @_locked
    def import_zip(self, zip_path: Path, force: bool = False) -> str:
        """Validate a bounded WAV/JSON exchange, stage it, then promote.

        Pickle caches are never exchange assets. Forced replacement retains the
        original directory under .backups, including its trusted local caches.
        """
        with zipfile.ZipFile(zip_path) as zf:
            entries = zf.infolist()
            if not entries or len(entries) > 64:
                raise ValueError("Empty archive or too many entries")
            seen = set()
            slug = None
            total = 0
            files = {}
            for entry in entries:
                name = entry.filename
                parts = PurePosixPath(name).parts
                if ("\\" in name or name.startswith("/") or not parts
                        or name.rstrip("/") != "/".join(parts)
                        or any(p in (".", "..") for p in name.rstrip("/").split("/"))):
                    raise ValueError("Unsafe archive path")
                root = validate_slug(parts[0])
                if root != parts[0] or (slug is not None and root != slug):
                    raise ValueError("Archive must contain exactly one voice root")
                slug = root
                if name in seen:
                    raise ValueError("Duplicate archive path")
                seen.add(name)
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise ValueError("Archive links and special files are forbidden")
                if entry.flag_bits & 1:
                    raise ValueError("Encrypted archive entries are forbidden")
                if entry.is_dir():
                    if len(parts) != 1:
                        raise ValueError("Unexpected archive directory")
                    continue
                if len(parts) != 2 or parts[1] not in ("voice.json", "ref.wav", "source_clip.wav"):
                    raise ValueError("Only portable WAV/JSON assets may be imported; pickle caches forbidden")
                total += entry.file_size
                if total > 256 * 1024 * 1024 or entry.file_size > 128 * 1024 * 1024:
                    raise ValueError("Archive exceeds decompressed size limit")
                if parts[1] == "voice.json" and entry.file_size > 1024 * 1024:
                    raise ValueError("Voice metadata too large")
                files[parts[1]] = entry
            if "voice.json" not in files:
                raise ValueError("Archive requires voice.json")
            data = json.loads(zf.read(files["voice.json"]))
            if not isinstance(data, dict) or data.get("slug") != slug:
                raise ValueError("Archive metadata slug mismatch")
            data = self._normalise_record(data)
            data["prompts"] = {}
            data["tones"] = {}
            data["schema_version"] = 2
            dest = self.voice_dir(slug)
            if dest.exists() and not force:
                raise ValueError(f"Voice '{slug}' already exists. Use --force to overwrite.")
            stage = Path(tempfile.mkdtemp(prefix=".import-", dir=self.root))
            backup = None
            try:
                for filename, entry in files.items():
                    if filename == "voice.json":
                        continue
                    with zf.open(entry) as src, open(stage / filename, "wb") as out:
                        shutil.copyfileobj(src, out, length=1024 * 1024)
                    import wave
                    with wave.open(str(stage / filename), "rb") as wav:
                        if wav.getnchannels() not in (1, 2) or wav.getframerate() <= 0:
                            raise ValueError("Invalid reference WAV")
                atomic_json(stage / "voice.json", data)
                if dest.exists():
                    backups = self.root / ".backups"
                    backups.mkdir(exist_ok=True)
                    backup = backups / (slug + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
                    os.replace(dest, backup)
                try:
                    os.replace(stage, dest)
                except BaseException:
                    if backup is not None:
                        os.replace(backup, dest)
                    raise
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        return slug

    @_locked
    def migrate(self, slug: str, *, dry_run: bool = True) -> dict[str, Any]:
        """Explicit additive v2 migration; preserve original bytes before writing."""
        path = self.voice_json(slug)
        data = self.load(slug)
        previous = data.get("schema_version", 1)
        if previous == 2:
            return {"slug": slug, "changed": False, "dry_run": dry_run}
        if not dry_run:
            backup = path.with_name("voice.v1." + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".backup.json")
            shutil.copyfile(path, backup)
            self._save(slug, data)
        return {"slug": slug, "changed": True, "dry_run": dry_run, "from": previous, "to": 2}
