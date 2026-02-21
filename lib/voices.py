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
    slug = slug.strip().lower()
    if not SLUG_RE.match(slug):
        raise ValueError(
            f"Invalid voice name {slug!r}. "
            "Use only lowercase letters, digits, and hyphens "
            "(e.g. 'david-attenborough', 'my-narrator-v2')."
        )
    return slug


class VoiceRegistry:
    """
    Thin wrapper around ``/cache/voices/`` that provides create/read/update
    operations for named voices.

    All operations are single-file atomic (json.dump over the full voice.json),
    which is safe for our single-writer / multi-reader use pattern.
    """

    def __init__(self, cache: Path) -> None:
        self.root = Path(cache) / "voices"

    # ── path helpers ────────────────────────────────────────────────────────────

    def voice_dir(self, slug: str) -> Path:
        return self.root / slug

    def voice_json(self, slug: str) -> Path:
        return self.root / slug / "voice.json"

    def source_clip(self, slug: str) -> Path:
        return self.root / slug / "source_clip.wav"

    def ref_wav(self, slug: str) -> Path:
        return self.root / slug / "ref.wav"

    def prompts_dir(self, slug: str) -> Path:
        return self.root / slug / "prompts"

    # ── read ────────────────────────────────────────────────────────────────────

    def exists(self, slug: str) -> bool:
        return self.voice_json(slug).exists()

    def _normalise_record(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Normalise optional/additive fields for backward compatibility.

        Does not write to disk; callers may persist explicitly after mutation.
        """
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

    def best_prompt(self, slug: str) -> Path | None:
        """
        Return the most recently written ``.pkl`` in this voice's prompts/,
        or ``None`` if no prompts exist yet.
        """
        pd = self.prompts_dir(slug)
        if not pd.exists():
            return None
        pkls = sorted(pd.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
        return pkls[0] if pkls else None

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
        with open(vd / "voice.json", "w") as fh:
            json.dump(data, fh, indent=2)

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
        pd = self.prompts_dir(slug)
        pd.mkdir(parents=True, exist_ok=True)

        dest_pkl  = pd / f"{stem}.pkl"
        dest_meta = pd / f"{stem}.meta.json"

        shutil.copyfile(pkl_src, dest_pkl)
        with open(dest_meta, "w") as fh:
            json.dump(meta, fh, indent=2)

        data = self.load(slug)
        data["prompts"][stem] = f"prompts/{stem}.pkl"
        if tone:
            data.setdefault("tones", {})[tone] = stem
        self._save(slug, data)
        return dest_pkl

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

    def prompt_for_tone(self, slug: str, tone: str) -> Path | None:
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
        return pkl if pkl.exists() else None

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

    def delete(self, slug: str) -> None:
        """Permanently delete a named voice and all its files."""
        vd = self.voice_dir(slug)
        if not vd.exists():
            raise KeyError(f"Voice '{slug}' not found.")
        shutil.rmtree(vd)

    def export_zip(self, slug: str, dest: Path) -> Path:
        """
        Pack the voice directory into a zip archive at *dest*.
        The archive preserves paths relative to ``self.root`` so that
        ``import_zip`` can unpack it correctly on any machine.
        Returns *dest*.
        """
        import zipfile
        vd = self.voice_dir(slug)
        if not vd.exists():
            raise KeyError(f"Voice '{slug}' not found.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(vd.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(self.root))
        return dest

    def import_zip(self, zip_path: Path, force: bool = False) -> str:
        """
        Unpack a voice zip created by ``export_zip``.
        The root directory name inside the zip is used as the slug.
        Returns the imported slug.
        Raises ``ValueError`` if the voice already exists and *force* is False.
        """
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
        if not names:
            raise ValueError("Empty zip file.")
        slug = Path(names[0]).parts[0]
        validate_slug(slug)
        dest = self.voice_dir(slug)
        if dest.exists():
            if not force:
                raise ValueError(
                    f"Voice '{slug}' already exists. Use --force to overwrite."
                )
            shutil.rmtree(dest)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(self.root)
        return slug
