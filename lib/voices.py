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

    def load(self, slug: str) -> dict[str, Any]:
        vj = self.voice_json(slug)
        if not vj.exists():
            raise KeyError(
                f"Voice '{slug}' not found in registry. "
                f"Looked in: {vj}"
            )
        with open(vj) as fh:
            return json.load(fh)

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
                    d = json.load(fh)
            except Exception:
                continue
            pd = vj.parent / "prompts"
            count = len(list(pd.glob("*.pkl"))) if pd.exists() else 0
            d["_prompt_count"] = count
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
            shutil.copy2(wav_src, dest)
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
            shutil.copy2(segment_wav, dest)
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

        shutil.copy2(pkl_src, dest_pkl)
        with open(dest_meta, "w") as fh:
            json.dump(meta, fh, indent=2)

        data = self.load(slug)
        data["prompts"][stem] = f"prompts/{stem}.pkl"
        if tone:
            data.setdefault("tones", {})[tone] = stem
        self._save(slug, data)
        return dest_pkl

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
