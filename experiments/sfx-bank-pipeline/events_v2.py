#!/usr/bin/env python3
"""Canonical temporal-type-aware event schema helpers (phase-2 event model v2).

Implements CTO report §5: downstream stops pretending every event is
`label + timestamp`. Events gain (additively — v1 readers keep working):

    temporal_type : "impulse" | "continuous" | "repetitive" | "bed" | "hybrid"
                    | None   (detector-only events without a semantic class yet)
    subclass      : optional refinement string
    sequence_id   : groups members of one repetitive family (same clip+label)
    source        : {"object","material","surface"} | present-as-None fields
    physical      : {"intensity","velocity","size"}
    visual        : {"x","y","offscreen"}

The classifier is a documented keyword heuristic over retrieval labels; richer
inference belongs to later vision work, not here. Every upgraded dict also gets
"schema_event_v2": true so tooling can branch cheaply.
"""

from __future__ import annotations

from typing import Any

TEMPORAL_TYPES = ("impulse", "continuous", "repetitive", "bed", "hybrid")

#: (pattern substring → type) checked lower-case, first hit wins.
HEURISTICS: tuple[tuple[str, str], ...] = (
    ("wipers", "repetitive"),
    ("hammering", "repetitive"),
    ("footsteps", "repetitive"),
    ("walking", "repetitive"),
    ("ambience", "bed"),
    ("room tone", "bed"),
    ("wind", "bed"),
    ("rain", "bed"),
    ("surf", "bed"),
    ("ocean", "bed"),
    ("waves", "bed"),
    ("door", "hybrid"),
    ("creak", "hybrid"),
    ("latch", "hybrid"),
    ("scrape", "continuous"),
    ("scrubbing", "continuous"),
    ("drag", "continuous"),
    ("pour", "continuous"),
    ("zipper", "continuous"),
    ("whoosh", "continuous"),
    ("hiss", "continuous"),
    ("engine", "continuous"),
)

DEFAULT_TYPE = "impulse"


def infer_temporal_type(label: str | None) -> str | None:
    if not label:
        return None
    low = label.lower()
    for pattern, ttype in HEURISTICS:
        if pattern in low:
            return ttype
    return DEFAULT_TYPE


def upgrade_event_v1(event: dict[str, Any], label: str | None = None,
                     clip: str | None = None,
                     sequence_id: str | None = None) -> dict[str, Any]:
    """Return a NEW event dict enriched additively with v2 fields.

    Original keys are never mutated or removed.
    """
    out = dict(event)
    semantic_label = label if label is not None else out.get("label")
    inferred = infer_temporal_type(semantic_label)
    if "temporal_type" not in out:
        out["temporal_type"] = inferred          # may be None → unclassified detector event
    out.setdefault("subclass", None)
    out.setdefault("sequence_id", sequence_id)
    out.setdefault("source", {"object": None, "material": None, "surface": None})
    out.setdefault("physical", {"intensity": None, "velocity": None, "size": None})
    out.setdefault("visual", {"x": None, "y": None, "offscreen": False})
    out["schema_event_v2"] = True
    return out


def apply_sequence_ids(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group same-clip repetitive events sharing a classification label.

    Caller supplies events of ONE clip; label collisions across clips are
    impossible because ids embed the clip name upstream.
    """
    labels = [str(e.get("label") or "") for e in events]
    types = [infer_temporal_type(lbl) for lbl in labels]
    counts: dict[tuple[str, str], int] = {}
    for lbl, tt in zip(labels, types):
        key = (lbl, tt or "")
        counts[key] = counts.get(key, 0) + 1
    return [
        upgrade_event_v1(ev, label=lbl,
                         sequence_id=lbl if tt == "repetitive" and counts[(lbl, tt or "")] > 1 else None)
        for ev, lbl, tt in zip(events, labels, types)
    ]
