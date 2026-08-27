"""Structure + contract tests for the blinded foley review PAGE_TEMPLATE.

Static checks read the generator source. DOM checks execute `build()` in a
stub browser via tests/run_blind_dom.mjs so radios/tables match the live page.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BUILDER = HERE.parent / "build_blind_review.py"
RUNNER = HERE / "run_blind_dom.mjs"
RUN_DIR = REPO / "work" / "sfx-pipeline" / "20260826-poc"
ASSIGNMENT = RUN_DIR / "review" / "blind" / "assignment.json"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_blind_review", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def render_page(mod, assignment: dict, run_id: str = "test-run") -> str:
    blob, _n = mod.encode_config(assignment, run_id)
    page = mod.PAGE_TEMPLATE.replace("__CFG_B64__", blob)
    page = page.replace("__DIMS__", json.dumps(mod.CLIP_RATING_DIMENSIONS))
    page = page.replace("__DIM_LABELS__", json.dumps(mod.DIMENSION_LABELS))
    if "__DIM_HELP__" in page:
        page = page.replace("__DIM_HELP__", json.dumps(getattr(mod, "DIMENSION_HELP", {})))
    return page


def fixture_assignment() -> dict:
    return {
        "schema_version": 1,
        "kind": "blind-review-assignment",
        "run_dir": "/tmp/fake",
        "generated_utc": "2026-01-01T00:00:00Z",
        "seed": 1,
        "clips": {
            "clip-a": {
                "map": {"A": "sys-one", "B": "sys-two"},
                "events": [
                    {
                        "id": "ev_001",
                        "label": "footsteps on gravel",
                        "time_s": 1.109,
                        "tier": "mirelo-fallback",
                    },
                    {
                        "id": "gt_003",
                        "label": "a shutter click",
                        "time_s": 65.0,
                        "tier": "direct",
                    },
                ],
            }
        },
    }


def run_dom(html: str, tmp_path: Path) -> dict:
    page = tmp_path / "index.html"
    page.write_text(html)
    proc = subprocess.run(
        ["node", str(RUNNER), str(page)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


@pytest.fixture(scope="module")
def mod():
    return load_builder()


@pytest.fixture(scope="module")
def fixture_html(mod):
    return render_page(mod, fixture_assignment())


@pytest.fixture(scope="module")
def real_html(mod):
    if not ASSIGNMENT.is_file():
        pytest.skip("real assignment.json missing")
    assignment = json.loads(ASSIGNMENT.read_text())
    return render_page(mod, assignment, run_id=RUN_DIR.name)


@pytest.fixture
def fixture_dom(fixture_html, tmp_path):
    return run_dom(fixture_html, tmp_path)


@pytest.fixture
def real_dom(real_html, tmp_path):
    return run_dom(real_html, tmp_path)


def test_general_media_notes_count_real(real_dom):
    assert real_dom["gnoteCount"] == 10
    root = real_dom["root"]
    for clip in ("man", "ocean-long", "ocean-short", "walk1", "walk2"):
        for label in ("A", "B"):
            assert f'data-clip="{clip}"' in root
            assert re.search(
                rf'data-clip="{clip}"[^>]*data-id="general"[^>]*data-label="{label}"'
                rf'[^>]*data-field="text"|'
                rf'data-clip="{clip}"[^>]*data-label="{label}"[^>]*data-id="general"',
                root,
            )


def test_radio_count_real_is_650(real_dom):
    # 600 pre-existing (6 dims + 1 keep per clip/system, pref+keep per event/system)
    # + 50 from the whole-clip overall-preference scales (5 clips x 2 systems x 5)
    assert real_dom["radioCount"] == 650


def test_one_shared_event_table_per_clip(fixture_dom):
    assert fixture_dom["eventTableCount"] == 1
    root = fixture_dom["root"]
    a_pref = len(re.findall(
        r'<input[^>]*type="radio"[^>]*data-scope="event"[^>]*data-label="A"[^>]*data-field="pref"',
        root,
    ))
    b_pref = len(re.findall(
        r'<input[^>]*type="radio"[^>]*data-scope="event"[^>]*data-label="B"[^>]*data-field="pref"',
        root,
    ))
    if a_pref == 0:
        a_pref = len(re.findall(
            r'<input[^>]*data-scope="event"[^>]*data-label="A"[^>]*data-field="pref"',
            root,
        ))
        b_pref = len(re.findall(
            r'<input[^>]*data-scope="event"[^>]*data-label="B"[^>]*data-field="pref"',
            root,
        ))
    assert a_pref == 10
    assert b_pref == 10


def test_event_timestamp_is_readable(fixture_dom):
    root = fixture_dom["root"]
    assert "1.11s" not in root
    assert re.search(r"0:01\.\d", root)
    assert re.search(r"1:05\.0", root)


def test_event_id_is_secondary(fixture_dom):
    root = fixture_dom["root"]
    assert "ev_001" in root
    assert "footsteps on gravel" in root
    assert re.search(r'class="[^"]*ev-id[^"]*"', root)


def test_tier_is_explained(fixture_html, fixture_dom):
    blob = fixture_html + fixture_dom["root"]
    lowered = blob.lower()
    assert "auto-proposed" in lowered or "not hand-curated" in lowered
    assert "mirelo-fallback" in blob


def test_scale_endpoints_labelled(fixture_html, fixture_dom):
    blob = (fixture_html + fixture_dom["root"]).lower()
    assert "unusable" in blob
    assert "strongly prefer" in blob
    assert "poor" in blob
    assert "excellent" in blob


def test_keep_replace_not_welded_to_sync_row(fixture_dom):
    root = fixture_dom["root"]
    assert re.search(r'data-row="clip-keep"|class="[^"]*clip-keep', root)
    assert re.search(r'data-row="clip-notes"|class="[^"]*clip-notes', root)
    rows = re.findall(r"<tr\b[^>]*>.*?</tr>", root, flags=re.S)
    sync_and_keep = [
        r for r in rows
        if 'data-id="sync"' in r and 'data-id="overall"' in r and 'data-field="keep"' in r
    ]
    assert sync_and_keep == []


def test_keep_phrasing_is_ship_without_swapping(fixture_html, fixture_dom):
    blob = fixture_html + fixture_dom["root"]
    assert re.search(r"ship this sound without (manually )?swapp", blob, re.I)


def test_dimension_definitions_present(fixture_html, fixture_dom):
    text = (fixture_html + fixture_dom["root"]).lower()
    assert "timing errors" in text
    assert "right kind of sound" in text or "kind of sound for the action" in text
    assert "physically plausible" in text
    assert "fidelity" in text or "artifact-free" in text
    assert "sonic identity" in text or "palette" in text
    assert "glitch" in text or "truncation" in text or "pumping" in text


def test_general_notes_are_labelled(fixture_dom):
    assert "General media notes" in fixture_dom["root"]


def test_notes_textareas_default_two_rows(fixture_dom):
    assert fixture_dom["textareaCount"] == fixture_dom["rows2"]
    assert fixture_dom["textareaCount"] >= 2


def test_event_rows_drive_videos(fixture_html, fixture_dom):
    root = fixture_dom["root"]
    src = fixture_html
    assert re.search(r'data-time=', root)
    assert re.search(r'class="[^"]*play-ev|data-action="inspect"', root)
    assert "inspectEvent" in src or "function inspect" in src
    assert "0.75" in src
    videos = re.findall(r"<video\b[^>]*>", root)
    assert len(videos) == 2
    assert any('data-label="A"' in v for v in videos)
    assert any('data-label="B"' in v for v in videos)
    assert any('data-clip="clip-a"' in v for v in videos)


def test_progress_and_saved_affordance(fixture_html, fixture_dom):
    assert 'id="prog"' in fixture_html
    assert 'id="saved"' in fixture_html
    assert "clip-progress" in fixture_dom["root"] or "data-clip-progress" in fixture_dom["root"]
    assert re.search(r"\d+/\d+", fixture_dom["progressText"])


def test_submit_and_download_survive(mod):
    src = mod.PAGE_TEMPLATE
    assert 'id="submit"' in src
    assert 'id="download"' in src
    assert "/save" in src
    assert "blind-annotator" in src
    assert "blind-ratings:" in src


def test_completeness_math_equivalent(mod):
    src = mod.PAGE_TEMPLATE
    assert "total += DIMS.length * nSys + 2 * nSys" in src
    assert 'getRating(clip,"clip","overall",lbl)["keep"]' in src
    assert 'getRating(clip,"clip","overall",lbl)["pref"]' in src
    assert 'getRating(clip,"event",ev.id,lbl)["pref"]' in src
    assert 'getRating(clip,"event",ev.id,lbl)["keep"]' in src


def test_payload_key_format_unchanged(mod):
    src = mod.PAGE_TEMPLATE
    assert r'clip+"\u0000"+scope+"\u0000"+id+"\u0000"+label' in src
    assert r'k.replace(/\u0000/g,"|")' in src


def test_no_framework_no_cdn_no_emoji_in_template(mod):
    src = mod.PAGE_TEMPLATE
    assert "cdn." not in src.lower()
    assert "unpkg" not in src
    assert "googleapis" not in src
    assert "react" not in src.lower()
    for ch in ("😀", "⭐", "✅", "▶", "⚠️"):
        assert ch not in src, f"emoji/symbol {ch!r} found"


def test_radio_groups_have_names(fixture_dom):
    assert fixture_dom["radioCount"] > 0
    assert fixture_dom["namedRadios"] == fixture_dom["radioCount"]


def test_self_contained_single_file(fixture_html):
    assert "<link " not in fixture_html
    assert "<script src=" not in fixture_html
    assert 'id="cfg"' in fixture_html


def test_inspect_window_uses_preroll(fixture_dom):
    assert fixture_dom["inspect"]["start"] == pytest.approx(1.109 - 0.75)
    assert fixture_dom["inspect"]["end"] - fixture_dom["inspect"]["start"] == pytest.approx(2.5)
    assert fixture_dom["formatTime"] == "0:01.1"
    assert fixture_dom["formatTimeLong"] == "1:05.0"


def test_payload_keys_use_pipe_and_numeric_pref(fixture_dom):
    assert fixture_dom["prefKey"] == "clip-a|event|ev_001|A"
    assert fixture_dom["prefValue"]["pref"] == 3
    assert type(fixture_dom["prefValue"]["pref"]) is int
    assert "|" in fixture_dom["prefKey"]
    assert "\x00" not in fixture_dom["prefKey"]


def _required_control_tuples(assignment: dict, mod) -> set[tuple[str, str, str, str, str]]:
    """Every (clip, scope, id, label, field) missingItems() can emit with an empty rating store."""
    out: set[tuple[str, str, str, str, str]] = set()
    dims = mod.CLIP_RATING_DIMENSIONS
    for clip, info in assignment["clips"].items():
        labels = ["A", "B"][: len(info["map"])]
        for label in labels:
            for dim in dims:
                out.add((clip, "clip", dim, label, dim))
            out.add((clip, "clip", "overall", label, "keep"))
            out.add((clip, "clip", "overall", label, "pref"))
        for ev in info["events"]:
            for label in labels:
                out.add((clip, "event", ev["id"], label, "pref"))
                out.add((clip, "event", ev["id"], label, "keep"))
    return out


def _rendered_control_tuples(root: str) -> set[tuple[str, str, str, str, str]]:
    out: set[tuple[str, str, str, str, str]] = set()
    for m in re.finditer(r"<(?:input|textarea)\b[^>]*>", root):
        tag = m.group(0)
        attrs = dict(re.findall(r'data-(clip|scope|id|label|field)="([^"]*)"', tag))
        if len(attrs) == 5:
            out.add((attrs["clip"], attrs["scope"], attrs["id"], attrs["label"], attrs["field"]))
    return out


def test_every_missing_item_maps_to_a_rendered_control(fixture_html, fixture_dom, mod):
    """Regression: whole-clip 'Preference' pills once pointed at controls that were never
    rendered, so jump clicks silently no-opped and 160/160 was unreachable."""
    assignment = fixture_assignment()
    required = _required_control_tuples(assignment, mod)
    rendered = _rendered_control_tuples(fixture_dom["root"])
    missing = required - rendered
    assert not missing, f"jump/rating targets without a rendered control: {sorted(missing)}"


def test_every_missing_item_maps_to_a_rendered_control_real(real_html, real_dom, mod):
    if not ASSIGNMENT.is_file():
        pytest.skip("real assignment.json missing")
    assignment = json.loads(ASSIGNMENT.read_text())
    required = _required_control_tuples(assignment, mod)
    rendered = _rendered_control_tuples(real_dom["root"])
    missing = required - rendered
    assert not missing, f"jump/rating targets without a rendered control: {sorted(missing)}"
