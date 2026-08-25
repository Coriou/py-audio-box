"""Tests for Phase-2 T2 forensic windows (structure + selection logic).

Pure-synthetic: reuses the fake trace builders from test_render_markers;
no video files. Plot rendering needs matplotlib and is import-gated so
the suite stays green on hosts without it (docker runs everything).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import auto_sfx.forensics as forensics_mod  # noqa: E402
from auto_sfx.cache import TRACE_FORMAT, TRACE_FORMAT_VERSION  # noqa: E402
from auto_sfx.contacts import default_config  # noqa: E402
from auto_sfx.forensics import (  # noqa: E402
    GT_MATCH_HORIZON_S,
    ContactTraceIndex,
    build_forensics,
    build_window,
    load_contact_trace,
    select_windows,
)
from auto_sfx.gt import GTContact  # noqa: E402
from auto_sfx.types import PoseTrace  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_render_markers import make_event, make_meta, make_pf, make_trace  # noqa: E402


def _gt(i: int, foot: str) -> dict:
    base = i * 1_000_000
    return {
        "foot": foot,
        "preferred_s": base / 1e6 + 0.033333,
        "earliest_s": base / 1e6,
        "latest_s": base / 1e6 + 0.066667,
    }


def _events_doc_events(trace: PoseTrace):
    return [
        {
            "id": "evt_001",
            "time_s": trace.frames[2].t_s,
            "source_pts_us": trace.frames[2].source_pts_us,
            "earliest_s": trace.frames[2].t_s,
            "latest_s": trace.frames[2].t_s + 0.033,
            "foot": "left",
        },
    ]


class TestSelectWindows:
    def test_predicted_and_missed_gt_split(self):
        trace = make_trace(n=12, fps=12.0)
        events = _events_doc_events(trace)
        gt = [_gt(2, "left"), _gt(8, "right")]   # 2.033s and 8.033s
        # evt_001 at ~0.167s matches nothing; gt@2.033 has no pred either.
        windows = select_windows(events, gt, None)
        kinds = [k for k, _c, _l in windows]
        assert kinds.count("predicted") == 1
        assert kinds.count("missed_gt") == 2

    def test_matched_gt_gets_no_extra_window(self):
        trace = make_trace(n=12, fps=12.0)
        events = _events_doc_events(trace)
        near = {
            "foot": "left",
            "preferred_s": events[0]["time_s"] + 0.01,
            "earliest_s": events[0]["time_s"],
            "latest_s": events[0]["time_s"] + 0.02,
        }
        windows = select_windows(events, [near], None)
        assert [k for k, _c, _l in windows] == ["predicted"]

    def test_event_filter_selects_one(self):
        trace = make_trace(n=12, fps=12.0)
        events = _events_doc_events(trace)
        windows = select_windows(events, None, "evt_001")
        assert len(windows) == 1 and windows[0][0] == "predicted"
        with pytest.raises(ValueError):
            select_windows(events, None, "evt_999")


class TestBuildWindow:
    def test_schema_shape_and_gt_overlay(self):
        trace = make_trace(n=24, fps=12.0)
        events = _events_doc_events(trace)
        gt = [{
            "foot": "left",
            "preferred_s": events[0]["time_s"] + 1.0 / 30.0,
            "earliest_s": events[0]["time_s"],
            "latest_s": events[0]["time_s"] + 2.0 / 30.0,
        }]
        win = build_window(
            trace, events, gt, events[0]["source_pts_us"],
            kind="predicted", half_s=0.5, method="hybrid",
        )
        assert win["schema_version"] == 1
        assert win["kind"] == "predicted"
        assert win["center_pts_us"] == events[0]["source_pts_us"]
        assert win["gt"] is not None and win["gt"]["foot"] == "left"
        s = win["series"]
        for key in ("t_s", "pts_us", "heel_xy", "foot_center_xy", "pelvis_xy",
                    "body_scale_px", "conf_raw", "vel", "accel",
                    "support_band_score", "state", "thresholds"):
            assert key in s, key
        assert win["transitions"] == []
        assert len(s["t_s"]) == len(s["pts_us"]) > 0
        # conf_smoothed mirrors conf_raw by construction (no smoothing here)
        assert s["conf_smoothed"] == s["conf_raw"]

    def test_no_json_nans(self):
        trace = make_trace(n=24, fps=12.0)
        events = _events_doc_events(trace)
        win = build_window(
            trace, events, None, events[0]["source_pts_us"],
            kind="predicted", half_s=0.4,
        )
        text = json.dumps(win)          # NaN would serialize as NaN (invalid)
        assert "NaN" not in text and "Infinity" not in text


# ---------------------------------------------------------------------------
# Synthetic contact-trace.jsonl fixtures (exact machine internals)
# ---------------------------------------------------------------------------

_FPS = 12.0


def _contact_trace_rows(trace: PoseTrace) -> list[dict]:
    """Header + segment + velocity frame rows + noise rows the reader
    must ignore. Machine states planted on the LEFT foot at frames 4-8
    with transitions at 5 and 7, plus one transition at 15 that must NOT
    leak into a window around frame 2."""
    header = {
        "kind": "header", "method": "hybrid",
        "backend": "mediapipe-lite", "scale_px": 95.3,
        "config": {"speed_drop_frac": 0.42, "confirm_frames": 3,
                   "min_step_interval_s": 0.25},
        "video": {"path": "synthetic"},
    }
    rows: list[dict] = [{"kind": "segment", "index": 0, "start_frame": 0,
                         "end_frame": len(trace.frames) - 1}]
    plan = {
        4: ("UNKNOWN", "SWING", None),
        5: ("SWING", "CANDIDATE", {"from": "SWING", "to": "CANDIDATE"}),
        6: ("CANDIDATE", "CANDIDATE", None),
        7: ("CANDIDATE", "STANCE", {"from": "CANDIDATE", "to": "STANCE"}),
        8: ("STANCE", "STANCE", None),
        15: ("SWING", "CANDIDATE", {"from": "SWING", "to": "CANDIDATE"}),
    }
    for fi, (before, after, tr) in plan.items():
        fr = trace.frames[fi]
        rows.append({
            "kind": "frame", "side": "left", "frame": fi,
            "t_s": round(fr.t_s, 6), "pts_us": fr.source_pts_us,
            "state_before": before, "state_after": after, "transition": tr,
            "spd_bs_s": 1.2345, "vx_bs_s": 0.1, "vy_bs_s": 1.23,
            "thr_bs_s": 0.168, "swing_peak_bs_s": 2.5,
            "band_h_px": 19.06, "ref_x_px": 30.0, "ref_y_px": 40.0,
            "heel_vis": 0.9, "ankle_vis": 0.8, "toe_vis": 0.7,
            "run_start_frame": 4, "relaxed_run": False,
        })
    # guard/candidate/pelvis rows carry no machine state and must be skipped
    rows.append({"kind": "guard", "side": "left", "frame": 8,
                 "decision": "confirm", "reasons": ["ok"], "n_frames": 3})
    rows.append({"kind": "candidate", "side": "left", "frame": 7,
                 "t_s": round(7 / _FPS, 6), "score": 0.9,
                 "relaxed_run": False, "evidence": {}, "provenance": {}})
    rows.append({"kind": "frame", "side": "right", "frame": 5,
                 "t_s": round(5 / _FPS, 6), "pts_us": int(5 / _FPS * 1e6),
                 "method": "pelvis", "rel_offset_raw_px": 1.0,
                 "rel_offset_smoothed_px": 0.9, "smooth_window_s": 0.25})
    return [header] + rows


def _write_contact_trace(run_dir: Path, trace: PoseTrace) -> Path:
    path = Path(run_dir) / "contact-trace.jsonl"
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n"
                for r in _contact_trace_rows(trace)),
        "utf-8",
    )
    return path


class TestContactTraceLoader:
    def test_absent_file_returns_none_pair(self, tmp_path):
        assert load_contact_trace(tmp_path) == (None, None)

    def test_rows_exclude_header(self, tmp_path):
        trace = make_trace(n=24, fps=_FPS)
        _write_contact_trace(tmp_path, trace)
        rows, header = load_contact_trace(tmp_path)
        assert header["kind"] == "header"
        assert header["config"]["speed_drop_frac"] == pytest.approx(0.42)
        assert all(r.get("kind") != "header" for r in rows)


class TestExactTraceWindow:
    def _win(self, tmp_path, trace, events):
        _write_contact_trace(tmp_path, trace)
        rows, header = load_contact_trace(tmp_path)
        return build_window(
            trace, events, None, events[0]["source_pts_us"],
            kind="predicted", half_s=0.5,
            trace_index=ContactTraceIndex.from_rows(rows, header),
            gt_record={
                "gt_id": "gt#000", "foot": "left",
                "preferred_s": events[0]["time_s"] + 0.01,
                "earliest_s": events[0]["time_s"],
                "latest_s": events[0]["time_s"] + 0.02,
                "visibility": "clear", "viewpoint": "side",
            },
            assoc_pair={"err_s": -0.01, "err_frames": 0,
                        "inside_gt_window": True, "side_ok": True},
        )

    def test_exact_states_transitions_thresholds(self, tmp_path):
        trace = make_trace(n=24, fps=_FPS)
        win = self._win(tmp_path, trace, _events_doc_events(trace))
        assert win["trace"] == {"source": "contact-trace.jsonl", "exact": True}
        s = win["series"]
        # window covers frames 0..8: untraced 0..3 stay None, never invented
        states = s["state"]["left"]
        assert states[:4] == [None] * 4
        assert [v for v in states if v is not None] == [
            "SWING", "CANDIDATE", "CANDIDATE", "STANCE"]
        assert all(v is None for v in s["state"]["right"])
        # real transitions inside the window only (frame 15 stays out)
        assert [(t["frame"], t["from"], t["to"])
                for t in win["transitions"]] == [
            (5, "SWING", "CANDIDATE"), (7, "CANDIDATE", "STANCE")]
        assert win["transitions"][0]["pts_us"] == \
            trace.frames[5].source_pts_us
        # thresholds: scalar from the trace's own config + per-frame arrays
        assert s["thresholds"]["speed_drop_frac"] == pytest.approx(0.42)
        assert "exact active config" in s["thresholds"]["note"]
        assert len(s["thr_bs_s"]["left"]) == len(s["t_s"])
        assert [v for v in s["thr_bs_s"]["left"] if v is not None] == [
            pytest.approx(0.168)] * 4
        # support band straight from band_h_px / scale_px
        assert s["support_band_source"].startswith("contact-trace.jsonl")
        assert [v for v in s["support_band_by_side"]["left"]
                if v is not None] == [
            pytest.approx(19.06 / 95.3, abs=1e-4)] * 4
        assert s["support_band_score"][4] == pytest.approx(
            19.06 / 95.3, abs=1e-4)
        # detector's own per-frame reference speed
        assert [v for v in s["vel"]["reference"]["left"] if v is not None] \
            == [pytest.approx(1.2345)] * 4
        # GT overlay carries the diagnostic pair + record metadata
        g = win["gt"]
        assert g["gt_id"] == "gt#000" and g["foot"] == "left"
        assert g["visibility"] == "clear" and g["viewpoint"] == "side"
        assert g["assoc_err_frames"] == 0 and g["assoc_side_ok"] is True
        assert "exact" in win["notes"][0].lower()


class TestReconstructedWindow:
    def test_absent_trace_flags_and_defaults(self):
        trace = make_trace(n=24, fps=12.0)
        events = _events_doc_events(trace)
        win = build_window(
            trace, events, None, events[0]["source_pts_us"],
            kind="predicted", half_s=0.5,
        )
        assert win["trace"] == {"source": "none", "exact": False}
        s = win["series"]
        assert "thr_bs_s" not in s
        assert s["thresholds"]["speed_drop_frac"] == pytest.approx(
            default_config("hybrid")["speed_drop_frac"])
        assert any("approximat" in n.lower() for n in win["notes"])
        assert set(s["state"]["left"]) <= {"PRED_WINDOW", "-"}


class TestMissedGtDiagnosticSelection:
    def test_pred_between_two_gts_yields_one_missed(self):
        trace = make_trace(n=24, fps=12.0)
        events = _events_doc_events(trace)
        base = events[0]["time_s"]

        def gt(dt):
            return {"foot": "left", "preferred_s": base + dt,
                    "earliest_s": base + dt - 0.02,
                    "latest_s": base + dt + 0.02}

        # One prediction between two GT contacts: one-to-one diagnostic
        # pairing matches exactly ONE of them -> the other is missed.
        # (The old any-prediction-within-horizon loop reported neither.)
        windows = select_windows(events, [gt(-0.30), gt(+0.25)], None)
        kinds = [k for k, _c, _l in windows]
        assert kinds.count("predicted") == 1
        assert kinds.count("missed_gt") == 1
        missed_label = next(l for k, _c, l in windows if k == "missed_gt")
        assert missed_label == f"gt_{int(round((base - 0.30) * 1000))}ms"


def _write_run_dir(base: Path, trace: PoseTrace, events: list[dict],
                   *, with_trace: bool) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    header = {
        "format": TRACE_FORMAT, "format_version": TRACE_FORMAT_VERSION,
        "key": "test-key", "backend": trace.backend, "mode": trace.mode,
        "meta": trace.meta.to_dict(),
    }
    body = "".join(json.dumps(r, sort_keys=True) + "\n"
                   for r in trace.to_rows())
    (base / "pose.jsonl").write_text(
        json.dumps(header, sort_keys=True) + "\n" + body, "utf-8")
    (base / "events.json").write_text(json.dumps(
        {"contact_method": "hybrid", "events": events}), "utf-8")
    if with_trace:
        _write_contact_trace(base, trace)
    return base


class TestBuildForensicsEndToEnd:
    """Full pipeline on a synthetic run dir; PNG rendering is stubbed here
    and covered for real by TestPlotRendering (matplotlib-gated)."""

    def _run(self, tmp_path, monkeypatch, *, with_trace):
        monkeypatch.setattr(forensics_mod, "plot_window",
                            lambda win, p: str(p))
        monkeypatch.setattr(forensics_mod, "plot_summary",
                            lambda trace, g, p, cycles=4: str(p))
        trace = make_trace(n=24, fps=_FPS)
        events = _events_doc_events(trace)
        run_dir = _write_run_dir(tmp_path / "run", trace, events,
                                 with_trace=with_trace)
        base = events[0]["time_s"]
        gt_path = tmp_path / "gt.json"
        gt_path.write_text(json.dumps([
            {"time_s": base + 0.03, "foot": "left"},
            {"time_s": 1.5, "foot": "right"},  # far from any pred: missed
        ]), "utf-8")
        out = build_forensics(run_dir, gt_path, tmp_path / "out")
        index = json.loads((out / "index.json").read_text())
        wins = index["windows"]
        pred = next(w for w in wins if w["kind"] == "predicted")
        missed = next(w for w in wins if w["kind"] == "missed_gt")
        win = json.loads((out / pred["json"]).read_text())
        return win, pred, missed

    def test_with_trace_exact_and_index_fields(self, tmp_path, monkeypatch):
        win, pred, missed = self._run(tmp_path, monkeypatch, with_trace=True)
        assert win["trace"] == {"source": "contact-trace.jsonl",
                                "exact": True}
        # matched prediction: gt_id + signed frame error in the index
        assert pred["gt_id"] == "gt#000"
        assert isinstance(pred["assoc_err_frames"], int)
        assert win["gt"]["gt_id"] == "gt#000"
        # missed GT: carries its record identity, never a fabricated error
        assert missed["gt_id"] == "gt#001"
        assert "assoc_err_frames" not in missed

    def test_without_trace_honest_fallback(self, tmp_path, monkeypatch):
        win, _pred, _missed = self._run(tmp_path, monkeypatch,
                                        with_trace=False)
        assert win["trace"] == {"source": "none", "exact": False}
        assert any("approximat" in n.lower() for n in win["notes"])


class TestPlotRendering:
    def test_exact_window_renders_state_ribbon(self, tmp_path):
        pytest.importorskip("matplotlib")
        trace = make_trace(n=24, fps=_FPS)
        events = _events_doc_events(trace)
        _write_contact_trace(tmp_path, trace)
        rows, header = load_contact_trace(tmp_path)
        win = build_window(
            trace, events, None, events[0]["source_pts_us"],
            kind="predicted", half_s=0.5,
            trace_index=ContactTraceIndex.from_rows(rows, header),
        )
        out = forensics_mod.plot_window(win, tmp_path / "exact.png")
        assert Path(out).is_file() and Path(out).stat().st_size > 0

    def test_reconstructed_window_still_plots(self, tmp_path):
        pytest.importorskip("matplotlib")
        trace = make_trace(n=24, fps=12.0)
        events = _events_doc_events(trace)
        win = build_window(
            trace, events, None, events[0]["source_pts_us"],
            kind="predicted", half_s=0.5,
        )
        out = forensics_mod.plot_window(win, tmp_path / "approx.png")
        assert Path(out).is_file()


