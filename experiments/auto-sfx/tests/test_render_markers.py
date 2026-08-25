"""Tests for auto_sfx.render / markers / resources.

Pure-synthetic: no video files, models, network or ffmpeg requirement
(the end-to-end render test degrades gracefully when either is missing).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx import markers, render, resources  # noqa: E402
from auto_sfx.types import (  # noqa: E402
    LEFT,
    RIGHT,
    ContactEvent,
    FootObservation,
    PersonFrame,
    Point,
    PoseTrace,
    VideoMeta,
)

SR = markers.DEFAULT_SAMPLE_RATE


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_event(foot=LEFT, t=1.0, *, observable=True, score=0.9,
               kind="initial_contact", window=0.08) -> ContactEvent:
    return ContactEvent(
        id=f"ev-{foot}-{t}", family="locomotion.foot_contact",
        actor_id="person_01", foot=foot, source_pts_us=int(t * 1e6), time_s=t,
        earliest_s=t - window, latest_s=t + window, score=score,
        observable=observable, detector="test-detector", method="hybrid",
        evidence={}, kind=kind,
    )


def make_meta(w=64, h=48, fps=12.0, n=6) -> VideoMeta:
    return VideoMeta(path="synthetic", codec="raw", width=w, height=h,
                     fps_stream=fps, n_frames=n, duration_s=n / fps,
                     cfr=True, vfr=False, time_base=None, audio_present=False)


def make_pf(i: int, fps=12.0, *, lost=False) -> PersonFrame:
    heel_l = Point(10.0 + i, 40.0 - i, 0.0, 0.9)
    return PersonFrame(
        t_s=i / fps, source_pts_us=int(i / fps * 1e6), frame_index=i,
        person_id=0, score=0.9,
        feet={
            LEFT: FootObservation(
                heel=heel_l,
                ankle=Point(10.0 + i, 36.0 - i, 0.0, 0.8),
                toe=Point(14.0 + i, 44.0 - i, 0.0, 0.7),
            ),
            RIGHT: FootObservation(
                heel=Point(50.0 - i, 40.0 - i, 0.0, 0.4),
                ankle=None,
                toe=Point(46.0 - i, 44.0 - i, 0.0, 0.35),
            ),
        },
        hips=Point(32.0, 24.0 + i, 0.0, 0.9),
        shoulders_mid=Point(32.0, 10.0 + i, 0.0, 0.9),
        landmarks=[Point(x=float((k % 4) * 16 + 8), y=float(k // 4 * 12 + 6),
                         z=0.0, visibility=0.9) for k in range(17)],
        lost=lost,
    )


def make_trace(n=6, fps=12.0, backend="mediapipe-full") -> PoseTrace:
    return PoseTrace(backend=backend, mode="video",
                     meta=make_meta(fps=fps, n=n),
                     frames=[make_pf(i, fps) for i in range(n)])


# ---------------------------------------------------------------------------
# markers.py
# ---------------------------------------------------------------------------


class TestClickSynthesis:
    def test_click_shape_and_decay(self):
        sig = markers.click_waveform(markers.CLICK_LEFT_HZ, sr=SR)
        assert len(sig) == int(round(SR * markers.CLICK_DURATION_S))
        assert np.all(np.isfinite(sig))
        assert abs(np.max(np.abs(sig)) - 1.0) < 1e-9
        quarter = len(sig) // 4
        early = float(np.sum(sig[:quarter] ** 2))
        late = float(np.sum(sig[-quarter:] ** 2))
        assert early > late * 10.0          # percussive decay

    def test_track_pans_by_side(self):
        events = [make_event(LEFT, t=0.5), make_event(RIGHT, t=1.5)]
        track = markers.render_click_track(events, duration_s=2.4)
        assert track.dtype == np.int16 and track.shape[1] == 2

        def band_energy(chan, t0, t1):
            seg = track[int(t0 * SR):int(t1 * SR), chan].astype(np.float64)
            return float(np.sqrt(np.mean(seg ** 2)))

        # left event: left channel dominates; right event: right dominates
        assert band_energy(0, 0.45, 0.65) > band_energy(1, 0.45, 0.65)
        assert band_energy(1, 1.45, 1.65) > band_energy(0, 1.45, 1.65)

    def test_unobservable_events_are_silent(self):
        track = markers.render_click_track([make_event(LEFT, 0.3, observable=False)],
                                           duration_s=1.0)
        assert int(np.max(np.abs(track))) == 0

    def test_out_of_range_events_dropped(self):
        events = [make_event(LEFT, -0.2), make_event(RIGHT, 5.0)]
        track = markers.render_click_track(events, duration_s=1.0)
        assert int(np.max(np.abs(track))) == 0

    def test_overlapping_clicks_do_not_clip(self):
        events = [make_event(LEFT, 0.5, score=1.0),
                  make_event(RIGHT, 0.5, score=1.0),
                  make_event(LEFT, 0.52, score=1.0)]
        track = markers.render_click_track(events, duration_s=1.0, gain=1.0)
        assert int(track.max()) <= 32767 and int(track.min()) >= -32768
        assert int(np.max(np.abs(track))) > 20000

    def test_toe_off_quieter_than_initial_contact(self):
        loud = markers.render_click_track([make_event(LEFT, 0.5)], duration_s=1.0)
        soft = markers.render_click_track(
            [make_event(LEFT, 0.5, kind="toe_off")], duration_s=1.0)
        e = lambda tr: float(np.max(np.abs(tr).astype(np.int64)))
        assert e(loud) > e(soft)

    def test_wav_round_trip(self, tmp_path):
        out = tmp_path / "clicks.wav"
        dur = 1.25
        path = markers.synthesize_marker_wav([make_event(RIGHT, 0.7)],
                                             duration_s=dur, out_path=out)
        assert path == str(out) and out.stat().st_size > 44
        with wave.open(str(out), "rb") as wf:
            assert wf.getnchannels() == 2
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == SR
            n = wf.getnframes()
            raw = wf.readframes(n)
        assert n == int(round(dur * SR))
        data = np.frombuffer(raw, dtype="<i2").reshape(-1, 2)
        assert data.shape == (n, 2)
        assert int(np.max(np.abs(data))) > 10000


# ---------------------------------------------------------------------------
# resources.py
# ---------------------------------------------------------------------------


class TestResourceTracker:
    def test_stage_timing_and_totals(self):
        with resources.ResourceTracker() as tr:
            with tr.stage("alpha"):
                time.sleep(0.02)
            with tr.stage("beta"):
                pass
            rep = tr.report()
        assert rep.stage_names == ["alpha", "beta"]
        by_name = {s.name: s.duration_s for s in rep.stages}
        assert by_name["alpha"] >= 0.015
        assert by_name["beta"] >= 0.0
        assert rep.total_s >= sum(by_name.values())

    def test_failing_stage_still_recorded(self):
        tr = resources.ResourceTracker()
        with pytest.raises(RuntimeError):
            with tr.stage("boom"):
                raise RuntimeError("x")
        rep = tr.report()
        assert [s.name for s in rep.stages] == ["boom"]

    def test_sampler_collects_timeline(self):
        tr = resources.ResourceTracker(sample_interval_s=0.02).start()
        time.sleep(0.09)
        rep = tr.report()
        assert len(rep.rss_samples) >= 1
        ts = [t for t, _b in rep.rss_samples]
        assert ts == sorted(ts)
        assert all(b > 0 for _t, b in rep.rss_samples)
        assert rep.peak_rss_bytes and rep.peak_rss_bytes > 0

    def test_report_dict_is_json_safe(self):
        with resources.ResourceTracker() as tr:
            with tr.stage("ingest"):
                pass
            rep = tr.report()
        d = rep.to_dict()
        assert set(d) == {"stages", "total_s", "peak_rss_bytes",
                          "rss_samples", "sample_interval_s", "platform"}
        assert all(set(s) == {"name", "duration_s"} for s in d["stages"])
        json.dumps(d)                       # must not raise

    def test_rss_helpers_return_positive_bytes(self):
        assert resources.current_rss_bytes() > 0
        assert resources.peak_rss_bytes() > 0


# ---------------------------------------------------------------------------
# render.py — pure helpers
# ---------------------------------------------------------------------------


class TestRenderHelpers:
    def test_module_import_does_not_pull_cv2(self):
        code = (
            "import sys; import auto_sfx.markers, auto_sfx.render, "
            "auto_sfx.resources; "
            "sys.exit(1 if 'cv2' in sys.modules else 0)"
        )
        env = {**os.environ, "PYTHONPATH": _SRC}
        subprocess.run([sys.executable, "-c", code], env=env, check=True,
                       timeout=60)

    def test_active_events_window_membership(self):
        evs = [make_event(LEFT, 1.0, window=0.08),
               make_event(RIGHT, 2.0, window=0.05)]
        assert [e.foot for e in render.active_events(evs, 1.05)] == [LEFT]
        assert render.active_events(evs, 1.5) == []
        assert [e.foot for e in render.active_events(evs, 2.0)] == [RIGHT]

    def test_classify_foot_states(self):
        obs_good = FootObservation(heel=Point(1, 1, 0, 0.9),
                                   toe=Point(2, 2, 0, 0.8), ankle=None)
        obs_low = FootObservation(heel=Point(1, 1, 0, 0.3), toe=None, ankle=None)
        contact_ev = make_event(LEFT, 1.0, window=0.1)
        assert render.classify_foot(obs_good, [contact_ev]) == "contact"
        assert render.classify_foot(obs_low, []) == "lowconf"
        assert render.classify_foot(obs_good, []) == "swing"
        assert render.classify_foot(None, [contact_ev]) == "lost"
        assert render.classify_foot(
            FootObservation(heel=None, toe=None, ankle=None), []) == "lost"

    def test_update_state_builds_speed_history(self):
        state = render.RenderState(window_s=2.0)
        for i in range(4):
            pf = make_pf(i)
            render.update_state(state, pf, t_s=i / 12.0)
        hist = list(state.hist[LEFT])
        assert len(hist) >= 3                      # first frame seeds only
        assert all(np.isfinite(v) and v >= 0 for _t, v in hist)
        assert state.hist[RIGHT][0][1] >= 0.0      # low-vis right heel still tracked

    def test_coerce_frame_item_variants(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        assert render._coerce_frame_item(img) == (None, None, img)
        assert render._coerce_frame_item((3, img)) == (3, None, img)
        idx, ts, got = render._coerce_frame_item((img, 0.25, 7))
        assert (idx, ts, got is img) == (7, 0.25, True)


class TestAnnotateFrame:
    def test_annotate_draws_marks_on_copy(self):
        img = np.zeros((120, 160, 3), dtype=np.uint8)
        original = img.copy()
        pf = make_pf(2)
        state = render.RenderState()
        states = render.annotate_frame(img, pf, t_s=2 / 12.0,
                                       events=[make_event(LEFT, 2 / 12.0, window=0.2)],
                                       state=state, backend="mediapipe-full")
        assert set(states) == {LEFT, RIGHT}
        assert states[LEFT] == "contact"           # window covers t_s
        assert not np.array_equal(img, original)   # pixels changed
        assert img.shape == original.shape         # same geometry

    def test_annotate_handles_missing_pose_row(self):
        img = np.zeros((90, 120, 3), dtype=np.uint8)
        states = render.annotate_frame(img, None, t_s=0.5, events=[],
                                       backend="mediapipe-full")
        assert states == {LEFT: "lost", RIGHT: "lost"}

    def test_graph_history_windowed(self):
        img = np.zeros((120, 480, 3), dtype=np.uint8)
        state = render.RenderState(window_s=0.5, graph_wh=(160, 60))
        for i in range(10):
            render.update_state(state, make_pf(i), t_s=i / 12.0)
        render.draw_scrolling_graph(img, state, t_now=9 / 12.0)
        visible = [t for t, _v in state.hist[LEFT]]
        assert all(t >= 9 / 12.0 - state.window_s - 1e-9 for t in visible)


# ---------------------------------------------------------------------------
# render.py — end-to-end
# ---------------------------------------------------------------------------


class TestRenderAnnotated:
    def _frames(self, trace, w=64, h=48):
        for i in range(trace.meta.n_frames):
            yield i, np.full((h, w, 3), (i * 30) % 255, dtype=np.uint8)

    def test_end_to_end_with_mux(self, tmp_path):
        trace = make_trace(n=6, fps=12.0)
        events = [make_event(LEFT, 0.25, observable=True),
                  make_event(RIGHT, 0.6, observable=False)]
        wav = tmp_path / "clicks.wav"
        markers.synthesize_marker_wav(events, trace.meta.duration_s, wav)
        out = tmp_path / "diag.mp4"

        result = render.render_annotated(trace, events, self._frames(trace),
                                         out, click_wav=str(wav))
        assert result.n_frames == 6
        assert (result.width, result.height) == (64, 48)
        assert result.fps == pytest.approx(12.0)
        assert result.duration_s == pytest.approx(0.5)
        assert Path(result.path) == out and out.stat().st_size > 0
        assert result.audio_muxed == render.have_ffmpeg()

    def test_no_audio_path_moves_silent_video(self, tmp_path):
        trace = make_trace(n=4, fps=12.0)
        out = tmp_path / "silent.mp4"
        result = render.render_annotated(trace, [], self._frames(trace), out)
        assert result.n_frames == 4
        assert result.audio_muxed is False
        assert out.exists() and out.stat().st_size > 0

    def test_mux_skipped_when_ffmpeg_absent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(render, "have_ffmpeg", lambda: False)
        trace = make_trace(n=3, fps=12.0)
        wav = tmp_path / "clicks.wav"
        markers.synthesize_marker_wav([], 0.25, wav)
        out = tmp_path / "diag.mp4"
        result = render.render_annotated(trace, [], self._frames(trace), out,
                                         click_wav=str(wav))
        assert result.audio_muxed is False
        assert out.stat().st_size > 0

    def test_bare_ndarray_items_accepted(self, tmp_path):
        trace = make_trace(n=3, fps=12.0)
        frames = iter(np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(3))
        result = render.render_annotated(trace, [], frames,
                                         tmp_path / "bare.mp4")
        assert result.n_frames == 3

    def test_ingest_native_triple_items_accepted(self, tmp_path):
        trace = make_trace(n=3, fps=12.0)
        frames = iter((np.zeros((48, 64, 3), dtype=np.uint8), i / 12.0, i)
                      for i in range(3))
        result = render.render_annotated(trace, [], frames,
                                         tmp_path / "triples.mp4")
        assert result.n_frames == 3
        assert result.duration_s == pytest.approx(0.25)

    def test_empty_iterator_raises(self, tmp_path):
        trace = make_trace(n=6)
        with pytest.raises(ValueError):
            render.render_annotated(trace, [], iter(()), tmp_path / "x.mp4")

    def test_bad_payload_raises(self, tmp_path):
        trace = make_trace(n=2)
        with pytest.raises(ValueError):
            render.render_annotated(trace, [], iter([None]), tmp_path / "y.mp4")


# ---------------------------------------------------------------------------
# Phase-2 T15: GT voice in the marker WAV + GT-vs-prediction truth strip
# ---------------------------------------------------------------------------


def _gt_contact(i: int, foot: str):
    """GTContact without importing the loader machinery."""

    from auto_sfx.gt import GTContact

    base = i * 1_000_000
    return GTContact(
        foot=foot, earliest_pts_us=base,
        preferred_pts_us=base + 33_333, latest_pts_us=base + 66_666,
    )


class TestGroundTruthVoice:
    def test_gt_voice_constants_are_distinct_from_predictions(self):
        assert markers.GT_CLICK_HZ < markers.CLICK_LEFT_HZ
        assert markers.GT_CLICK_DURATION_S > markers.CLICK_DURATION_S
        # Centred image vs hard-panned prediction clicks.
        assert markers.GT_CLICK_PAN[0] == markers.GT_CLICK_PAN[1]

    def test_gt_marks_land_in_the_mix_without_events(self):
        stereo = markers.render_click_track([], duration_s=2.0, gt_marks=[1.0])
        n = stereo.shape[0]
        lo, hi = int(0.95 * SR), min(int(1.10 * SR), n)
        assert np.max(np.abs(stereo[lo:hi])) > 1000
        before = stereo[max(0, lo - SR // 4): lo - 100]
        assert np.max(np.abs(before)) < 100      # silence before the mark

    def test_marks_outside_duration_are_dropped(self):
        stereo = markers.render_click_track([], duration_s=0.5, gt_marks=[9.9])
        assert np.max(np.abs(stereo)) == 0

    def test_track_len_covers_last_mark(self, tmp_path):
        out = tmp_path / "m.wav"
        markers.synthesize_marker_wav(
            [], duration_s=3.0, out_path=out, gt_marks=[2.5]
        )
        with wave.open(str(out), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
        assert frames / rate == pytest.approx(3.0, abs=1e-6)


class TestTruthStripHelpers:
    def test_gt_state_midpoint_phase_map(self):
        gtc = [_gt_contact(10, "left"), _gt_contact(12, "right"),
               _gt_contact(14, "left")]
        # left marks ~10.03/14.03 -> midpoint 12.03 splits STANCE|SWING
        assert render.gt_state_at(gtc, "left", 11.5) == "STANCE"
        assert render.gt_state_at(gtc, "left", 12.5) == "SWING"
        # before first / after last mark: unknown, never invented
        assert render.gt_state_at(gtc, "left", 9.5) == "--"
        assert render.gt_state_at(gtc, "left", 15.5) == "--"
        # right has a single mark: no next mark, so always unknown
        assert render.gt_state_at(gtc, "right", 13.5) == "--"
        assert render.gt_state_at([], "left", 10.0) == "--"

    def test_nearest_diff_frames_signed_and_rounded(self):
        gtc = [_gt_contact(10, "left")]          # preferred at ~10.033 s
        t_true = gtc[0].preferred_s
        assert render.nearest_gt_diff_frames(
            gtc, "left", t_true + 3 / 30.0, 30.0) == 3
        assert render.nearest_gt_diff_frames(
            gtc, "left", t_true - 2 / 30.0, 30.0) == -2
        assert render.nearest_gt_diff_frames(
            gtc, "left", t_true + 1.0, 30.0) is None   # beyond horizon


class TestDrawTruthStrip:
    def _canvas(self, w=640, h=360):
        return np.zeros((h, w, 3), dtype=np.uint8)

    def test_draws_when_room_and_changes_pixels(self):
        img = self._canvas()
        baseline = img.copy()
        gtc = [_gt_contact(10, "left"), _gt_contact(12, "right")]
        events = [make_event(LEFT, 10.05)]
        render.draw_truth_strip(img, gtc, events, 11.0, fps_display=30.0)
        assert not np.array_equal(img, baseline)

    def test_pred_marker_gets_error_text_colouring(self):
        gtc = [_gt_contact(10, "left")]
        events = [
            make_event(LEFT, gtc[0].preferred_s + 5 / 30.0),
            make_event(RIGHT, 10.9),
        ]
        img = self._canvas()
        render.draw_truth_strip(img, gtc, events, 10.5, fps_display=30.0)
        assert (img != 0).any()                  # something was drawn

    def test_skips_tiny_frames_silently(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        baseline = img.copy()
        render.draw_truth_strip(img, [_gt_contact(1, "left")], [],
                                1.0, fps_display=30.0)
        assert np.array_equal(img, baseline)

    def test_annotate_frame_v1_unchanged_without_gt(self):
        trace = make_trace(n=2)
        pf = trace.frames[0]
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        states = render.annotate_frame(
            img, pf, pf.t_s, [make_event(LEFT, pf.t_s)],
            render.RenderState(), backend="t", mode="video",
        )
        assert set(states) == {LEFT, RIGHT}

    def test_annotate_frame_with_gt_draws_strip(self):
        trace = make_trace(n=2)
        pf = trace.frames[0]
        with_gt = np.zeros((360, 640, 3), dtype=np.uint8)
        without = with_gt.copy()
        events = sorted(
            [make_event(LEFT, 0.05), make_event(RIGHT, 0.30)],
            key=lambda e: e.time_s,
        )
        render.annotate_frame(without, pf, 0.08, [events[0]],
                              render.RenderState())
        render.annotate_frame(
            with_gt, pf, 0.08, [events[0]], render.RenderState(),
            all_events=events, gt_contacts=[_gt_contact(0, "left")],
        )
        assert not np.array_equal(with_gt, without)
