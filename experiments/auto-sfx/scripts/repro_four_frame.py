"""Four-frame metric-failure reproducer (Phase-2 measurement-gate repair).

Scenario: one GT initial contact at t=10.0 s; the detector fires exactly
FOUR displayed frames late at 30 fps (+0.1333 s) -- the centre of the
2-6-frame wrong-phase band the T4 audit predicts.

Before the repair, metrics v2 used the strict ~66 ms tolerance for BOTH
existence and timing: the late prediction was an unmatched miss/FP and
never entered the timing arrays, so catastrophic_gt3frame reported 0 --
the exact known failure mode was invisible to the measurement gate.

Run:  python scripts/repro_four_frame.py [--after]
Prints the observable quantities; exit code 0 when the documented
before/after expectations hold for the current code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from auto_sfx import metrics as M  # noqa: E402

FPS = 30.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", action="store_true",
                    help="assert post-repair behaviour")
    args = ap.parse_args()

    # GT window is +/-1 frame around the preferred instant (defensible
    # interval); the prediction sits 4 frames after preferred.
    gt = [{
        "gt_id": "g1", "foot": "left",
        "preferred_s": 10.0, "earliest_s": 10.0 - 1 / FPS,
        "latest_s": 10.0 + 1 / FPS,
        "visibility": "clear", "viewpoint": "sagittal", "notes": "",
        "preferred_pts_us": 10_000_000,
        "earliest_pts_us": 9_966_667, "latest_pts_us": 10_033_333,
    }]
    events = [{"id": "e1", "time_s": 10.0 + 4 / FPS, "foot": "left",
               "detector": "mediapipe-contact-hybrid-v0", "method": "hybrid"}]

    ex = M.evaluate(events, [{"time_s": r["preferred_s"], "foot": r["foot"]}
                             for r in gt], FPS)
    v2 = M.evaluate_v2(events, gt, FPS)

    print(f"existence.matched            {ex['matched']}")
    print(f"existence.misses             {ex['misses']}")
    print(f"existence.false_positives    {ex['false_positives']}")
    print(f"timing.n                     {v2['timing']['n']}")
    print(f"timing.signed_errors_s       {v2['timing'].get('signed_errors_s')}")
    print(f"timing.catastrophic_gt3frame "
          f"{v2['timing'].get('catastrophic_gt3frame')}")
    diag_rows = v2["timing"].get("diagnostic_pairs") or []
    if diag_rows:
        print("timing.diagnostic_pairs[0]   "
              + ", ".join(f"{k}={diag_rows[0].get(k)}"
                          for k in ("pred_id", "gt_id", "err_frames")))
    pp = v2.get("phase", {}).get("per_prediction") or []
    if pp:
        print("phase.per_prediction[0]      "
              + ", ".join(f"{k}={pp[0].get(k)}"
                          for k in ("pred_id", "assoc_gt_id", "phase_class")))

    if not args.after:
        # Documented broken state: the +4f error never reaches timing.
        assert ex["matched"] == 0 and ex["false_positives"] == 1
        assert v2["timing"]["n"] == 0
        assert not v2["timing"].get("catastrophic_gt3frame")
        print("\nBEFORE-state reproduced: +4f prediction is an existence "
              "miss/FP AND absent from timing (catastrophic_gt3frame=0).")
        return 0

    # Post-repair contract: existence unchanged (strict), timing now sees it.
    assert ex["matched"] == 0 and ex["false_positives"] == 1, \
        "strict existence semantics must not change"
    assert v2["existence"]["matched"] == 0 and \
        v2["existence"]["false_positives"] == 1
    assert v2["timing"]["n"] == 1, "diagnostic timing must see the +4f pair"
    pair = v2["timing"]["diagnostic_pairs"][0]
    assert pair["err_frames"] == 4, pair
    assert v2["timing"]["catastrophic_gt3frame"] == 1
    assert pair["inside_gt_window"] is False
    print("\nAFTER-state verified: existence still strict miss/FP, but the "
          "+4f error appears in diagnostic timing and increments "
          "catastrophic_gt3frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
