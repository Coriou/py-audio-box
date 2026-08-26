# Foot-contact operator annotation: two-pass guide

This path records visible initial foot contact only. It does not load detector
events, traces, overlays, generated audio, or another operator pass. Source PTS
comes from `ffprobe` for every displayed frame; the UI never substitutes
`frame / fps`, including for VFR clips.

## Prepare two isolated passes

Create a source-only manifest. `actor_ids` should name every person who may be
marked; `actor-1` is the default when the field is omitted.

```json
{
  "clips": [
    {
      "path": "work/operator-sources/walk-a.mp4",
      "clip_id": "walk-a",
      "actor_ids": ["lead", "background-1"]
    }
  ]
}
```

From the repository root, build each pass into a different empty directory and
use a different recorded order seed:

```bash
python3 experiments/auto-sfx/scripts/build_foot_contact_session.py \
  --clips work/operator-sources/challenge-clips.json \
  --out-dir work/auto-sfx/operator-gold/challenge-01/pass-a \
  --session-id challenge-01 --pass-id pass-a --order-seed challenge-01-a

python3 experiments/auto-sfx/scripts/build_foot_contact_session.py \
  --clips work/operator-sources/challenge-clips.json \
  --out-dir work/auto-sfx/operator-gold/challenge-01/pass-b \
  --session-id challenge-01 --pass-id pass-b --order-seed challenge-01-b

python3 -m http.server 8765
```

Open the exact URL printed by the builder. It is always repo-root-relative,
for example:

```text
http://127.0.0.1:8765/experiments/auto-sfx/foot_contact_annotator.html?session=work/auto-sfx/operator-gold/challenge-01/pass-a/session.json
```

The builder fails rather than synthesize missing timestamps, accept prediction
fields, reuse a non-empty pass directory, or serve a path outside the repo.

## Mark pass A, then pass B

1. Enter a stable annotator ID. Work only from the source video. Keep it muted
   while deciding the visible physical-contact window.
2. Step with `,` / `.` or the frame buttons. Set the earliest defensible,
   preferred, and latest defensible displayed frames. Do not widen a window to
   include a detector result or guess a fully invisible contact.
3. Record anatomical side (`left`, `right`, or `uncertain`), visibility,
   viewpoint, actor, contact observability, cropping, and concise observation
   notes. Retain uncertainty; do not infer side from image-x alone.
4. Audible onset is optional and separate. If it is in scope, first finish the
   physical window while muted, then listen and add the audible frame without
   moving the physical mark.
5. Download each clip's canonical JSON and place it in that pass directory's
   `annotations/` folder. An unchanged import then download is byte-stable.
6. Finish pass A and close it before opening pass B. Pass B must not see pass A
   exports. Distinct seeds randomize the pass order independently; verify the
   recorded orders differ when there is more than one clip. If one person
   performs both passes, separate them in time and describe them as two passes by one
   operator—not as independent annotators.

Do not compare passes or adjudicate until both raw export sets are immutable.
Keep both raw sets; adjudication is a separate append-only step.

## Validate an export

The enriched export remains readable by the existing schema-v2 GT loader. Use
the stricter operator loader with the matching PTS map before accepting it:

```bash
PYTHONPATH=experiments/auto-sfx/src python3 - <<'PY'
from auto_sfx.foot_contact_truth import load_annotation, load_pts_map

pts = load_pts_map("work/auto-sfx/operator-gold/challenge-01/pass-a/pts/walk-a.source-pts.json")
doc = load_annotation(
    "work/auto-sfx/operator-gold/challenge-01/pass-a/annotations/walk-a.operator-ben.pass-a.foot-contact.gt.json",
    pts_map=pts,
)
print(f"validated {len(doc['contacts'])} contacts")
PY
```

The strict loader verifies source and PTS-map hashes, exact displayed-frame to
source-PTS correspondence, window order, vocabularies, provenance, contact
ordering, and contradictory visibility/observability/cropping states.
