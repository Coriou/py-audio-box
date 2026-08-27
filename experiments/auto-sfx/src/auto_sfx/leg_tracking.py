"""Opt-in offline full-leg temporal identity prototype.

This module is intentionally not wired into the contact detector or CLI.  It
operates on a :class:`~auto_sfx.types.PoseTrace` and returns a separate audit
result.  Two concepts stay distinct throughout:

* temporal tracks follow continuous physical leg chains; and
* anatomical ``left``/``right`` is a segment-level posterior which may abstain.

MediaPipe's per-frame labels are observations, not track identities.  An
offline second-order two-state Viterbi decode assigns proximal and distal
components to two local tracks using bounded constant-velocity innovation and
a switch penalty.  Component disagreement is represented explicitly and is
never contact-evidence eligible.  The anatomical posterior uses only coherent
label consistency and unsigned torso depth/orientation strength as soft
evidence.  Image-x ordering is never used to decide anatomy.

No points are synthesized.  Missing/partial observations remain explicit, and
only a complete backend-observed hip/knee/ankle/heel/toe chain is marked as
eligible observed-pose input for a future contact stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .types import Point, PoseTrace


JOINTS = ("hip", "knee", "ankle", "heel", "toe")
TRACKS = ("track_0", "track_1")
ANATOMICAL_SIDES = ("left", "right", "uncertain")

# MediaPipe Pose's documented 33-landmark topology.  Kept local so this
# research branch does not alter backend or shared type modules.
MEDIAPIPE_LEG_INDICES = {
    "left": {"hip": 23, "knee": 25, "ankle": 27, "heel": 29, "toe": 31},
    "right": {"hip": 24, "knee": 26, "ankle": 28, "heel": 30, "toe": 32},
}
MEDIAPIPE_TORSO_INDICES = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
}


@dataclass(frozen=True)
class LegChain:
    """One backend-labelled leg chain with explicit per-joint missingness."""

    observed_label: str
    hip: Point | None
    knee: Point | None
    ankle: Point | None
    heel: Point | None
    toe: Point | None

    def point(self, joint: str) -> Point | None:
        if joint not in JOINTS:
            raise ValueError(f"unknown leg joint {joint!r}")
        return getattr(self, joint)

    @property
    def valid_joints(self) -> tuple[str, ...]:
        return tuple(joint for joint in JOINTS if self.point(joint) is not None)

    @property
    def complete(self) -> bool:
        return len(self.valid_joints) == len(JOINTS)

    @property
    def mean_visibility(self) -> float:
        values = [self.point(joint).visibility for joint in self.valid_joints]
        return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class TrackedLeg:
    """A local temporal track observation and separate anatomy claim."""

    track_id: str
    source_label: str | None
    chain: LegChain | None
    observed: bool
    interpolated: bool
    contact_evidence_eligible: bool
    anatomical_side: str
    posterior_left: float
    posterior_margin: float


@dataclass(frozen=True)
class TrackedLegFrame:
    frame_index: int
    t_s: float
    source_pts_us: int
    segment_id: str | None
    reset_reason: str | None
    observation_status: str
    assignment_state: str | None
    tracks: dict[str, TrackedLeg]


@dataclass(frozen=True)
class SegmentAttribution:
    segment_id: str
    track_0_side: str
    track_1_side: str
    track_0_posterior_left: float
    posterior_margin: float
    threshold: float
    orientation_strength: float
    paired_observation_fraction: float
    evidence_frames: int


@dataclass
class LegTrackingResult:
    frames: list[TrackedLegFrame]
    records: list[dict]
    summary: dict
    attributions: dict[str, SegmentAttribution]


@dataclass(frozen=True)
class _FrameObservation:
    position: int
    frame_index: int
    t_s: float
    source_pts_us: int
    chains: dict[str, LegChain | None]
    anchor: Point | None
    orientation_strength: float
    status: str

    @property
    def usable(self) -> bool:
        return any(chain is not None for chain in self.chains.values())


@dataclass(frozen=True)
class _Segment:
    segment_id: str
    valid_positions: tuple[int, ...]
    reset_reason: str


def _valid_point(point: Point | None, min_visibility: float) -> Point | None:
    if point is None:
        return None
    values = (point.x, point.y, point.z, point.visibility)
    if any(not math.isfinite(float(value)) for value in values):
        return None
    if point.visibility < min_visibility:
        return None
    # Isolate the audit result from later mutation of the input trace.
    return Point(float(point.x), float(point.y), float(point.z), float(point.visibility))


def _landmark(frame, index: int, min_visibility: float) -> Point | None:
    if index < 0 or index >= len(frame.landmarks):
        return None
    return _valid_point(frame.landmarks[index], min_visibility)


def _extract_chain(frame, side: str, min_visibility: float,
                   min_valid_joints: int) -> LegChain | None:
    points = {
        joint: _landmark(frame, index, min_visibility)
        for joint, index in MEDIAPIPE_LEG_INDICES[side].items()
    }
    chain = LegChain(observed_label=side, **points)
    return chain if len(chain.valid_joints) >= min_valid_joints else None


def _distance(left: Point, right: Point) -> float:
    return math.sqrt(
        (left.x - right.x) ** 2
        + (left.y - right.y) ** 2
        + (left.z - right.z) ** 2
    )


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _body_scale(observations: list[_FrameObservation], trace: PoseTrace) -> float:
    lengths: list[float] = []
    for observation in observations:
        for chain in observation.chains.values():
            if chain is None:
                continue
            if chain.hip is not None and chain.knee is not None:
                upper = _distance(chain.hip, chain.knee)
            else:
                upper = 0.0
            if chain.knee is not None and chain.ankle is not None:
                lower = _distance(chain.knee, chain.ankle)
            else:
                lower = 0.0
            if upper > 0.0 and lower > 0.0:
                lengths.append(upper + lower)
    fallback = max(float(trace.meta.height) * 0.35, 1.0)
    scale = _median(lengths)
    return scale if math.isfinite(scale) and scale > 1e-6 else fallback


def _anchor(frame, min_visibility: float) -> Point | None:
    direct = _valid_point(frame.hips, min_visibility)
    if direct is not None:
        return direct
    hips = [
        _landmark(frame, MEDIAPIPE_TORSO_INDICES[name], min_visibility)
        for name in ("left_hip", "right_hip")
    ]
    valid = [point for point in hips if point is not None]
    if not valid:
        return None
    return Point(
        x=sum(point.x for point in valid) / len(valid),
        y=sum(point.y for point in valid) / len(valid),
        z=sum(point.z for point in valid) / len(valid),
        visibility=min(point.visibility for point in valid),
    )


def _torso_orientation_strength(frame, scale: float, min_visibility: float) -> float:
    """Unsigned depth separation used only to weight anatomical evidence.

    The sign of x (and of z) is deliberately discarded.  This cannot turn an
    image-left leg into an anatomical-left leg.  Near-symmetric frontal/rear
    geometry therefore weakens attribution instead of forcing an inversion.
    """

    pairs = (("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"))
    separations: list[float] = []
    for left_name, right_name in pairs:
        left = _landmark(
            frame, MEDIAPIPE_TORSO_INDICES[left_name], min_visibility,
        )
        right = _landmark(
            frame, MEDIAPIPE_TORSO_INDICES[right_name], min_visibility,
        )
        if left is not None and right is not None:
            separations.append(abs(left.z - right.z) / max(scale, 1e-9))
    if not separations:
        return 0.0
    # Roughly one quarter of a leg length in left/right depth separation is
    # strong evidence of a side/diagonal torso.  It remains only a weight.
    return min(1.0, max(0.0, _median(separations) / 0.25))


def _extract_observations(
    trace: PoseTrace,
    *,
    min_visibility: float,
    min_valid_joints: int,
) -> tuple[list[_FrameObservation], float]:
    mediapipe = trace.backend.startswith("mediapipe-")
    preliminary: list[_FrameObservation] = []
    for position, frame in enumerate(trace.frames):
        chains: dict[str, LegChain | None] = {"left": None, "right": None}
        if not mediapipe:
            status = "unsupported_backend"
        elif frame.lost:
            status = "pose_lost"
        elif len(frame.landmarks) < 33:
            status = "missing_landmarks"
        else:
            chains = {
                side: _extract_chain(frame, side, min_visibility, min_valid_joints)
                for side in ("left", "right")
            }
            present = sum(chain is not None for chain in chains.values())
            status = (
                "observed_pair" if present == 2
                else "partial_pair" if present == 1
                else "insufficient_chain"
            )
        preliminary.append(_FrameObservation(
            position=position,
            frame_index=frame.frame_index,
            t_s=float(frame.t_s),
            source_pts_us=int(frame.source_pts_us),
            chains=chains,
            anchor=_anchor(frame, min_visibility) if mediapipe else None,
            orientation_strength=0.0,
            status=status,
        ))

    scale = _body_scale(preliminary, trace)
    observations = []
    for observation, frame in zip(preliminary, trace.frames):
        orientation = (
            _torso_orientation_strength(frame, scale, min_visibility)
            if mediapipe and not frame.lost and len(frame.landmarks) >= 33
            else 0.0
        )
        observations.append(_FrameObservation(
            position=observation.position,
            frame_index=observation.frame_index,
            t_s=observation.t_s,
            source_pts_us=observation.source_pts_us,
            chains=observation.chains,
            anchor=observation.anchor,
            orientation_strength=orientation,
            status=observation.status,
        ))
    return observations, scale


def _boundary_between(previous: _FrameObservation, current: _FrameObservation,
                      boundaries: set[int]) -> bool:
    return any(previous.frame_index < boundary <= current.frame_index
               for boundary in boundaries)


def _build_segments(observations: list[_FrameObservation], boundaries: set[int],
                    max_missing_gap_frames: int) -> list[_Segment]:
    valid = [observation for observation in observations if observation.usable]
    if not valid:
        return []
    groups: list[tuple[list[int], str]] = []
    current = [valid[0].position]
    current_reason = "trace_start"
    previous = valid[0]
    for observation in valid[1:]:
        missing_gap = observation.position - previous.position - 1
        reason = None
        if _boundary_between(previous, observation, boundaries):
            reason = "shot_boundary"
        elif observation.t_s <= previous.t_s:
            reason = "timestamp_discontinuity"
        elif missing_gap > max_missing_gap_frames:
            reason = "missing_gap"
        if reason is not None:
            groups.append((current, current_reason))
            current = [observation.position]
            current_reason = reason
        else:
            current.append(observation.position)
        previous = observation
    groups.append((current, current_reason))
    return [
        _Segment(f"segment_{index:03d}", tuple(positions), reason)
        for index, (positions, reason) in enumerate(groups)
    ]


def _relative_xyz(point: Point, anchor: Point | None) -> tuple[float, float, float]:
    if anchor is None:
        return point.x, point.y, point.z
    return point.x - anchor.x, point.y - anchor.y, point.z - anchor.z


def _huber(value: float, delta: float = 0.25) -> float:
    value = abs(value)
    if value <= delta:
        return 0.5 * value * value
    return delta * (value - 0.5 * delta)


def _source_for_track(state: int, track: str) -> str:
    if state == 0:
        return "left" if track == "track_0" else "right"
    return "right" if track == "track_0" else "left"


def _component_point(observation: _FrameObservation, state: int, track: str,
                     joint: str) -> Point | None:
    chain = observation.chains[_source_for_track(state, track)]
    return None if chain is None else chain.point(joint)


def _coordinates(point: Point, anchor: Point | None,
                 dimensions: str) -> tuple[float, ...]:
    x, y, z = _relative_xyz(point, anchor)
    return (x, y) if dimensions == "xy" else (x, y, z)


def _component_shape_cost(
    previous: _FrameObservation,
    current: _FrameObservation,
    previous_state: int,
    current_state: int,
    *,
    joints: tuple[str, ...],
    dimensions: str,
    scale: float,
    missing_penalty: float,
) -> float:
    losses: list[float] = []
    for track in TRACKS:
        for first, second in zip(joints, joints[1:]):
            pa = _component_point(previous, previous_state, track, first)
            pb = _component_point(previous, previous_state, track, second)
            qa = _component_point(current, current_state, track, first)
            qb = _component_point(current, current_state, track, second)
            if any(point is None for point in (pa, pb, qa, qb)):
                continue
            before_a = _coordinates(pa, None, dimensions)
            before_b = _coordinates(pb, None, dimensions)
            after_a = _coordinates(qa, None, dimensions)
            after_b = _coordinates(qb, None, dimensions)
            before = tuple(b - a for a, b in zip(before_a, before_b))
            after = tuple(b - a for a, b in zip(after_a, after_b))
            difference = math.sqrt(sum((b - a) ** 2 for a, b in zip(before, after)))
            losses.append(_huber(difference / max(scale, 1e-9)))
    return _median(losses) if losses else missing_penalty


def _component_position_cost(
    previous: _FrameObservation,
    current: _FrameObservation,
    previous_state: int,
    current_state: int,
    *,
    joints: tuple[str, ...],
    dimensions: str,
    scale: float,
    expected_dt: float,
    missing_penalty: float,
) -> float:
    elapsed = max(current.t_s - previous.t_s, expected_dt)
    gap_factor = max(1.0, elapsed / max(expected_dt, 1e-9))
    track_costs: list[float] = []
    for track in TRACKS:
        residuals: list[float] = []
        for joint in joints:
            before = _component_point(previous, previous_state, track, joint)
            after = _component_point(current, current_state, track, joint)
            if before is None or after is None:
                continue
            a = _coordinates(before, previous.anchor, dimensions)
            b = _coordinates(after, current.anchor, dimensions)
            distance = math.sqrt(sum((right - left) ** 2 for left, right in zip(a, b)))
            residuals.append(distance / max(scale * math.sqrt(gap_factor), 1e-9))
        track_costs.append(
            _median([_huber(value) for value in residuals])
            if residuals else missing_penalty
        )
    shape = _component_shape_cost(
        previous, current, previous_state, current_state,
        joints=joints, dimensions=dimensions, scale=scale,
        missing_penalty=missing_penalty,
    )
    return sum(track_costs) + 0.15 * shape


def _bounded_prediction(previous: tuple[float, ...], current: tuple[float, ...],
                        ratio: float, maximum_delta: float) -> tuple[float, ...]:
    delta = tuple((right - left) * ratio for left, right in zip(previous, current))
    magnitude = math.sqrt(sum(value * value for value in delta))
    if magnitude > maximum_delta > 0.0:
        delta = tuple(value * maximum_delta / magnitude for value in delta)
    return tuple(value + step for value, step in zip(current, delta))


def _component_velocity_cost(
    older: _FrameObservation,
    previous: _FrameObservation,
    current: _FrameObservation,
    older_state: int,
    previous_state: int,
    current_state: int,
    *,
    joints: tuple[str, ...],
    dimensions: str,
    scale: float,
    expected_dt: float,
    missing_penalty: float,
    max_prediction_step_bs: float,
) -> tuple[float, dict[str, dict[str, float | None]]]:
    older_dt = max(previous.t_s - older.t_s, expected_dt)
    current_dt = max(current.t_s - previous.t_s, expected_dt)
    ratio = current_dt / older_dt
    gap_factor = max(1.0, current_dt / max(expected_dt, 1e-9))
    maximum_delta = max_prediction_step_bs * scale * gap_factor
    details: dict[str, dict[str, float | None]] = {}
    track_costs: list[float] = []
    for track in TRACKS:
        residuals: list[float] = []
        joint_details: dict[str, float | None] = {}
        for joint in joints:
            first = _component_point(older, older_state, track, joint)
            second = _component_point(previous, previous_state, track, joint)
            observed = _component_point(current, current_state, track, joint)
            if first is None or second is None or observed is None:
                joint_details[joint] = None
                continue
            a = _coordinates(first, older.anchor, dimensions)
            b = _coordinates(second, previous.anchor, dimensions)
            c = _coordinates(observed, current.anchor, dimensions)
            predicted = _bounded_prediction(a, b, ratio, maximum_delta)
            residual = math.sqrt(sum((actual - estimate) ** 2
                                     for actual, estimate in zip(c, predicted)))
            normalized = residual / max(scale * math.sqrt(gap_factor), 1e-9)
            residuals.append(normalized)
            joint_details[joint] = round(normalized, 6)
        details[track] = joint_details
        track_costs.append(
            _median([_huber(value) for value in residuals])
            if residuals else missing_penalty
        )
    shape = _component_shape_cost(
        previous, current, previous_state, current_state,
        joints=joints, dimensions=dimensions, scale=scale,
        missing_penalty=missing_penalty,
    )
    return sum(track_costs) + 0.15 * shape, details


def _decode_component(
    segment: _Segment,
    observations: list[_FrameObservation],
    *,
    joints: tuple[str, ...],
    dimensions: str,
    scale: float,
    expected_dt: float,
    switch_penalty: float,
    seed_swap_penalty: float,
    missing_penalty: float,
    max_prediction_step_bs: float,
) -> tuple[dict[int, int], dict[int, dict]]:
    positions = segment.valid_positions
    if len(positions) == 1:
        return {positions[0]: 0}, {}

    first = observations[positions[0]]
    second = observations[positions[1]]
    pair_costs: dict[tuple[int, int], float] = {}
    for first_state in (0, 1):
        for second_state in (0, 1):
            pair_costs[(first_state, second_state)] = (
                (seed_swap_penalty if first_state else 0.0)
                + _component_position_cost(
                    first, second, first_state, second_state,
                    joints=joints, dimensions=dimensions, scale=scale,
                    expected_dt=expected_dt, missing_penalty=missing_penalty,
                )
                + (switch_penalty if first_state != second_state else 0.0)
            )

    backpointers: list[dict[tuple[int, int], int]] = []
    for offset in range(2, len(positions)):
        older = observations[positions[offset - 2]]
        previous = observations[positions[offset - 1]]
        current = observations[positions[offset]]
        next_costs: dict[tuple[int, int], float] = {}
        pointer: dict[tuple[int, int], int] = {}
        for previous_state in (0, 1):
            for current_state in (0, 1):
                candidates = []
                for older_state in (0, 1):
                    innovation, _ = _component_velocity_cost(
                        older, previous, current,
                        older_state, previous_state, current_state,
                        joints=joints, dimensions=dimensions, scale=scale,
                        expected_dt=expected_dt, missing_penalty=missing_penalty,
                        max_prediction_step_bs=max_prediction_step_bs,
                    )
                    total = pair_costs[(older_state, previous_state)] + innovation
                    if previous_state != current_state:
                        total += switch_penalty
                    candidates.append((total, older_state != previous_state, older_state))
                chosen = min(candidates)
                next_costs[(previous_state, current_state)] = chosen[0]
                pointer[(previous_state, current_state)] = chosen[2]
        pair_costs = next_costs
        backpointers.append(pointer)

    final_pair = min((cost, pair) for pair, cost in pair_costs.items())[1]
    decoded = [0] * len(positions)
    decoded[-2], decoded[-1] = final_pair
    for offset in range(len(positions) - 1, 1, -1):
        pointer = backpointers[offset - 2]
        decoded[offset - 2] = pointer[(decoded[offset - 1], decoded[offset])]
    states = dict(zip(positions, decoded))

    diagnostics: dict[int, dict] = {}
    for offset in range(2, len(positions)):
        older = observations[positions[offset - 2]]
        previous = observations[positions[offset - 1]]
        current = observations[positions[offset]]
        older_state = decoded[offset - 2]
        previous_state = decoded[offset - 1]
        candidates = {}
        for current_state in (0, 1):
            cost, joint_residuals = _component_velocity_cost(
                older, previous, current,
                older_state, previous_state, current_state,
                joints=joints, dimensions=dimensions, scale=scale,
                expected_dt=expected_dt, missing_penalty=missing_penalty,
                max_prediction_step_bs=max_prediction_step_bs,
            )
            candidates["direct" if current_state == 0 else "swapped"] = {
                "innovation_cost": round(cost, 6),
                "joint_residual_bs": joint_residuals,
            }
        diagnostics[positions[offset]] = {
            "model": f"bounded_constant_velocity_{dimensions}",
            "selected_state": "direct" if decoded[offset] == 0 else "swapped",
            "candidates": candidates,
        }
    return states, diagnostics


def _quarantine_long_reassignment_runs(
    states: dict[int, int],
    ordered_positions: tuple[int, ...],
    maximum_run: int,
) -> tuple[dict[int, int | None], set[int]]:
    """Keep bounded label repairs; mark longer crossed episodes unknowable."""

    safe: dict[int, int | None] = dict(states)
    quarantined: set[int] = set()
    start = 0
    while start < len(ordered_positions):
        position = ordered_positions[start]
        if states[position] != 1:
            start += 1
            continue
        end = start + 1
        while end < len(ordered_positions) and states[ordered_positions[end]] == 1:
            end += 1
        run = ordered_positions[start:end]
        if len(run) > maximum_run:
            for item in run:
                safe[item] = None
                quarantined.add(item)
        start = end
    return safe, quarantined


def _segment_attribution(
    segment: _Segment,
    observations: list[_FrameObservation],
    proximal_states: dict[int, int | None],
    distal_states: dict[int, int | None],
    threshold: float,
) -> SegmentAttribution:
    vote = 0.0
    vote_weight = 0.0
    orientation_weighted = 0.0
    paired = 0
    for position in segment.valid_positions:
        observation = observations[position]
        chains = [chain for chain in observation.chains.values() if chain is not None]
        visibility = sum(chain.mean_visibility for chain in chains) / len(chains)
        if (proximal_states[position] is None or distal_states[position] is None
                or proximal_states[position] != distal_states[position]):
            # An incoherent backend skeleton is representation evidence, not
            # anatomical-side evidence.  Exclude it from the label vote.
            continue
        vote_weight += visibility
        vote += visibility * (1.0 if distal_states[position] == 0 else -1.0)
        orientation_weighted += visibility * observation.orientation_strength
        paired += int(len(chains) == 2)
    orientation = orientation_weighted / vote_weight if vote_weight else 0.0
    consistency = abs(vote) / vote_weight if vote_weight else 0.0
    paired_fraction = paired / len(segment.valid_positions)
    margin = min(1.0, consistency * orientation * paired_fraction)
    posterior_left = 0.5 + 0.5 * margin * (1.0 if vote >= 0.0 else -1.0)
    if margin < threshold:
        track_0_side = track_1_side = "uncertain"
    elif posterior_left >= 0.5:
        track_0_side, track_1_side = "left", "right"
    else:
        track_0_side, track_1_side = "right", "left"
    return SegmentAttribution(
        segment_id=segment.segment_id,
        track_0_side=track_0_side,
        track_1_side=track_1_side,
        track_0_posterior_left=round(posterior_left, 6),
        posterior_margin=round(margin, 6),
        threshold=float(threshold),
        orientation_strength=round(orientation, 6),
        paired_observation_fraction=round(paired_fraction, 6),
        evidence_frames=len(segment.valid_positions),
    )


def _compose_chain(observation: _FrameObservation, track: str,
                   proximal_state: int | None,
                   distal_state: int | None) -> LegChain | None:
    sources = {
        "hip": None if proximal_state is None else _source_for_track(proximal_state, track),
        "knee": None if proximal_state is None else _source_for_track(proximal_state, track),
        "ankle": None if distal_state is None else _source_for_track(distal_state, track),
        "heel": None if distal_state is None else _source_for_track(distal_state, track),
        "toe": None if distal_state is None else _source_for_track(distal_state, track),
    }
    points = {}
    for joint, source in sources.items():
        chain = None if source is None else observation.chains[source]
        points[joint] = None if chain is None else chain.point(joint)
    if all(point is None for point in points.values()):
        return None
    source_labels = {source for source in sources.values() if source is not None}
    if not source_labels:
        return None
    observed_label = next(iter(source_labels)) if len(source_labels) == 1 else "mixed"
    return LegChain(observed_label=observed_label, **points)


def _empty_tracked_leg(track_id: str, segment_id: str | None,
                       attribution: SegmentAttribution | None) -> TrackedLeg:
    if attribution is None:
        side, posterior, margin = "uncertain", 0.5, 0.0
    elif track_id == "track_0":
        side = attribution.track_0_side
        posterior = attribution.track_0_posterior_left
        margin = attribution.posterior_margin
    else:
        side = attribution.track_1_side
        posterior = 1.0 - attribution.track_0_posterior_left
        margin = attribution.posterior_margin
    return TrackedLeg(
        track_id=f"{segment_id}.{track_id}" if segment_id is not None else track_id,
        source_label=None,
        chain=None,
        observed=False,
        interpolated=False,
        contact_evidence_eligible=False,
        anatomical_side=side,
        posterior_left=round(posterior, 6),
        posterior_margin=margin,
    )


def track_leg_chains(
    trace: PoseTrace,
    *,
    shot_boundary_frames: Iterable[int] = (),
    min_visibility: float = 0.35,
    min_valid_joints: int = 3,
    max_missing_gap_frames: int = 3,
    switch_penalty: float = 0.045,
    seed_swap_penalty: float = 0.10,
    missing_penalty: float = 0.20,
    max_prediction_step_bs: float = 0.35,
    max_reassignment_run_frames: int = 2,
    attribution_margin_threshold: float = 0.35,
) -> LegTrackingResult:
    """Run the opt-in offline leg-chain identity/attribution prototype.

    Explicit ``shot_boundary_frames`` begin new local track segments.  Gaps up
    to ``max_missing_gap_frames`` preserve the segment but remain missing;
    longer gaps reset identity on re-entry.  The input trace is never mutated.
    """

    if not 0.0 <= min_visibility <= 1.0:
        raise ValueError("min_visibility must be in [0, 1]")
    if not 1 <= min_valid_joints <= len(JOINTS):
        raise ValueError("min_valid_joints must be between 1 and 5")
    if max_missing_gap_frames < 0:
        raise ValueError("max_missing_gap_frames must be non-negative")
    if max_reassignment_run_frames < 1:
        raise ValueError("max_reassignment_run_frames must be positive")
    for name, value in (
        ("switch_penalty", switch_penalty),
        ("seed_swap_penalty", seed_swap_penalty),
        ("missing_penalty", missing_penalty),
        ("max_prediction_step_bs", max_prediction_step_bs),
    ):
        if value < 0.0 or not math.isfinite(value):
            raise ValueError(f"{name} must be finite and non-negative")
    if not 0.0 <= attribution_margin_threshold <= 1.0:
        raise ValueError("attribution_margin_threshold must be in [0, 1]")
    boundaries = set()
    for boundary in shot_boundary_frames:
        if not isinstance(boundary, int) or isinstance(boundary, bool):
            raise ValueError("shot_boundary_frames must contain integers")
        boundaries.add(boundary)

    observations, scale = _extract_observations(
        trace,
        min_visibility=min_visibility,
        min_valid_joints=min_valid_joints,
    )
    segments = _build_segments(observations, boundaries, max_missing_gap_frames)
    expected_dt = 1.0 / max(float(trace.meta.fps_stream), 1e-6)
    proximal_states_by_position: dict[int, int | None] = {}
    distal_states_by_position: dict[int, int | None] = {}
    innovations_by_position: dict[int, dict] = {}
    segment_by_position: dict[int, str] = {}
    reset_by_position: dict[int, str] = {}
    attributions: dict[str, SegmentAttribution] = {}
    transitions = 0
    corrected_frames = 0
    proximal_reassigned_frames = 0
    distal_reassigned_frames = 0
    incoherent_frames = 0
    quarantined_proximal_frames = 0
    quarantined_distal_frames = 0

    for segment in segments:
        proximal_states, proximal_diagnostics = _decode_component(
            segment, observations,
            joints=("hip", "knee"),
            dimensions="xyz",
            scale=scale,
            expected_dt=expected_dt,
            switch_penalty=switch_penalty,
            seed_swap_penalty=seed_swap_penalty,
            missing_penalty=missing_penalty,
            max_prediction_step_bs=max_prediction_step_bs,
        )
        distal_states, distal_diagnostics = _decode_component(
            segment, observations,
            joints=("ankle", "heel", "toe"),
            # The opened f364 fault exchanges distal image trajectories while
            # retaining contradictory depth labels.  Temporal 2-D identity is
            # decoded in xy; z remains present in the represented chain and
            # the proximal/distal disagreement prevents evidence eligibility.
            dimensions="xy",
            scale=scale,
            expected_dt=expected_dt,
            switch_penalty=switch_penalty,
            seed_swap_penalty=seed_swap_penalty,
            missing_penalty=missing_penalty,
            max_prediction_step_bs=max_prediction_step_bs,
        )
        safe_proximal, quarantined_proximal = _quarantine_long_reassignment_runs(
            proximal_states, segment.valid_positions, max_reassignment_run_frames,
        )
        safe_distal, quarantined_distal = _quarantine_long_reassignment_runs(
            distal_states, segment.valid_positions, max_reassignment_run_frames,
        )
        proximal_states_by_position.update(safe_proximal)
        distal_states_by_position.update(safe_distal)
        quarantined_proximal_frames += len(quarantined_proximal)
        quarantined_distal_frames += len(quarantined_distal)
        for position in segment.valid_positions:
            innovations_by_position[position] = {
                "proximal": proximal_diagnostics.get(position),
                "distal": distal_diagnostics.get(position),
            }
        first_position = segment.valid_positions[0]
        reset_by_position[first_position] = segment.reset_reason
        for position in range(segment.valid_positions[0], segment.valid_positions[-1] + 1):
            segment_by_position[position] = segment.segment_id
        ordered_states = [
            (safe_proximal[position], safe_distal[position])
            for position in segment.valid_positions
        ]
        transitions += sum(left != right for left, right in zip(ordered_states, ordered_states[1:]))
        corrected_frames += sum(
            proximal == 1 or distal == 1 for proximal, distal in ordered_states
        )
        proximal_reassigned_frames += sum(proximal == 1 for proximal, _ in ordered_states)
        distal_reassigned_frames += sum(distal == 1 for _, distal in ordered_states)
        incoherent_frames += sum(
            proximal is not None and distal is not None and proximal != distal
            for proximal, distal in ordered_states
        )
        attributions[segment.segment_id] = _segment_attribution(
            segment, observations, safe_proximal, safe_distal,
            attribution_margin_threshold,
        )

    frames: list[TrackedLegFrame] = []
    records: list[dict] = []
    for observation in observations:
        segment_id = segment_by_position.get(observation.position)
        attribution = attributions.get(segment_id) if segment_id is not None else None
        proximal_state = proximal_states_by_position.get(observation.position)
        distal_state = distal_states_by_position.get(observation.position)
        usable_state = observation.usable and segment_id is not None
        tracks: dict[str, TrackedLeg] = {}
        for track in TRACKS:
            if not usable_state:
                tracked = _empty_tracked_leg(track, segment_id, attribution)
            else:
                chain = _compose_chain(
                    observation, track, proximal_state, distal_state,
                )
                empty = _empty_tracked_leg(track, segment_id, attribution)
                tracked = TrackedLeg(
                    track_id=f"{segment_id}.{track}",
                    source_label=chain.observed_label if chain is not None else None,
                    chain=chain,
                    observed=chain is not None,
                    interpolated=False,
                    # Tracking alone never creates contact evidence.  This flag
                    # only states that all five inputs are truly observed.
                    contact_evidence_eligible=bool(
                        chain is not None and chain.complete
                        and proximal_state == distal_state
                    ),
                    anatomical_side=empty.anatomical_side,
                    posterior_left=empty.posterior_left,
                    posterior_margin=empty.posterior_margin,
                )
            tracks[track] = tracked

        if not usable_state:
            assignment = None
        elif proximal_state is None and distal_state is None:
            assignment = "quarantined"
        elif proximal_state is None:
            assignment = "proximal_quarantined"
        elif distal_state is None:
            assignment = "distal_quarantined"
        elif proximal_state == distal_state == 0:
            assignment = "direct"
        elif proximal_state == distal_state == 1:
            assignment = "swapped"
        elif proximal_state == 0:
            assignment = "distal_swapped"
        else:
            assignment = "proximal_swapped"
        if assignment is None:
            whole_chain_status = "missing"
        elif "quarantined" in assignment:
            whole_chain_status = "component_quarantined"
        elif assignment in {"distal_swapped", "proximal_swapped"}:
            whole_chain_status = "incoherent_mixed"
        else:
            whole_chain_status = "coherent_observed"
        reset_reason = reset_by_position.get(observation.position)
        frame_result = TrackedLegFrame(
            frame_index=observation.frame_index,
            t_s=observation.t_s,
            source_pts_us=observation.source_pts_us,
            segment_id=segment_id,
            reset_reason=reset_reason,
            observation_status=observation.status,
            assignment_state=assignment,
            tracks=tracks,
        )
        frames.append(frame_result)
        records.append({
            "kind": "leg_tracking",
            "frame": observation.frame_index,
            "t_s": round(observation.t_s, 6),
            "source_pts_us": observation.source_pts_us,
            "segment_id": segment_id,
            "reset_reason": reset_reason,
            "observation_status": observation.status,
            "assignment_state": assignment,
            "whole_chain_status": whole_chain_status,
            "orientation_strength": round(observation.orientation_strength, 6),
            "innovation": innovations_by_position.get(observation.position),
            "tracks": {
                track: {
                    "track_id": tracked.track_id,
                    "source_label": tracked.source_label,
                    "source_labels_by_joint": (
                        None if not usable_state else {
                            "hip": (
                                None if proximal_state is None
                                else _source_for_track(proximal_state, track)
                            ),
                            "knee": (
                                None if proximal_state is None
                                else _source_for_track(proximal_state, track)
                            ),
                            "ankle": (
                                None if distal_state is None
                                else _source_for_track(distal_state, track)
                            ),
                            "heel": (
                                None if distal_state is None
                                else _source_for_track(distal_state, track)
                            ),
                            "toe": (
                                None if distal_state is None
                                else _source_for_track(distal_state, track)
                            ),
                        }
                    ),
                    "observed": tracked.observed,
                    "interpolated": tracked.interpolated,
                    "valid_joints": (
                        list(tracked.chain.valid_joints) if tracked.chain is not None else []
                    ),
                    "contact_evidence_eligible": tracked.contact_evidence_eligible,
                    "anatomical_side": tracked.anatomical_side,
                    "posterior_left": tracked.posterior_left,
                    "posterior_margin": tracked.posterior_margin,
                }
                for track, tracked in tracks.items()
            },
        })

    uncertain_segments = sum(
        attribution.track_0_side == "uncertain" for attribution in attributions.values()
    )
    same_side_segments = sum(
        attribution.track_0_side == attribution.track_1_side != "uncertain"
        for attribution in attributions.values()
    )
    missing_frames = sum(not observation.usable for observation in observations)
    partial_frames = sum(observation.status == "partial_pair" for observation in observations)
    complete_track_observations = sum(
        tracked.contact_evidence_eligible
        for frame in frames for tracked in frame.tracks.values()
    )
    unsafe_whole_chain_frames = incoherent_frames + len({
        position
        for position, state in proximal_states_by_position.items()
        if state is None and observations[position].usable
    } | {
        position
        for position, state in distal_states_by_position.items()
        if state is None and observations[position].usable
    })
    segment_boundaries = []
    for segment in segments:
        first = observations[segment.valid_positions[0]]
        last = observations[segment.valid_positions[-1]]
        segment_boundaries.append({
            "segment_id": segment.segment_id,
            "reset_reason": segment.reset_reason,
            "first_frame": first.frame_index,
            "last_frame": last.frame_index,
            "first_source_pts_us": first.source_pts_us,
            "last_source_pts_us": last.source_pts_us,
            "valid_observation_frames": len(segment.valid_positions),
        })
    summary = {
        "prototype": "full-leg-chain-v1-opt-in",
        "backend": trace.backend,
        "mode": trace.mode,
        "fps_stream": float(trace.meta.fps_stream),
        "backend_supported": trace.backend.startswith("mediapipe-"),
        "frames": len(observations),
        "segments": len(segments),
        "segment_resets": max(0, len(segments) - 1),
        "scale_px": round(scale, 6),
        "assignment_state_transitions": transitions,
        "backend_label_reassigned_frames": corrected_frames,
        "proximal_reassigned_frames": proximal_reassigned_frames,
        "distal_reassigned_frames": distal_reassigned_frames,
        "incoherent_partial_chain_frames": incoherent_frames,
        "quarantined_proximal_frames": quarantined_proximal_frames,
        "quarantined_distal_frames": quarantined_distal_frames,
        "unsafe_whole_chain_frames": unsafe_whole_chain_frames,
        "missing_frames": missing_frames,
        "partial_pair_frames": partial_frames,
        "interpolated_frames": 0,
        "complete_observed_track_chains": complete_track_observations,
        "uncertain_attribution_segments": uncertain_segments,
        "confident_attribution_segments": len(segments) - uncertain_segments,
        "same_side_attribution_segments": same_side_segments,
        "attribution_margin_threshold": float(attribution_margin_threshold),
        "switch_penalty": float(switch_penalty),
        "seed_swap_penalty": float(seed_swap_penalty),
        "max_missing_gap_frames": int(max_missing_gap_frames),
        "effective_config": {
            "shot_boundary_frames": sorted(boundaries),
            "min_visibility": float(min_visibility),
            "min_valid_joints": int(min_valid_joints),
            "max_missing_gap_frames": int(max_missing_gap_frames),
            "switch_penalty": float(switch_penalty),
            "seed_swap_penalty": float(seed_swap_penalty),
            "missing_penalty": float(missing_penalty),
            "max_prediction_step_bs": float(max_prediction_step_bs),
            "max_reassignment_run_frames": int(max_reassignment_run_frames),
            "attribution_margin_threshold": float(attribution_margin_threshold),
        },
        "segment_boundaries": segment_boundaries,
        "whole_chain_claim": {
            "status": "rejected" if unsafe_whole_chain_frames else "coherent_observations_only",
            "reason": (
                "proximal/distal disagreement or bounded-run quarantine prevents a "
                "coherent whole-chain claim"
                if unsafe_whole_chain_frames else
                "all represented observed chains use coherent component assignments"
            ),
        },
        "identity_note": (
            "assignment transitions count backend-label reassignment, not measured physical ID errors"
        ),
        "anatomy_note": (
            "anatomical posterior is an uncalibrated joint/complementary research score and "
            "abstains below margin; image-x sign unused"
        ),
        "evidence_note": (
            "missing/partial chains are explicit; no interpolation or contact event is produced"
        ),
    }
    return LegTrackingResult(
        frames=frames,
        records=records,
        summary=summary,
        attributions=attributions,
    )


__all__ = [
    "ANATOMICAL_SIDES",
    "JOINTS",
    "MEDIAPIPE_LEG_INDICES",
    "LegChain",
    "LegTrackingResult",
    "SegmentAttribution",
    "TrackedLeg",
    "TrackedLegFrame",
    "track_leg_chains",
]
