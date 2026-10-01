"""Selective stable-offset workflow: response evidence, offset estimation, boundaries and QC.

Callers supply explicit policy and revision arguments; no default policy is supplied."""
from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
for folder in ("step_corrector_v1", "step_corrector_c1", "step_corrector_d1"):
    sys.path.insert(0, str(HERE.parent / folder))
from step_corrector_d import (  # amplitude helpers; response timing is evaluated by D.1
    OUTER_S, closure_reason, combine_A, estimate_a_off_d, estimate_a_on_d,
)
from timing_windows import DAY_S, MAX_TRANSITION_S, ban_windows, node_window

PATCH = "v2.1-public-release"
NO_EXPAND = True


@dataclass(frozen=True)
class D1Policy:
    """Explicit numerical criteria supplied by the caller."""
    policy_id: str
    evidence_s: int
    min_peers: int
    consensus_span_s: int
    consensus_fraction: None  # Retired percentage gate; explicit None prevents silent reuse.
    target_tolerance_s: int
    change_min_gain: float
    change_min_snr: float
    noise_floor_nt: float
    state_window_s: int
    min_plateau_s: int
    max_transient_s: int
    max_slope_nt_s: float
    max_drift_nt: float
    noise_ratio: float
    reversal_fraction: float
    max_join_nt: float
    max_residual_nt: float
    max_boundary_difference_nt: float

    def __post_init__(self):
        if not self.policy_id.strip():
            raise ValueError("policy_id is required")
        integer_fields = ("evidence_s", "min_peers", "consensus_span_s",
                          "target_tolerance_s", "state_window_s",
                          "min_plateau_s", "max_transient_s")
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.evidence_s < 4 or self.state_window_s < 4:
            raise ValueError("insufficient samples/independent peers")
        if self.min_peers != 1 or self.consensus_fraction is not None:
            raise ValueError("Line support requires one independent station; percentage gate is retired")
        if self.state_window_s > self.evidence_s + 1 or self.min_plateau_s < self.state_window_s:
            raise ValueError("background/plateau must support the state evidence window")
        for name in ("change_min_gain", "reversal_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"invalid {name}")
        for name in ("change_min_snr", "noise_floor_nt", "max_slope_nt_s",
                     "max_drift_nt", "noise_ratio", "max_join_nt",
                     "max_residual_nt", "max_boundary_difference_nt"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"invalid {name}")


# Callers must supply an explicit policy; keep the default unset.
FROZEN_POLICY: D1Policy | None = None


@dataclass(frozen=True)
class LocalState:
    morphology: str
    on_node: int
    off_operation_node: int
    off_end: int
    on_stable: int | None = None
    peak: int | None = None


@dataclass
class StepD1Result:
    corrected: np.ndarray
    flags: np.ndarray
    decision: str = "abstain"
    reason: str = "unverified"
    final_status: str = "abstain"
    morphology: str = "uncertain"
    transition_mode: str = "none"
    operation_confirmed: bool = False
    on_node_final: int | None = None
    on_stable_final: int | None = None
    off_node_final: int | None = None
    off_stable_final: int | None = None
    off_end_final: int | None = None
    peak_final: int | None = None
    on_bg_end: int | None = None
    on_plateau_start: int | None = None
    off_plateau_end: int | None = None
    off_bg_start: int | None = None
    line_consensus_on: float | None = None
    line_consensus_off: float | None = None
    n_peer_on: int = 0
    n_peer_off: int = 0
    line_on_supported: bool = False
    line_off_supported: bool = False
    A_on_nT: float = math.nan
    A_off_nT: float = math.nan
    A_star_nT: float = math.nan
    A_used_nT: float = math.nan
    closure_gap_nT: float = math.nan
    evidence: dict = field(default_factory=dict)

    def summary(self) -> dict:
        payload = {k: v for k, v in self.__dict__.items()
                   if k not in {"corrected", "flags"}}
        # Unavailable amplitudes/times are JSON null, never invented catalog points.
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [clean(v) for v in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return clean(payload)


def _same(a, b):
    return (np.isnan(a) & np.isnan(b)) | (a == b)


def outside_changed(raw, corr, on_node, off_stable):
    return int(np.count_nonzero(~_same(raw[:on_node], corr[:on_node])) +
               np.count_nonzero(~_same(raw[off_stable + 1:], corr[off_stable + 1:])))


def assert_write_lock(raw, corr, flags, on_node, off_stable):
    if not (0 <= on_node <= off_stable < len(raw)):
        raise RuntimeError("invalid write support")
    if outside_changed(raw, corr, on_node, off_stable):
        raise RuntimeError("D.1 changed raw outside locked support")
    if np.any(flags[:on_node] != 0) or np.any(flags[off_stable + 1:] != 0):
        raise RuntimeError("D.1 flagged outside locked support")


def _overlaps(lo, hi, bans):
    return any(lo < end and start < hi for start, end in bans)


def _fit(y):
    x = np.arange(len(y), dtype=float)
    origin = float(np.mean(y))
    centered = y - origin
    coeff = np.linalg.lstsq(np.column_stack((np.ones(len(y)), x)), centered, rcond=None)[0]
    residual = centered - coeff[0] - coeff[1] * x
    return float(coeff[0] + origin), float(coeff[1]), float(np.sqrt(np.mean(residual ** 2)))


def _noise(y, p):
    d = np.diff(y)
    return max(p.noise_floor_nt, float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)))


# 1. Operation/time confirmation. Peers locate their own change; target verifies.
def _change_point(raw, lo, hi, p, bans, *, all_nodes=False):
    left, right = max(0, lo - p.evidence_s), min(len(raw), hi + p.evidence_s)
    for a, b in bans:
        if b <= lo:
            left = max(left, b)
        if a >= hi:
            right = min(right, a)
    lo, hi = max(lo, left + p.evidence_s), min(hi, right - p.evidence_s)
    if hi <= lo:
        return None
    if _overlaps(left, right, bans) or not np.isfinite(raw[left:right]).all():
        return None
    y = raw[left:right]
    y = y - np.median(y)
    x = np.arange(left, right, dtype=float)
    base = np.column_stack((np.ones(len(x)), x - left))
    residual = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
    sse0 = float(residual @ residual)
    noise = _noise(raw[left:lo + 1], p)
    candidates = []
    for node in range(lo, hi):
        # Local segmented regression locates the START, not the middle of a ramp.
        # This basis is timing evidence only; it is never subtracted from raw.
        for end in range(node + 1, min(right, node + MAX_TRANSITION_S) + 1):
            basis = np.clip((x - node) / (end - node), 0, 1)
            design = np.column_stack((base, basis))
            coeff = np.linalg.lstsq(design, y, rcond=None)[0]
            resid = y - design @ coeff
            sse = float(resid @ resid)
            effect = float(coeff[2])
            gain = (sse0 - sse) / max(sse0, np.finfo(float).eps)
            if gain >= p.change_min_gain and abs(effect) >= p.change_min_snr * noise:
                candidates.append((sse, end, node, effect, gain))
    if not candidates:
        return None
    if all_nodes:
        # Retain the best unchanged fit per node for explicit timing revisions.
        by_node = {}
        for sse, end, node, effect, gain in candidates:
            if node not in by_node or (sse, end) < by_node[node][:2]:
                by_node[node] = (sse, end, effect, gain)
        return [{"node": node, "effect_nt": c[2], "gain": c[3],
                 "noise_nt": noise, "sse": c[0], "ramp_end": c[1]}
                for node, c in sorted(by_node.items())]
    best_sse = min(c[0] for c in candidates)
    # Numerically equivalent fits choose the earliest supported corner.
    near = [c for c in candidates if c[0] <= best_sse + np.finfo(float).eps * max(1, sse0) * 100]
    sse, end, node, effect, gain = min(near, key=lambda c: (c[2], c[1]))
    return {"node": node, "effect_nt": effect, "gain": gain, "noise_nt": noise, "sse": sse}


def _consensus(hits, n_peers, p):
    ordered = sorted(hits.items(), key=lambda item: item[1]["node"])
    clusters = set()
    for i, (_, hit) in enumerate(ordered):
        members = tuple(sorted(s for s, h in ordered[i:]
                               if h["node"] - hit["node"] <= p.consensus_span_s))
        clusters.add(members)
    maximum = max((len(c) for c in clusters), default=0)
    winners = [c for c in clusters if len(c) == maximum]
    # Only this station-count/percentage gate changes. Anchor clustering is unchanged.
    if maximum < 1:
        return None, (), "line_consensus_insufficient"
    if len(winners) != 1:
        return None, (), "line_consensus_ambiguous"
    members = winners[0]
    return float(np.median([hits[s]["node"] for s in members])), members, "ok"


def confirm_operation_time(raw, catalog_s, peers, p, bans, *, timing_revision=None, nonzero_radius_s=0,
                           line_veto_audit=False, catalog_end_s=None, remaining_edge_window=None):
    lo, hi = node_window(catalog_s, nonzero_radius_s=nonzero_radius_s)
    if catalog_end_s is not None:
        if (catalog_s % 60 or nonzero_radius_s or isinstance(catalog_end_s, bool)
                or not isinstance(catalog_end_s, (int, np.integer))
                or not catalog_s < catalog_end_s <= DAY_S):
            raise ValueError("invalid explicit catalogue edge interval")
        hi = int(catalog_end_s)
    if remaining_edge_window is not None:
        a, b = remaining_edge_window
        if (catalog_end_s is not None or nonzero_radius_s or a != hi
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (a, b))
                or not 0 <= a < b <= DAY_S):
            raise ValueError("fallback must use only the remaining catalogue edge interval")
        lo, hi = int(a), int(b)
    hits, missing, unavailable = {}, [], {}
    target_candidates = None
    if timing_revision == "target-first-dev01":
        target_candidates = _change_point(raw, lo, hi, p, bans, all_nodes=True) or []
    # Usability is per boundary: complete observations over the detector's existing
    # evidence interval. A finite series with no detected response remains usable.
    left, right = max(0, lo - p.evidence_s), min(len(raw), hi + p.evidence_s)
    for a, b in bans:
        if b <= lo:
            left = max(left, b)
        if a >= hi:
            right = min(right, a)
    for station, series in peers.items():
        arr = np.asarray(series, dtype=float)
        if arr.shape != raw.shape or not np.isfinite(arr[left:right]).all():
            unavailable[station] = "invalid_day_shape" if arr.shape != raw.shape else "missing_or_nonfinite_boundary_samples"
            continue
        hit = _change_point(arr, lo, hi, p, bans)
        if hit is None:
            missing.append(station)
        else:
            hits[station] = hit
    available_count = len(peers) - len(unavailable)
    consensus, members, reason = _consensus(hits, available_count, p)
    evidence = {"catalog_s": catalog_s, "legal_window": [lo, hi],
                "peer_hits": hits, "peer_unverified": missing,
                "peer_unavailable": unavailable, "available_peer_count": available_count,
                "observation_window": [left, right], "cross_station_status": None,
                "supporting_stations": [], "peer_minus_target_s": {},
                "inlier_stations": list(members), "consensus_s": consensus,
                "confirmed": False, "reason": reason}
    if available_count == 0:
        evidence["cross_station_status"] = "cross_station_unavailable"
        evidence["cross_station_skipped"] = True
        # The explicitly requested zero-reference exception uses the SAME local
        # locator and existing catalog window; it does not fabricate a line time.
        hit = _change_point(raw, lo, hi, p, bans)
        if hit is None:
            evidence["reason"] = "local_response_unverified"
            return None, evidence
        evidence.update(confirmed=True, reason="cross_station_unavailable_skipped", target_hit=hit)
        return hit["node"], evidence
    if timing_revision == "target-first-dev01" or (
            timing_revision == "tie-target-dev01" and reason == "line_consensus_ambiguous"):
        return _target_admission(raw, lo, hi, hits, p, bans, evidence,
                                 timing_revision, target_candidates)
    if consensus is None:
        evidence["cross_station_status"] = "cross_station_unsupported"
        evidence["target_evaluation"] = "not_reached_no_line_anchor"
        if line_veto_audit and reason == "line_consensus_insufficient":
            # Shadow counterfactual only: reuse the existing no-reference local
            # search, retaining unsupported line evidence and the original window.
            evidence["line_veto_bypassed"] = reason
            hit = _change_point(raw, lo, hi, p, bans)
            evidence["target_evaluation"] = "shadow_existing_catalog_locator_without_line_anchor"
            if hit is None:
                evidence["reason"] = "local_response_unverified"
                return None, evidence
            evidence.update(confirmed=True, reason="line_veto_shadow_local_only", target_hit=hit)
            return hit["node"], evidence
        return None, evidence
    near_lo = max(lo, math.ceil(consensus - p.target_tolerance_s))
    near_hi = min(hi, math.floor(consensus + p.target_tolerance_s) + 1)
    hit = _change_point(raw, near_lo, near_hi, p, bans)
    if hit is None:
        evidence["reason"] = "local_response_unverified"
        evidence["target_evaluation"] = "searched_existing_consensus_band_no_hit"
        return None, evidence
    offsets = {s: h["node"] - hit["node"] for s, h in hits.items()}
    supporting = [s for s, delta in offsets.items() if abs(delta) <= p.target_tolerance_s]
    evidence.update(target_hit=hit, peer_minus_target_s=offsets, supporting_stations=supporting)
    if not supporting:
        evidence.update(cross_station_status="cross_station_unsupported", reason="no_peer_time_match")
        return None, evidence
    evidence["cross_station_status"] = "cross_station_supported"
    evidence.update(confirmed=True, reason="ok", target_hit=hit)
    return hit["node"], evidence


def _target_admission(raw, lo, hi, hits, p, bans, evidence, revision, target_candidates):
    """Explicit timing revision with no amplitude, endpoint or QC feedback."""
    proposed = []
    groups = []
    if revision == "tie-target-dev01":
        ordered = sorted(hits.items(), key=lambda item: item[1]["node"])
        clusters = {tuple(sorted(s for s, h in ordered[i:]
                                if h["node"] - hit["node"] <= p.consensus_span_s))
                    for i, (_, hit) in enumerate(ordered)}
        size = max(map(len, clusters), default=0)
        for members in sorted(c for c in clusters if len(c) == size):
            center = float(np.median([hits[s]["node"] for s in members]))
            a = max(lo, math.ceil(center - p.target_tolerance_s))
            b = min(hi, math.floor(center + p.target_tolerance_s) + 1)
            hit = _change_point(raw, a, b, p, bans)
            groups.append({"members": list(members), "center": center,
                           "target_window": [a, b], "target_hit": hit})
            if hit is not None:
                proposed.append(hit)
    else:
        proposed = target_candidates
    evaluated = []
    for hit in proposed:
        offsets = {s: h["node"] - hit["node"] for s, h in hits.items()}
        supporting = sorted(s for s, delta in offsets.items()
                            if abs(delta) <= p.target_tolerance_s)
        distance = min((abs(v) for v in offsets.values()), default=math.inf)
        evaluated.append({"hit": hit, "supporting_stations": supporting,
                          "nearest_peer_distance_s": distance,
                          "score": [hit["gain"], abs(hit["effect_nt"]) / hit["noise_nt"],
                                    len(supporting), -distance, -hit["node"]]})
    evidence.update(timing_revision=revision, legacy_line_reason=evidence["reason"],
                    target_evaluation="target_candidates_compared",
                    tied_groups=groups, target_candidates=evaluated)
    eligible = [c for c in evaluated if c["supporting_stations"]]
    if not eligible:
        evidence.update(confirmed=False, cross_station_status="cross_station_unsupported",
                        reason="no_peer_time_match" if evaluated else "local_response_unverified")
        return None, evidence
    chosen = max(eligible, key=lambda c: c["score"])
    hit, supporting = chosen["hit"], chosen["supporting_stations"]
    evidence.update(target_hit=hit, supporting_stations=supporting,
                    peer_minus_target_s={s: h["node"] - hit["node"] for s, h in hits.items()},
                    inlier_stations=supporting,
                    consensus_s=float(np.median([hits[s]["node"] for s in supporting])),
                    consensus_role="support_summary_after_target_selection",
                    confirmed=True, cross_station_status="cross_station_supported", reason="ok")
    return hit["node"], evidence


# 2. Local state. Only target data determines stable and plateau presence.
def _quiet(y, background, p):
    if len(y) < p.state_window_s or not np.isfinite(y).all():
        return False
    _, slope, rms = _fit(y)
    _, bg_slope, bg_rms = _fit(background)
    detrended = y - bg_slope * np.arange(len(y))
    middle = len(y) // 2
    drift = abs(float(np.mean(detrended[middle:]) - np.mean(detrended[:middle])))
    return (abs(slope - bg_slope) <= p.max_slope_nt_s
            and drift <= p.max_drift_nt
            and rms <= p.noise_ratio * max(bg_rms, p.noise_floor_nt))


def find_local_stable(raw, node, hard_end, background, p, bans, *, candidate_audit=False):
    fit_end = min(hard_end, node + MAX_TRANSITION_S + p.state_window_s + 1)
    if fit_end - node <= p.state_window_s or _overlaps(node, fit_end, bans):
        return None, {"reason": "stable_evidence_window_unavailable"}
    y = raw[node:fit_end]
    if not np.isfinite(y).all():
        return None, {"reason": "stable_missing_data"}
    y = y - np.median(y)
    x = np.arange(len(y), dtype=float)
    candidates = []
    for t in range(node + 1, min(node + MAX_TRANSITION_S + 1, fit_end - p.state_window_s + 1)):
        tail = raw[t:fit_end]
        if not candidate_audit and not _quiet(tail, background, p):
            continue
        hinge = np.maximum(x - (t - node), 0)
        design = np.column_stack((np.ones(len(y)), x, hinge))
        error = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        candidates.append((float(error @ error), t))
    if not candidates:
        return None, {"reason": "ongoing_drift_or_unstable"}
    best = min(sse for sse, _ in candidates)
    tolerance = np.finfo(float).eps * max(1, float(y @ y)) * 100
    stable = min(t for sse, t in candidates if sse <= best + tolerance)
    return stable, {"reason": "ok", "fit_end": fit_end, "segmented_sse": best,
                    "acceptable_candidates": len(candidates),
                    "method": "segmented_regression_and_future_slope_drift_noise"}


def determine_local_state(raw, on, off, p, bans, *, candidate_audit=False):
    evidence = {}
    if _overlaps(on, off + 1, bans):
        return None, {"reason": "neighbor_inside_event"}
    before = raw[on - p.evidence_s:on + 1]
    if on < p.evidence_s or not np.isfinite(before).all() or _overlaps(on - p.evidence_s, on + 1, bans):
        return None, {"reason": "background_before_unverified"}
    # Off confirmation anchors the search. Evidence never crosses a neighbor.
    future_bans = [a for a, b in bans if a > off]
    hard_end = min([len(raw)] + future_bans)
    end, evidence["off_stable"] = find_local_stable(raw, off, hard_end, before, p, bans, candidate_audit=candidate_audit)
    if end is None:
        return None, {**evidence, "reason": "off_end_unverified"}
    if end + p.evidence_s >= hard_end:
        return None, {**evidence, "reason": "background_after_unverified"}
    after = raw[end:end + p.evidence_s + 1]
    if not np.isfinite(after).all() or (not candidate_audit and not _quiet(after, before, p)):
        return None, {**evidence, "reason": "background_after_unverified"}
    if _overlaps(on, end + 1, bans) or not np.isfinite(raw[on:end + 1]).all():
        return None, {**evidence, "reason": "neighbor_or_missing_data"}
    slope_before, slope_after = _fit(before)[1], _fit(after)[1]
    if not candidate_audit and (abs(raw[on] - (_fit(before)[0] + slope_before * (len(before) - 1))) > p.max_join_nt
            or abs(raw[end] - _fit(after)[0]) > p.max_join_nt):
        return None, {**evidence, "reason": "background_endpoint_unreliable"}
    stable, evidence["on_stable"] = find_local_stable(raw, on, off + 1, before, p, bans, candidate_audit=candidate_audit)
    if stable is not None and off - stable + 1 >= p.min_plateau_s:
        if not candidate_audit and not _quiet(raw[stable:off + 1], before, p):
            return None, {**evidence, "reason": "plateau_uncertain"}
        return LocalState("plateau", on, off, end, stable), {**evidence, "reason": "ok"}
    # Confirm one excursion with opposing monotone limbs. Failure is uncertainty.
    affected = raw[on:end + 1]
    baseline = np.linspace(raw[on], raw[end], len(affected))
    excursion = affected - baseline
    peak = int(np.argmax(np.abs(excursion)))
    amplitude = float(excursion[peak])
    noise = max(_noise(before, p), _noise(after, p))
    if not 0 < peak < len(excursion) - 1 or abs(amplitude) < p.change_min_snr * noise:
        return None, {**evidence, "reason": "plateau_presence_uncertain"}
    signed = excursion * np.sign(amplitude)
    reverse = (np.maximum(-np.diff(signed[:peak + 1]), 0).sum()
               + np.maximum(np.diff(signed[peak:]), 0).sum())
    evidence["transient"] = {"peak_s": on + peak, "reverse_fraction": float(reverse / abs(amplitude)),
                             "duration_s": end - on, "A_star_applicable": False}
    if reverse > p.reversal_fraction * abs(amplitude):
        return None, {**evidence, "reason": "complex_or_uncertain_transient"}
    state = LocalState("transient", on, off, end, peak=on + peak)
    if end - on > p.max_transient_s:
        return state, {**evidence, "reason": "transient_too_long"}
    return state, {**evidence, "reason": "ok"}


def refine_transition_membership(raw, state, a_star, p, bans, *, off_revision=None, on_revision=None,
                                 selection_revision=None, on_background_node_guard=False):
    """Refine sample membership without re-estimating A or operation times.

    Local reference levels, including full-A platform observations, determine
    sample membership. Existing evidence/persistence lengths and noise multiplier
    are explicitly reused; no slope gate or event-specific shift is introduced.
    """
    if state.morphology != "plateau" or not math.isfinite(a_star) or a_star == 0:
        return None, {"reason": "endpoint_revision_requires_plateau_and_A"}
    on, off = state.on_node, state.off_operation_node
    w, hold, cap = p.evidence_s, p.state_window_s, MAX_TRANSITION_S
    windows = {"on_background": (on - w, on),
               "on_platform": (on + cap, on + cap + w),
               "off_platform": (off - w, off),
               "off_background": (off + cap, off + cap + w)}
    nominal_windows = windows.copy()
    if windows["on_platform"][1] > windows["off_platform"][0]:
        midpoint = (on + off) // 2
        windows["on_platform"] = (max(state.on_stable, midpoint - w), midpoint)
        windows["off_platform"] = (max(midpoint, off - w), off)
    lo, hi = windows["off_background"]
    # Keep the existing left anchor; never borrow samples after the neighbor.
    for ban_start, ban_end in sorted(bans):
        if lo < ban_end and ban_start < hi:
            hi = max(lo, ban_start)
            break
    windows["off_background"] = (lo, hi)
    evidence = {"revision": "sample-state-dev03", "fixed_A": a_star,
                "reference_window_revision": "reference-window-dev01",
                "nominal_reference_windows": nominal_windows,
                "reference_windows": windows, "persistence_samples": hold,
                "reference_sample_counts": {name: max(0, hi - lo) for name, (lo, hi) in windows.items()},
                "minimum_reference_samples": hold,
                "noise_multiplier": p.change_min_snr, "references": {},
                "arrival_max_offset_s": cap + w - hold}
    refs, observations = {}, {}
    for name, (lo, hi) in windows.items():
        if hi - lo < hold:
            return None, {**evidence, "reason": "endpoint_reference_too_short", "missing": name}
        if lo < 0 or hi > len(raw) or _overlaps(lo, hi, bans) or not np.isfinite(raw[lo:hi]).all():
            return None, {**evidence, "reason": "endpoint_reference_unavailable", "missing": name}
        # Both sides share a background fit: A must not cancel algebraically in
        # an independent platform-only fit extrapolated across the whole ramp.
        y = raw[lo:hi] - (a_star if "platform" in name else 0)
        observations[name] = (np.arange(lo, hi, dtype=float), y, _noise(y, p))
    for prefix in ("on", "off"):
        names = (prefix + "_background", prefix + "_platform")
        x = np.concatenate([observations[name][0] for name in names])
        y = np.concatenate([observations[name][1] for name in names])
        origin = int(np.min(x))
        x = x - origin
        i, j = np.triu_indices(len(y), 1)
        slope = float(np.median((y[j] - y[i]) / (x[j] - x[i])))
        intercept = float(np.median(y - slope * x))
        band = p.change_min_snr * max(observations[name][2] for name in names)
        for name in names:
            offset = a_star if "platform" in name else 0
            refs[name] = (origin, intercept + offset, slope, band)
            evidence["references"][name] = {"origin": origin, "level": intercept + offset,
                                             "slope": slope, "band_nt": band}

    def errors(name, times):
        origin, level, slope, band = refs[name]
        return (raw[times] - (level + slope * (times - origin))) / band

    def departure(name, operation, limit):
        lo = max(0, operation - w)
        hi = min(limit, operation + cap)
        for t in range(lo, hi):
            future = np.arange(t, t + hold)
            if future[-1] >= len(raw) or _overlaps(t, t + hold, bans):
                continue
            e = errors(name, future)
            if abs(e[0]) > 1 and abs(float(np.median(e))) > 1:
                anchor = t - 1
                history = np.arange(anchor - hold + 1, anchor + 1)
                if history[0] >= 0 and abs(errors(name, np.array([anchor]))[0]) <= 1:
                    if float(np.sqrt(np.mean(errors(name, history) ** 2))) <= 1:
                        return anchor
        return None

    def arrival(name, operation, start):
        if start is None:
            return None
        # Endpoints are not operation timestamps. Retain full persistence inside
        # the already loaded reference extent, rather than truncate at its start.
        for t in range(max(operation - w, start + 1), operation + cap + w - hold + 1):
            times = np.arange(t, t + hold)
            if times[-1] >= len(raw) or _overlaps(t, t + hold, bans):
                continue
            e = errors(name, times)
            if abs(e[0]) <= 1 and float(np.sqrt(np.mean(e ** 2))) <= 1:
                return t
        return None

    main_edges = selection_revision == "main-edge-dev01"
    if main_edges:
        evidence["selection_revision"] = selection_revision
        evidence["main_edge"] = {}

        def main_edge(prefix, operation):
            # Locate the main change before checking level evidence. The fitted
            # local amplitude is never passed to the correction or A estimator.
            names = (prefix + "_background", prefix + "_platform")
            lo = min(windows[name][0] for name in names)
            hi = max(windows[name][1] for name in names)
            item = {"observation_window": [lo, hi], "pre_samples_min": 3,
                    "post_samples_min": hold, "method": "three_piece_linear_main_change"}
            evidence["main_edge"][prefix] = item
            if _overlaps(lo, hi, bans) or not np.isfinite(raw[lo:hi]).all():
                item["reason"] = "main_edge_observations_unavailable"
                return None, None
            x = np.arange(lo, hi, dtype=float)
            y = raw[lo:hi] - np.median(raw[lo:hi])
            candidates = []
            starts = range(max(lo + 3, operation - w), min(operation + cap, hi - hold))
            for start in starts:
                end_max = min(start + cap, operation + cap + w - hold, hi - hold)
                for end in range(start + 1, end_max + 1):
                    ramp = np.clip((x - start) / (end - start), 0, 1)
                    design = np.column_stack((np.ones(len(x)), x - lo, ramp, np.maximum(x - end, 0)))
                    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
                    residual = y - design @ coefficients
                    candidates.append((float(residual @ residual), start, end, coefficients))
            if not candidates:
                item["reason"] = "main_edge_bracket_empty"
                return None, None
            best = min(c[0] for c in candidates)
            tol = np.finfo(float).eps * max(1, float(y @ y)) * 100
            score, start, end, coefficients = min(
                (c for c in candidates if c[0] <= best + tol),
                key=lambda c: (abs(c[1] - operation), c[2] - c[1], c[1]))
            local_a = float(coefficients[2]) * (1 if prefix == "on" else -1)
            closure = closure_reason(a_star, local_a)
            item.update(start=start, end=end, sse=score, local_offset_nt=local_a,
                        slope_before_nt_s=float(coefficients[1]),
                        slope_after_nt_s=float(coefficients[1] + coefficients[3]),
                        closure=closure, candidate_count=len(candidates),
                        reason="ok" if closure == "ok" else "main_edge_" + closure)
            if closure != "ok":
                return None, None
            return start, end

        bg_on, platform_on = main_edge("on", on)
        main_off = main_edge("off", off)
    else:
        bg_on = departure("on_background", on, off)
        platform_on = arrival("on_platform", on, bg_on)
    if on_revision == "bounded-contraction-dev01" and bg_on is not None and platform_on is not None:
        evidence["on_q_contraction"] = []
        while bg_on < platform_on:
            edge = raw[bg_on:platform_on + 1]
            q = (edge - np.linspace(edge[0], edge[-1] - a_star, len(edge))) / a_star
            tol = np.finfo(float).eps * max(1, np.max(np.abs(edge))) / abs(a_star) * 8
            evidence["on_q_contraction"].append({"background": bg_on, "platform": platform_on,
                                                "min": float(q.min()), "max": float(q.max()), "roundoff_tolerance": float(tol)})
            below, above = np.flatnonzero(q < -tol), np.flatnonzero(q > 1 + tol)
            if not len(below) and not len(above):
                break
            if len(below):
                proposed_bg = bg_on + int(below[-1])
                if on_background_node_guard and proposed_bg > on:
                    evidence["on_background_node_guard"] = {
                        "revision": "on-background-node-guard-shadow01", "detected_on_node": on,
                        "retained_background": bg_on, "blocked_background": proposed_bg,
                        "retained_platform": platform_on, "initial_background_already_late": bg_on > on,
                        "action": "retain_pair_before_forbidden_q_move; original_saved_proposal_and_QC"}
                    break
                bg_on = proposed_bg
            else:
                platform_on = bg_on + int(np.flatnonzero(q >= 1 - tol)[0])
        evidence["on_endpoint_revision"] = on_revision
    if main_edges:
        platform_off, bg_off = main_off
    else:
        platform_off = departure("off_platform", off, len(raw) - hold)
        bg_off = arrival("off_background", off, platform_off)
    if not main_edges and (off_revision == "local-edge-dev01" or (off_revision == "supported-fallback-dev01" and platform_off is None)):
        platform_off, bg_off = None, None
        evidence["off_local_edge"] = {"pre_samples": 3, "post_samples": hold, "tested": []}
        local_lo = off - p.target_tolerance_s if off_revision == "supported-fallback-dev01" else off - w
        local_hi = off + p.target_tolerance_s + 1 if off_revision == "supported-fallback-dev01" else off + cap
        evidence["off_local_edge"]["candidate_window"] = [local_lo, local_hi]
        for t in range(max(3, local_lo), min(len(raw) - hold + 1, local_hi)):
            pre, post = raw[t - 3:t], raw[t:t + hold]
            if _overlaps(t - 3, t + hold, bans) or not np.isfinite(raw[t - 3:t + hold]).all():
                continue
            before, after = float(np.median(pre)), float(np.median(post))
            if -np.sign(a_star) * (raw[t] - before) <= refs["off_platform"][3]:
                continue
            edge_a = before - after
            closure = closure_reason(a_star, edge_a)
            check = {"candidate": t, "pre_level": before, "post_level": after,
                     "observed_offset_nt": edge_a, "closure": closure}
            evidence["off_local_edge"]["tested"].append(check)
            if closure != "ok":
                continue
            background = arrival("off_background", off, t - 1)
            check["background_arrival"] = background
            if background is not None:
                platform_off, bg_off = t - 1, background
                evidence["off_local_edge"]["accepted"] = check
                break
        evidence["off_departure_method"] = "local_direction_amplitude_closure_and_post_background"
    if off_revision in ("zero-crossing-dev02", "supported-fallback-dev01") and platform_off is not None and bg_off is not None:
        # This opt-in replaces only OFF arrival, never clips I or changes A.
        previous_end = bg_off
        previous_line = np.linspace(raw[platform_off] - a_star, raw[previous_end], previous_end - platform_off + 1)
        previous_q = (raw[platform_off:previous_end + 1] - previous_line) / a_star
        tol = np.finfo(float).eps * max(1, np.max(np.abs(raw[platform_off:previous_end + 1]))) / abs(a_star) * 8
        needs_zero = off_revision == "zero-crossing-dev02" or previous_q.min() < -tol
        # The actual candidate I, not an independently fitted B, defines zero.
        # Beyond its old support the candidate already equals raw, hence q=0.
        extended_q = np.pad(previous_q, (0, hold), constant_values=0)
        bg_off = None
        for t in (range(platform_off + 1, previous_end + 1) if needs_zero else (previous_end,)):
            times = np.arange(t, t + hold)
            if times[-1] >= len(raw) or _overlaps(t, t + hold, bans):
                continue
            signed = extended_q[t - platform_off:t - platform_off + hold]
            if np.isfinite(signed).all() and signed[0] <= 0 and np.median(signed) <= 0:
                bg_off = t
                break
        evidence["off_arrival_method"] = "first_actual_candidate_q_zero_crossing_and_future_median"
        evidence["off_previous_endpoint"] = previous_end
        if bg_off is not None:
            corrected_edge = np.linspace(raw[platform_off] - a_star, raw[bg_off], bg_off - platform_off + 1)
            q = (raw[platform_off:bg_off + 1] - corrected_edge) / a_star
            tol = np.finfo(float).eps * max(1, np.max(np.abs(raw[platform_off:bg_off + 1]))) / abs(a_star) * 8
            evidence["off_q_check"] = {"min": float(q.min()), "max": float(q.max()), "roundoff_tolerance": float(tol), "candidate_endpoint": bg_off}
            if (q.min() < -tol or q.max() > 1 + tol) and off_revision == "zero-crossing-dev02":
                return None, {**evidence, "reason": "off_linear_q_out_of_bounds"}
    evidence["off_endpoint_revision"] = off_revision
    endpoints = dict(on_bg_end=bg_on, on_plateau_start=platform_on,
                     off_plateau_end=platform_off, off_bg_start=bg_off)
    evidence["endpoints"] = endpoints
    # The endpoint is the START of the forward validation interval, never its
    # last sample. Store both explicitly so plots cannot confuse the two.
    evidence["arrival_validation_windows"] = {
        name: {"start": t, "end_inclusive": t + hold - 1, "samples": hold}
        for name, t in (("on_plateau_start", platform_on), ("off_bg_start", bg_off))
        if t is not None
        and not main_edges
        and not (name == "on_plateau_start" and len(evidence.get("on_q_contraction", [])) > 1)
    }
    if any(t is None for t in endpoints.values()):
        return None, {**evidence, "reason": "endpoint_membership_unresolved",
                      "missing": [k for k, v in endpoints.items() if v is None]}
    if not bg_on < platform_on <= platform_off < bg_off:
        return None, {**evidence, "reason": "endpoint_order_invalid"}
    if platform_off - platform_on + 1 < p.min_plateau_s:
        return None, {**evidence, "reason": "endpoint_plateau_too_short"}
    if (bg_on < w or bg_off + w + 1 > len(raw)
            or _overlaps(bg_on - w, bg_off + w + 1, bans)
            or not np.isfinite(raw[bg_on - w:bg_off + w + 1]).all()):
        return None, {**evidence, "reason": "endpoint_support_or_QC_shoulders_unavailable"}
    final_state = LocalState("plateau", bg_on, platform_off, bg_off, platform_on)
    if off_revision == "supported-fallback-dev01":
        evidence["transition_q_check"] = {}
        for name, lo, hi, left_offset, right_offset in (
                ("ON", bg_on, platform_on, 0, a_star),
                ("OFF", platform_off, bg_off, a_star, 0)):
            edge = raw[lo:hi + 1]
            q = (edge - np.linspace(edge[0] - left_offset, edge[-1] - right_offset, len(edge))) / a_star
            tol = np.finfo(float).eps * max(1, np.max(np.abs(edge))) / abs(a_star) * 8
            evidence["transition_q_check"][name] = {"min": float(q.min()), "max": float(q.max()),
                                                    "passed": bool(q.min() >= -tol and q.max() <= 1 + tol), "roundoff_tolerance": float(tol)}
        # OFF upper excursions are diagnostic: I includes natural variation
        # around the unchanged corrected-space connector. Never clip the wave.
        evidence["transition_q_check"]["OFF"]["upper_bound_role"] = "diagnostic_only"
        on_q, off_q = evidence["transition_q_check"]["ON"], evidence["transition_q_check"]["OFF"]
        if not on_q["passed"] or off_q["min"] < -off_q["roundoff_tolerance"]:
            return None, {**evidence, "candidate_state_before_q_gate": asdict(final_state), "reason": "transition_q_unresolved"}
    return final_state, {**evidence, "reason": "ok"}


# 3. Fixed correction. No time search and no post-hoc spike/shape patch.
def _side_pair_fallback(raw, state, a_star, p, bans, endpoint_evidence, *, missing_only=False):
    """Keep complete validated sides; never discard one side to rescue the other."""
    fine = endpoint_evidence.get("endpoints", {})
    coarse = {"ON": (state.on_node - 1, state.on_stable),
              "OFF": (state.off_operation_node - 1, state.off_end)}
    chosen, audit = {}, {"revision": "missing-pair-dev02" if missing_only else "side-pair-dev01", "sides": {}}
    for side, keys in (("ON", ("on_bg_end", "on_plateau_start")),
                       ("OFF", ("off_plateau_end", "off_bg_start"))):
        pair = tuple(fine.get(k) for k in keys)
        left, right = pair
        reason = "ok"
        q_check = None
        if left is None or right is None:
            reason = "incomplete_refined_pair"
        elif not (0 <= left < right < len(raw)):
            reason = "invalid_refined_pair_order"
        elif (left < p.evidence_s or right + p.evidence_s >= len(raw)
              or _overlaps(left - p.evidence_s, right + p.evidence_s + 1, bans)
              or not np.isfinite(raw[left - p.evidence_s:right + p.evidence_s + 1]).all()):
            reason = "refined_pair_observations_unavailable"
        else:
            edge = raw[left:right + 1]
            offsets = (0, a_star) if side == "ON" else (a_star, 0)
            q = (edge - np.linspace(edge[0] - offsets[0], edge[-1] - offsets[1], len(edge))) / a_star
            tol = np.finfo(float).eps * max(1, np.max(np.abs(edge))) / abs(a_star) * 8
            valid = bool(q.min() >= -tol and (side == "OFF" or q.max() <= 1 + tol))
            q_check = {"min": float(q.min()), "max": float(q.max()), "roundoff_tolerance": float(tol),
                       "passed": valid, "upper_bound_role": "diagnostic_only" if side == "OFF" else "hard"}
            if not valid and not missing_only:
                reason = "refined_pair_q_failed"
        if missing_only and reason not in ("ok", "incomplete_refined_pair"):
            return None, {**audit, "reason": "complete_pair_invalid:" + reason, "side": side, "pair": pair}
        source = "refined" if reason == "ok" else "coarse"
        chosen[side] = pair if source == "refined" else coarse[side]
        audit["sides"][side] = {"source": source, "refined_pair": pair, "coarse_pair": coarse[side],
                                  "selected_pair": chosen[side], "reason": reason, "refined_q": q_check}
    left, stable = chosen["ON"]
    off, right = chosen["OFF"]
    if not (0 <= left < stable <= off < right < len(raw)):
        return None, {**audit, "reason": "side_pair_combined_order_invalid"}
    if off - stable + 1 < p.min_plateau_s:
        return None, {**audit, "reason": "side_pair_plateau_too_short"}
    if (left < p.evidence_s or right + p.evidence_s >= len(raw)
            or _overlaps(left - p.evidence_s, right + p.evidence_s + 1, bans)
            or not np.isfinite(raw[left - p.evidence_s:right + p.evidence_s + 1]).all()):
        return None, {**audit, "reason": "side_pair_combined_support_unavailable"}
    return LocalState("plateau", left, off, right, stable), {**audit, "reason": "ok"}


def apply_correction(raw, state, a_star):
    corr = raw.copy()
    flags = np.zeros(len(raw), dtype=np.int8)
    left, right = state.on_node, state.off_end
    if state.morphology == "transient":
        corr[left:right + 1] = np.linspace(raw[left], raw[right], right - left + 1)
        flags[left:right + 1] = 2
    else:
        stable, off = state.on_stable, state.off_operation_node
        if not (left < stable <= off < right) or not math.isfinite(a_star):
            raise ValueError("invalid locked plateau state")
        corr[stable:off + 1] = raw[stable:off + 1] - a_star
        flags[stable:off + 1] = 1
        corr[left:stable] = np.linspace(raw[left], corr[stable], stable - left + 1)[:-1]
        flags[left:stable] = 2
        corr[off + 1:right + 1] = np.linspace(corr[off], raw[right], right - off + 1)[1:]
        flags[off + 1:right + 1] = 2
    assert_write_lock(raw, corr, flags, left, right)
    return corr, flags


# 4. Final QC. Reject a candidate; never fix or relocate it here.
def quality_control(raw, corr, flags, state, a_star, p, bans):
    left, right = state.on_node, state.off_end
    failures = []
    metrics = {"outside_changed": outside_changed(raw, corr, left, right)}
    if metrics["outside_changed"] or np.any(flags[:left]) or np.any(flags[right + 1:]):
        failures.append("outside_support")
    if _overlaps(left, right + 1, bans):
        failures.append("neighbor_overlap")
    if not np.isfinite(corr[left:right + 1]).all():
        failures.append("nonfinite_correction")
    if not np.array_equal(np.isnan(raw), np.isnan(corr)):
        failures.append("missing_data_changed")
    if state.morphology == "plateau":
        stable, off = state.on_stable, state.off_operation_node
        metrics["on_flag2_samples"] = int(np.count_nonzero(flags[left:stable] == 2))
        metrics["off_flag2_samples"] = int(np.count_nonzero(flags[off + 1:right + 1] == 2))
        if not (np.all(flags[left:stable] == 2) and np.all(flags[stable:off + 1] == 1)
                and np.all(flags[off + 1:right + 1] == 2)
                and stable > left and right > off):
            failures.append("missing_transition_or_wrong_flags")
        if not np.array_equal(corr[stable:off + 1], raw[stable:off + 1] - a_star):
            failures.append("platform_not_single_A")
        boundaries = (left, stable, off, right)
        joins = [abs(float(corr[stable] - raw[left])), abs(float(corr[off] - raw[right]))]
        # Natural baseline slope is estimated from each nearby background independently.
        b1 = raw[left - p.evidence_s:left + 1]
        b2 = raw[right:right + p.evidence_s + 1]
        joins[0] = abs(float(corr[stable] - raw[left] - _fit(b1)[1] * (stable - left)))
        joins[1] = abs(float(corr[off] - raw[right] - _fit(b2)[1] * (off - right)))
        metrics["plateau_join_residual_nt"] = max(joins)
        if max(joins) > p.max_join_nt:
            failures.append("plateau_join_residual")
        plateau = corr[stable:off + 1]
        metrics["plateau_detrended_max_nt"] = float(np.max(np.abs(
            plateau - _fit(plateau)[0] - _fit(plateau)[1] * np.arange(len(plateau)))))
        # A constant subtraction must preserve interior variability. This value
        # cannot establish that correction created a bump: retain it for audit.
        # platform_not_single_A above remains the hard interior invariant.
        metrics["plateau_detrended_role"] = "diagnostic_only"
    else:
        boundaries = (left, right)
        metrics["plateau_join_residual_nt"] = None
        metrics["plateau_detrended_max_nt"] = None
        if not np.all(flags[left:right + 1] == 2):
            failures.append("transient_not_all_flag2")
    # Both permitted modes must contain exactly straight replacement segments.
    spans = [(left, right)] if state.morphology == "transient" else [
        (left, state.on_stable), (state.off_operation_node, right)]
    for lo, hi in spans:
        expected = np.linspace(corr[lo], corr[hi], hi - lo + 1)
        if not np.allclose(corr[lo:hi + 1], expected, rtol=0, atol=np.finfo(float).eps * max(1, np.max(np.abs(expected))) * 8):
            failures.append("transition_not_corrected_space_linear")
    differences = []
    for boundary in boundaries:
        lo, hi = max(0, boundary - 1), min(len(raw), boundary + 2)
        differences.extend(np.abs(np.diff(corr[lo:hi])).tolist())
    metrics["boundary_max_first_difference_nt"] = max(differences, default=0)
    if metrics["boundary_max_first_difference_nt"] > p.max_boundary_difference_nt:
        failures.append("boundary_spike")
    return {"passed": not failures, "failures": failures, "metrics": metrics,
            "visual_review": "unreviewed"}


def correct_step_d1p1(z, on_s, off_s, *, neighbor_catalog_s: Sequence[int] = (),
                     peer_series=None, a_ref=None, target_station=None,
                     policy: D1Policy | None = None,
                     endpoint_revision: str | None = None,
                     off_endpoint_revision: str | None = None,
                     on_endpoint_revision: str | None = None,
                     candidate_audit: bool = False,
                     admission_revision: str | None = None,
                     timing_revision: str | None = None,
                     endpoint_fallback_revision: str | None = None,
                     nonzero_radius_s: int = 0,
                     line_veto_audit: bool = False,
                     on_background_node_guard: bool = False,
                     on_catalog_end_s: int | None = None,
                     catalog_edge_fallback_ends: tuple[int, int] | None = None) -> StepD1Result:
    # Optional window modes require explicit shadow opt-in.
    node_window(int(on_s), nonzero_radius_s=nonzero_radius_s)
    if catalog_edge_fallback_ends is not None:
        if (not candidate_audit or nonzero_radius_s or line_veto_audit or on_background_node_guard
                or on_catalog_end_s is not None or timing_revision != "tie-target-dev01"
                or len(catalog_edge_fallback_ends) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in catalog_edge_fallback_ends)
                or not int(on_s) < catalog_edge_fallback_ends[0] <= DAY_S
                or not int(off_s) < catalog_edge_fallback_ends[1] <= DAY_S):
            raise ValueError("catalogue-edge fallback requires isolated shadow opt-in and valid original edge ends")
    if on_catalog_end_s is not None and (
            not candidate_audit or nonzero_radius_s or line_veto_audit or on_background_node_guard
            or int(on_s) % 60 or isinstance(on_catalog_end_s, bool)
            or not isinstance(on_catalog_end_s, (int, np.integer))
            or not int(on_s) < on_catalog_end_s <= DAY_S):
        raise ValueError("catalogue ON interval requires valid minute-level, isolated shadow opt-in")
    if nonzero_radius_s and not candidate_audit:
        raise ValueError("nonzero-second window experiment requires candidate_audit=True")
    if type(line_veto_audit) is not bool or (line_veto_audit and not candidate_audit):
        raise ValueError("line veto audit requires boolean opt-in and candidate_audit=True")
    if type(on_background_node_guard) is not bool or (on_background_node_guard and not (
            candidate_audit and endpoint_revision == "sample-state-dev03"
            and on_endpoint_revision == "bounded-contraction-dev01"
            and endpoint_fallback_revision == "missing-pair-dev02")):
        raise ValueError("ON background node guard requires explicit shadow and existing endpoint revisions")
    raw = np.asarray(z, dtype=float).copy()
    if not isinstance(candidate_audit, bool):
        raise ValueError("candidate_audit must be boolean")
    if timing_revision not in (None, "tie-target-dev01", "target-first-dev01"):
        raise ValueError("unknown timing revision")
    if timing_revision is not None and not candidate_audit:
        raise ValueError("experimental timing requires candidate_audit=True")
    if endpoint_fallback_revision not in (None, "side-pair-dev01", "missing-pair-dev02"):
        raise ValueError("unknown endpoint fallback revision")
    if endpoint_fallback_revision is not None and not (
            candidate_audit and endpoint_revision == "sample-state-dev03"
            and on_endpoint_revision == "bounded-contraction-dev01"
            and off_endpoint_revision == "supported-fallback-dev01"):
        raise ValueError("side-pair experiment requires shadow and existing endpoint revisions")
    if admission_revision not in (None, "constant-offset-dev01"):
        raise ValueError("unknown admission revision")
    allow_variable_background = candidate_audit or admission_revision is not None
    if endpoint_revision not in (None, "sample-state-dev03", "main-edge-dev01"):
        raise ValueError("unknown endpoint revision")
    if off_endpoint_revision not in (None, "zero-crossing-dev02", "local-edge-dev01", "supported-fallback-dev01"):
        raise ValueError("unknown OFF endpoint revision")
    if off_endpoint_revision is not None and endpoint_revision is None:
        raise ValueError("OFF revision requires explicit endpoint revision")
    if on_endpoint_revision not in (None, "bounded-contraction-dev01"):
        raise ValueError("unknown ON endpoint revision")
    if on_endpoint_revision is not None and endpoint_revision is None:
        raise ValueError("ON revision requires explicit endpoint revision")
    if raw.shape != (DAY_S,):
        raise ValueError("z must be a one-dimensional 86400-sample day")
    if (isinstance(on_s, bool) or isinstance(off_s, bool)
            or not isinstance(on_s, (int, np.integer)) or not isinstance(off_s, (int, np.integer))
            or not 0 <= on_s < off_s < DAY_S):
        raise ValueError("invalid catalog times; never silently clamp them")
    result = StepD1Result(raw.copy(), np.zeros(DAY_S, dtype=np.int8))
    result.evidence = {"version": PATCH, "catalog_on": int(on_s), "catalog_off": int(off_s),
                       "target_station": target_station, "visual_review": "unreviewed"}
    policy = FROZEN_POLICY if policy is None else policy
    if policy is None:
        result.reason = "numeric_policy_not_supplied"
        return result
    result.evidence["policy"] = asdict(policy)
    if target_station is None or not str(target_station).strip():
        result.reason = "target_identity_missing_for_LOO"
        return result
    target = str(target_station)
    peers = {str(k): v for k, v in (peer_series or {}).items() if str(k) != target}
    result.evidence["LOO_excluded_target"] = target
    bans = ban_windows(list(neighbor_catalog_s))
    timing_kw = {} if timing_revision is None else {"timing_revision": timing_revision}
    if nonzero_radius_s:
        timing_kw["nonzero_radius_s"] = nonzero_radius_s
    if line_veto_audit:
        timing_kw["line_veto_audit"] = True
    on_kw = dict(timing_kw)
    if on_catalog_end_s is not None:
        on_kw["catalog_end_s"] = int(on_catalog_end_s)
        result.evidence["on_catalogue_interval_shadow"] = [int(on_s), int(on_catalog_end_s)]
    on, on_ev = confirm_operation_time(raw, int(on_s), peers, policy, bans + [node_window(int(off_s))], **on_kw)
    off, off_ev = confirm_operation_time(raw, int(off_s), peers, policy, bans + [node_window(int(on_s))], **timing_kw)
    if catalog_edge_fallback_ends is not None:
        fallback = {}
        for side, cat, edge_end, node, ev, opposite in (
                ("on", int(on_s), int(catalog_edge_fallback_ends[0]), on, on_ev, int(off_s)),
                ("off", int(off_s), int(catalog_edge_fallback_ends[1]), off, off_ev, int(on_s))):
            detail = {"primary_node": node, "primary_evidence": ev, "attempted": False,
                      "catalogue_edge_end": edge_end, "primary_window": list(node_window(cat))}
            fallback[side] = detail
            if node is not None:
                detail["reason"] = "primary_node_locked"
                continue
            lo = node_window(cat)[1]
            caps = [edge_end, DAY_S] + [int(t) for t in neighbor_catalog_s if t > cat]
            if side == "on": caps.append(int(off_s))
            hi = min(caps)
            detail.update(remaining_window=[lo, hi], cap_candidates=sorted(set(caps)))
            if hi <= lo:
                detail["reason"] = "no_remaining_interval_before_catalogue_or_operation_boundary"
                continue
            detail["attempted"] = True
            found, found_ev = confirm_operation_time(raw, cat, peers, policy,
                bans + [node_window(opposite)], remaining_edge_window=(lo, hi), **timing_kw)
            detail.update(fallback_node=found, fallback_evidence=found_ev,
                          reason="fallback_confirmed" if found is not None else "fallback_unconfirmed")
            if found is not None:
                if side == "on": on, on_ev = found, found_ev
                else: off, off_ev = found, found_ev
        result.evidence["catalogue_edge_fallback"] = fallback
    result.evidence.update(on=on_ev, off=off_ev)
    result.line_consensus_on, result.line_consensus_off = on_ev["consensus_s"], off_ev["consensus_s"]
    result.n_peer_on, result.n_peer_off = len(on_ev["inlier_stations"]), len(off_ev["inlier_stations"])
    result.line_on_supported = on_ev["consensus_s"] is not None
    result.line_off_supported = off_ev["consensus_s"] is not None
    result.on_node_final, result.off_node_final = on, off
    if on is None or off is None:
        result.reason = ("on_" + on_ev["reason"]) if on is None else ("off_" + off_ev["reason"])
        return result
    if off <= on:
        result.reason = "operation_order_invalid"
        return result
    result.operation_confirmed = True
    state, local_ev = determine_local_state(raw, on, off, policy, bans)
    if allow_variable_background:
        # Quietness can select the unchanged proposal; it cannot veto a candidate.
        admission = {"mode": admission_revision or "admission-gates-dev01",
                     "writeback": not candidate_audit, "original_local_state": local_ev,
                     "plateau_is_hypothesis": True,
                     "interior_variability_role": "diagnostic_only"}
        result.evidence["candidate_audit" if candidate_audit else "constant_offset_admission"] = admission
        if state is None or local_ev["reason"] != "ok":
            state, local_ev = determine_local_state(raw, on, off, policy, bans, candidate_audit=True)
            admission["local_quality_gates_demoted"] = True
    result.evidence["local_state"] = local_ev
    if state is None:
        result.reason = local_ev["reason"]
        return result
    result.morphology = state.morphology
    result.off_end_final = state.off_end
    result.peak_final = state.peak
    if state.morphology == "plateau":
        result.on_stable_final, result.off_stable_final = state.on_stable, state.off_end
    if local_ev["reason"] != "ok":
        result.reason = local_ev["reason"]
        return result
    if state.morphology == "plateau":
        # Verify that the amplitude estimators have their required outer shoulders.
        if on < OUTER_S or state.off_end + OUTER_S > DAY_S or _overlaps(on - OUTER_S, state.off_end + OUTER_S, bans):
            result.reason = "amplitude_shoulders_unavailable"
            return result
        if not np.isfinite(raw[on - OUTER_S:state.off_end + OUTER_S]).all():
            result.reason = "amplitude_missing_data"
            return result
        result.A_on_nT = estimate_a_on_d(raw, on, state.on_stable, off)
        result.A_off_nT = estimate_a_off_d(raw, off, state.off_end, state.on_stable)
        result.closure_gap_nT = abs(result.A_on_nT - result.A_off_nT)
        closure = closure_reason(result.A_on_nT, result.A_off_nT)
        result.evidence["amplitude"] = {"method": "frozen_step_D", "closure": closure}
        if closure != "ok":
            result.reason = closure
            return result
        result.A_star_nT = combine_A(result.A_on_nT, result.A_off_nT)
        if a_ref is not None and math.isfinite(float(a_ref)) and float(a_ref) * result.A_star_nT < 0:
            result.reason = "catalog_amplitude_sign_conflict"
            return result
    else:
        result.evidence["amplitude"] = {"method": "not_applicable", "closure": None}
    if endpoint_revision is not None:
        result.evidence["amplitude_source_state"] = asdict(state)
        revised, endpoint_ev = refine_transition_membership(raw, state, result.A_star_nT, policy, bans,
                                                          off_revision=off_endpoint_revision, on_revision=on_endpoint_revision,
                                                          selection_revision=endpoint_revision,
                                                          on_background_node_guard=on_background_node_guard)
        result.evidence["endpoint_membership"] = endpoint_ev
        if (revised is None and allow_variable_background and state.morphology == "plateau"
                and endpoint_revision != "main-edge-dev01"):
            if (endpoint_fallback_revision == "side-pair-dev01" or
                    (endpoint_fallback_revision == "missing-pair-dev02" and endpoint_ev["reason"] == "endpoint_membership_unresolved")):
                revised, pair_evidence = _side_pair_fallback(raw, state, result.A_star_nT, policy, bans, endpoint_ev,
                    missing_only=endpoint_fallback_revision == "missing-pair-dev02")
                result.evidence["side_pair_fallback"] = pair_evidence
                if revised is None:
                    result.reason = pair_evidence["reason"]
                    return result
                admission.update(fine_endpoint_gate_demoted=endpoint_ev["reason"],
                                 endpoint_source="side_pair", candidate_state=asdict(revised))
            else:
                # Preserve the prior candidate proposal; unchanged QC evaluates it.
                saved = endpoint_ev.get("candidate_state_before_q_gate")
                proposed = LocalState(**saved) if saved else LocalState(
                    "plateau", state.on_node - 1, state.off_operation_node - 1,
                    state.off_end, state.on_stable)
                l, a, b, e = proposed.on_node, proposed.on_stable, proposed.off_operation_node, proposed.off_end
                if (0 <= l < a <= b < e < DAY_S and b - a + 1 >= policy.min_plateau_s
                        and l >= policy.evidence_s and e + policy.evidence_s < DAY_S
                        and not _overlaps(l - policy.evidence_s, e + policy.evidence_s + 1, bans)
                        and np.isfinite(raw[l - policy.evidence_s:e + policy.evidence_s + 1]).all()):
                    revised = proposed
                    admission.update(
                        fine_endpoint_gate_demoted=endpoint_ev["reason"],
                        endpoint_source="saved_proposal" if saved else "coarse_node_and_segmented_arrival",
                        candidate_state=asdict(proposed))
                    if endpoint_fallback_revision == "missing-pair-dev02":
                        source = "refined_saved_proposal" if saved else "coarse"
                        result.evidence["side_pair_fallback"] = {"revision": endpoint_fallback_revision,
                            "reason": "original_fallback_path_preserved", "sides": {
                                "ON": {"source": source, "selected_pair": [l, a]},
                                "OFF": {"source": source, "selected_pair": [b, e]}}}
        elif revised is not None and endpoint_fallback_revision is not None:
            result.evidence["side_pair_fallback"] = {"revision": endpoint_fallback_revision, "reason": "fine_complete_unchanged",
                "sides": {"ON": {"source": "refined", "selected_pair": [revised.on_node, revised.on_stable]},
                          "OFF": {"source": "refined", "selected_pair": [revised.off_operation_node, revised.off_end]}}}
        if revised is None:
            result.reason = endpoint_ev["reason"]
            return result
        state = revised
        result.on_stable_final, result.off_stable_final = state.on_stable, state.off_end
        result.off_end_final = state.off_end
    if state.morphology == "plateau":
        result.on_bg_end, result.on_plateau_start = state.on_node, state.on_stable
        result.off_plateau_end, result.off_bg_start = state.off_operation_node, state.off_end
    candidate, flags = apply_correction(raw, state, result.A_star_nT)
    qc = quality_control(raw, candidate, flags, state, result.A_star_nT, policy, bans)
    result.evidence["final_qc"] = qc
    if allow_variable_background:
        admission.update(candidate_state=asdict(state), original_QC_passed=qc["passed"],
                         candidate_available=True)
    if candidate_audit:
        result.reason = "diagnostic_candidate_only"
        return result
    if not qc["passed"]:
        result.reason = "final_qc_failed:" + ",".join(qc["failures"])
        return result
    result.corrected, result.flags = candidate, flags
    result.decision, result.final_status = "correct", "release"
    result.reason = "plateau_corrected" if state.morphology == "plateau" else "transient_continuity_placeholder"
    result.transition_mode = "corrected_space_two_transitions" if state.morphology == "plateau" else "corrected_space_whole_transient"
    result.A_used_nT = result.A_star_nT
    return result


def correct_with_locked_nodes(*args, **kwargs):
    """Disabled bypass; use correct_step_d1p1 with response evidence."""
    raise RuntimeError("Use correct_step_d1p1 with LOO/local evidence; locked-node bypass is disabled.")
