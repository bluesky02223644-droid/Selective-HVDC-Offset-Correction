"""Numerical helpers required by the D.1 amplitude dependencies.

Use correct_step_d1p1 as the sole publication entry point."""
from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

STEP_A = Path(__file__).resolve().parents[1] / "step_corrector_v1"
sys.path.insert(0, str(STEP_A))
from step_corrector import (  # noqa: E402
    A_MIN_NT,
    DAY_S,
    EDGE_BLEND_S,
    EDGE_MIN_S,
    LOCATE_WIN_S,
    MIN_PLATEAU_S,
    MIN_SAMPLES,
    SCAN_STEP_S,
    SETTLE_S,
    SHOULDER_S,
    _estimate_a_off,
    _estimate_a_on,
    _interp_edges,
    _local_jump_off,
    _local_jump_on,
    _median,
    _pick_jump,
)

# Helper constants retained for D.1 dependencies.
CLOSURE_ABS_NT = 1.0
CLOSURE_REL = 0.20
SEAM_OUT_CAP_S = 90
SEAM_IN_CAP_S = 60
# Catalog is a coarse anchor. Search both sides. Not "always later".
SLACK_MAX_S = 600


@dataclass(frozen=True)
class StepCResult:
    decision: str
    reason: str
    A_on_nT: float
    A_off_nT: float
    A_star_nT: float
    A_used_nT: float
    closure_gap_nT: float
    on_apply_s: int
    off_apply_s: int
    on_delay_s: int
    off_delay_s: int
    on_left_s: int
    on_right_s: int
    off_left_s: int
    off_right_s: int
    blend_s: int
    corrected: np.ndarray
    flags: np.ndarray

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("corrected")
        payload.pop("flags")
        return payload


def _finite(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def combine_A(a_on: float, a_off: float) -> float:
    return 0.5 * (a_on + a_off)


def official_residuals(
    corrected: np.ndarray,
    on_s: int,
    on_apply: int,
    off_apply: int,
) -> tuple[float, float]:
    pre = _median(corrected, on_s - SHOULDER_S, on_s)
    on_plat = _median(corrected, on_apply + SETTLE_S, on_apply + SETTLE_S + SHOULDER_S)
    settle = min(SETTLE_S, max(0, off_apply - on_apply - MIN_SAMPLES))
    off_plat = _median(corrected, off_apply - settle - SHOULDER_S, off_apply - settle)
    after = _median(corrected, off_apply + settle, off_apply + settle + SHOULDER_S)
    return on_plat - pre, off_plat - after


def closure_reason(a_on: float, a_off: float) -> str:
    if not math.isfinite(a_on) or not math.isfinite(a_off):
        return "offset_unreliable"
    if abs(a_on) < A_MIN_NT or abs(a_off) < A_MIN_NT:
        return "too_small"
    if a_on * a_off <= 0.0:
        return "sign_mismatch"
    scale = max(abs(a_on), abs(a_off))
    if abs(a_on - a_off) > max(CLOSURE_ABS_NT, CLOSURE_REL * scale):
        return "not_constant_step"
    if abs(combine_A(a_on, a_off)) < A_MIN_NT:
        return "too_small"
    return "ok"


def _jump8(z: np.ndarray, t: int) -> float:
    return _median(z, t, t + 8) - _median(z, t - 8, t)


def _slack_c(on_s: int, off_s: int) -> int:
    span = max(1, int(off_s) - int(on_s))
    return max(25, min(SLACK_MAX_S, max(120, span)))


def locate_on_c(z: np.ndarray, on_s: int, off_s: int) -> tuple[int | None, float, int]:
    slack = min(_slack_c(on_s, off_s), max(0, off_s - on_s - LOCATE_WIN_S - 1))
    pairs: list[tuple[int, float]] = []
    for delay in range(-slack, slack + 1, SCAN_STEP_S):
        t = on_s + delay
        if t < LOCATE_WIN_S + 1 or t > off_s - LOCATE_WIN_S - 1:
            continue
        pairs.append((delay, _local_jump_on(z, t)))
    delay, probe = _pick_jump(pairs)
    if delay is None:
        return None, probe, slack
    return on_s + delay, probe, slack


def locate_off_c(
    z: np.ndarray,
    on_s: int,
    off_s: int,
    on_apply: int | None = None,
    a_on: float | None = None,
) -> tuple[int | None, float, int]:
    slack = min(_slack_c(on_s, off_s), max(0, off_s - on_s - LOCATE_WIN_S - 1))
    lo = (int(on_apply) + MIN_PLATEAU_S) if on_apply is not None else on_s + LOCATE_WIN_S
    pairs: list[tuple[int, float]] = []
    for delay in range(-slack, slack + 1, SCAN_STEP_S):
        t = off_s + delay
        if t <= lo or t >= DAY_S - LOCATE_WIN_S:
            continue
        pairs.append((delay, _local_jump_off(z, t)))
    delay, probe = _pick_jump(pairs, sign=a_on)
    if delay is None:
        return None, probe, slack
    return off_s + delay, probe, slack


def settle_inward(z: np.ndarray, t0: int, direction: int, a_star: float, cap: int) -> int:
    floor = max(0.35, 0.18 * abs(float(a_star)))
    reached = t0 + direction * max(EDGE_BLEND_S, EDGE_MIN_S)
    for dt in range(EDGE_MIN_S, max(EDGE_MIN_S, cap) + 1):
        t = t0 + direction * dt
        if t < 20 or t > DAY_S - 20:
            break
        reached = t
        if abs(_jump8(z, t)) <= floor:
            return t
    return reached


def settle_outward(z: np.ndarray, t0: int, direction: int, a_star: float, cap: int) -> int:
    floor = max(0.30, 0.12 * abs(float(a_star)))
    min_pad = 8
    reached = t0 + direction * min_pad
    quiet = 0
    for dt in range(4, max(min_pad, cap) + 1):
        t = t0 + direction * dt
        if t < 30 or t > DAY_S - 30:
            break
        reached = t
        jump = abs(_jump8(z, t))
        if direction > 0:
            here = _median(z, t, t + 8)
            farther = _median(z, t + 12, t + 28)
        else:
            here = _median(z, t - 8, t)
            farther = _median(z, t - 28, t - 12)
        level_ok = (
            np.isfinite(here)
            and np.isfinite(farther)
            and abs(here - farther) <= max(floor, 0.35)
        )
        if jump <= floor and level_ok:
            quiet += 1
            if quiet >= 3 and dt >= min_pad:
                return t
        else:
            quiet = 0
    return reached


def _join_gap(z: np.ndarray, outer: int, inner: int, a_star: float, inward_positive: bool) -> float:
    if inward_positive:
        pre = _median(z, outer - 16, outer)
        inside = _median(z, inner, inner + 16)
    else:
        inside = _median(z, inner - 16, inner)
        pre = _median(z, outer, outer + 16)
    if not math.isfinite(pre) or not math.isfinite(inside):
        return math.inf
    return abs((inside - pre) - a_star)


def expand_seam(
    z: np.ndarray,
    t0: int,
    a_star: float,
    outward_dir: int,
    cap_out: int,
    cap_in: int,
) -> tuple[int, int]:
    """Widen the yellow band until the join matches A*, not a geometric floor."""
    outer = settle_outward(z, t0, outward_dir, a_star, cap_out)
    inner = settle_inward(z, t0, -outward_dir, a_star, cap_in)
    tol = max(0.28, 0.10 * abs(float(a_star)))
    floor = max(0.30, 0.12 * abs(float(a_star)))
    inward_positive = outward_dir < 0
    for _ in range(cap_out + cap_in):
        gap = _join_gap(z, outer, inner, a_star, inward_positive)
        j_out = abs(_jump8(z, outer))
        j_in = abs(_jump8(z, inner))
        if gap <= tol and j_out <= floor and j_in <= floor:
            break
        moved = False
        if abs(outer - t0) < cap_out:
            nxt = outer + outward_dir
            if 30 < nxt < DAY_S - 30:
                outer = nxt
                moved = True
        if abs(inner - t0) < cap_in:
            nxt = inner - outward_dir
            if 30 < nxt < DAY_S - 30:
                inner = nxt
                moved = True
        if not moved:
            break
    if outward_dir < 0:
        return outer, inner
    return inner, outer


def _empty(raw: np.ndarray, flags: np.ndarray, reason: str, a_on: float, a_off: float,
           on_a: int, off_a: int, on_s: int, off_s: int) -> StepCResult:
    gap = abs(a_on - a_off) if math.isfinite(a_on) and math.isfinite(a_off) else math.nan
    return StepCResult(
        decision="abstain",
        reason=reason,
        A_on_nT=_finite(a_on),
        A_off_nT=_finite(a_off),
        A_star_nT=math.nan,
        A_used_nT=math.nan,
        closure_gap_nT=gap,
        on_apply_s=on_a,
        off_apply_s=off_a,
        on_delay_s=on_a - on_s if on_a else 0,
        off_delay_s=off_a - off_s if off_a else 0,
        on_left_s=on_a,
        on_right_s=on_a,
        off_left_s=off_a,
        off_right_s=off_a,
        blend_s=0,
        corrected=raw.copy(),
        flags=flags,
    )


def correct_step_c(z: np.ndarray, on_s: int, off_s: int) -> StepCResult:
    raw = np.asarray(z, dtype=float).copy()
    if raw.shape[0] != DAY_S:
        raise ValueError("z must be a 86400-sample day")
    on_s = max(0, min(DAY_S - 2, int(on_s)))
    off_s = max(on_s + 2, min(DAY_S - 1, int(off_s)))
    flags = np.zeros(DAY_S, dtype=np.int8)

    on_apply, a_on_probe, _ = locate_on_c(raw, on_s, off_s)
    if on_apply is None:
        return _empty(raw, flags, "no_step_near_catalog", a_on_probe, math.nan, on_s, off_s, on_s, off_s)
    off_apply, a_off_probe, _ = locate_off_c(
        raw, on_s, off_s, on_apply=on_apply, a_on=a_on_probe
    )
    if off_apply is None:
        return _empty(raw, flags, "no_step_near_catalog", a_on_probe, a_off_probe, on_apply, off_s, on_s, off_s)
    if off_apply <= on_apply:
        return _empty(raw, flags, "window_collapsed", a_on_probe, a_off_probe, on_apply, off_apply, on_s, off_s)

    a_on = _estimate_a_on(raw, on_apply, on_apply, off_apply)
    a_off = _estimate_a_off(raw, off_apply, on_apply)
    closed = closure_reason(a_on, a_off)
    if closed != "ok":
        return _empty(raw, flags, closed, a_on, a_off, on_apply, off_apply, on_s, off_s)

    a_star = combine_A(a_on, a_off)
    on_left, on_right = expand_seam(raw, on_apply, a_star, -1, SEAM_OUT_CAP_S, SEAM_IN_CAP_S)
    off_left, off_right = expand_seam(raw, off_apply, a_star, 1, SEAM_OUT_CAP_S, SEAM_IN_CAP_S)
    if off_left - on_right < 8:
        mid = (on_apply + off_apply) // 2
        on_right = mid
        off_left = mid

    returned = raw.copy()
    returned[on_right:off_left] = raw[on_right:off_left] - a_star
    flags[on_right:off_left] = 1
    flags[on_left:on_right] = 2
    flags[off_left:off_right] = 2
    _interp_edges(returned, flags)

    return StepCResult(
        decision="correct",
        reason="local_edge_constant_step",
        A_on_nT=_finite(a_on),
        A_off_nT=_finite(a_off),
        A_star_nT=_finite(a_star),
        A_used_nT=_finite(a_star),
        closure_gap_nT=abs(a_on - a_off),
        on_apply_s=on_apply,
        off_apply_s=off_apply,
        on_delay_s=on_apply - on_s,
        off_delay_s=off_apply - off_s,
        on_left_s=on_left,
        on_right_s=on_right,
        off_left_s=off_left,
        off_right_s=off_right,
        blend_s=max(
            on_apply - on_left,
            off_right - off_apply,
            on_right - on_apply,
            off_apply - off_left,
            EDGE_MIN_S,
        ),
        corrected=returned,
        flags=flags,
    )
