"""Amplitude-estimation helpers required by D.1.

Other retained functions are dependency code, not publication entry points."""
from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

STEP_C = Path(__file__).resolve().parents[1] / "step_corrector_c1"
STEP_A = Path(__file__).resolve().parents[1] / "step_corrector_v1"
sys.path.insert(0, str(STEP_C))
sys.path.insert(0, str(STEP_A))
from step_corrector import DAY_S, EDGE_MIN_S, MIN_SAMPLES, _interp_edges, _median  # noqa: E402
from step_corrector_c import (  # noqa: E402
    _empty as _empty_c,
    _finite,
    _jump8,
    closure_reason,
    combine_A,
    locate_off_c,
    locate_on_c,
)


N_QUIET_S = 8
ABS_FLOOR_NT = 0.25
BG_K = 2.5
IN_CAP_S = 90
OUT_CAP_S = 90
BG_PAD_S = 20
BG_SPAN_S = 70
# Amplitude helper constants; transition rules are unchanged.
OUTER_S = 60
TILE_S = 16
PLAT_WIN_S = 24
SKIP_CAP_TILES = 2
FAR_TOL_NT = 0.30
U_SPAN_NT = 0.80
U_CONTINUE_NT = 0.25
N_TILES = 5


@dataclass(frozen=True)
class StepDResult:
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
    quiet_floor_on_nT: float
    quiet_floor_off_nT: float
    corrected: np.ndarray
    flags: np.ndarray

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("corrected")
        payload.pop("flags")
        return payload


def _bg_floor(z: np.ndarray, t0: int, outer_dir: int) -> float:
    """Local |jump8| on the outer side of the edge. Not the mid-event U."""
    if outer_dir < 0:
        lo, hi = t0 - BG_PAD_S - BG_SPAN_S, t0 - BG_PAD_S
    else:
        lo, hi = t0 + BG_PAD_S, t0 + BG_PAD_S + BG_SPAN_S
    jumps = []
    for t in range(max(30, lo), min(DAY_S - 30, hi), 2):
        val = abs(_jump8(z, t))
        if math.isfinite(val):
            jumps.append(val)
    bg = float(np.median(jumps)) if jumps else ABS_FLOOR_NT
    return max(ABS_FLOOR_NT, BG_K * bg)


def walk_until_quiet(z: np.ndarray, t0: int, direction: int, floor: float, cap: int) -> int:
    """Transition ends when |jump8| stays within the outer-side background for N seconds."""
    reached = t0 + direction * EDGE_MIN_S
    quiet = 0
    for dt in range(EDGE_MIN_S, max(EDGE_MIN_S, cap) + 1):
        t = t0 + direction * dt
        if t < 30 or t > DAY_S - 30:
            break
        reached = t
        if abs(_jump8(z, t)) <= floor:
            quiet += 1
            if quiet >= N_QUIET_S:
                return t
        else:
            quiet = 0
    return reached


def _segment(z: np.ndarray, a: int, b: int) -> float:
    lo, hi = (int(a), int(b)) if a <= b else (int(b), int(a))
    lo = max(0, lo)
    hi = min(DAY_S, hi)
    if hi - lo < MIN_SAMPLES:
        return math.nan
    return _median(z, lo, hi)


def _tiles(z: np.ndarray, inner: int, toward_mid: int, stop: int) -> list[float]:
    meds: list[float] = []
    for k in range(N_TILES):
        a = int(inner) + toward_mid * k * TILE_S
        b = a + toward_mid * TILE_S
        if toward_mid > 0 and b > stop:
            break
        if toward_mid < 0 and b < stop:
            break
        val = _segment(z, a, b)
        if math.isfinite(val):
            meds.append(val)
    return meds


def _inner_plateau(z: np.ndarray, inner: int, toward_mid: int, stop: int, outer_level: float) -> float:
    """After jump8-quiet: hold the near tile if a long U keeps going; else skip a short DC tail."""
    meds = _tiles(z, inner, toward_mid, stop)
    if not meds:
        return math.nan
    if (
        len(meds) >= 5
        and abs(meds[3] - meds[0]) >= U_SPAN_NT
        and (meds[3] - meds[0]) * (meds[4] - meds[3]) > 0
        and abs(meds[4] - meds[3]) >= U_CONTINUE_NT
    ):
        return meds[0]
    k = 0
    while k < SKIP_CAP_TILES and k + 2 < len(meds):
        if abs(meds[k + 2] - meds[k]) < FAR_TOL_NT:
            break
        if (meds[k] - outer_level) * (meds[k + 2] - meds[k]) <= 0:
            break
        k += 1
    start = int(inner) + toward_mid * k * TILE_S
    return _segment(z, start, start + toward_mid * PLAT_WIN_S)


def estimate_a_on_d(z: np.ndarray, on_left: int, on_right: int, off_left: int) -> float:
    pre = _segment(z, on_left - OUTER_S, on_left)
    plat = _inner_plateau(z, on_right, 1, off_left, pre)
    return plat - pre


def estimate_a_off_d(z: np.ndarray, off_left: int, off_right: int, on_right: int) -> float:
    after = _segment(z, off_right, off_right + OUTER_S)
    plat = _inner_plateau(z, off_left, -1, on_right, after)
    return plat - after


def join_residuals(corr: np.ndarray, on_left: int, on_right: int, off_left: int, off_right: int) -> tuple[float, float]:
    pre = _median(corr, on_left - PLAT_WIN_S, on_left)
    on_plat = _median(corr, on_right, on_right + PLAT_WIN_S)
    off_plat = _median(corr, off_left - PLAT_WIN_S, off_left)
    after = _median(corr, off_right, off_right + PLAT_WIN_S)
    return on_plat - pre, off_plat - after


def _empty(raw, flags, reason, a_on, a_off, on_a, off_a, on_s, off_s, f_on=math.nan, f_off=math.nan) -> StepDResult:
    base = _empty_c(raw, flags, reason, a_on, a_off, on_a, off_a, on_s, off_s)
    return StepDResult(
        decision=base.decision,
        reason=base.reason,
        A_on_nT=base.A_on_nT,
        A_off_nT=base.A_off_nT,
        A_star_nT=base.A_star_nT,
        A_used_nT=base.A_used_nT,
        closure_gap_nT=base.closure_gap_nT,
        on_apply_s=base.on_apply_s,
        off_apply_s=base.off_apply_s,
        on_delay_s=base.on_delay_s,
        off_delay_s=base.off_delay_s,
        on_left_s=base.on_left_s,
        on_right_s=base.on_right_s,
        off_left_s=base.off_left_s,
        off_right_s=base.off_right_s,
        blend_s=base.blend_s,
        quiet_floor_on_nT=_finite(f_on),
        quiet_floor_off_nT=_finite(f_off),
        corrected=base.corrected,
        flags=base.flags,
    )


def correct_step_d(z: np.ndarray, on_s: int, off_s: int) -> StepDResult:
    raw = np.asarray(z, dtype=float).copy()
    if raw.shape[0] != DAY_S:
        raise ValueError("z must be a 86400-sample day")
    on_s = max(0, min(DAY_S - 2, int(on_s)))
    off_s = max(on_s + 2, min(DAY_S - 1, int(off_s)))
    flags = np.zeros(DAY_S, dtype=np.int8)

    on_apply, a_on_probe, _ = locate_on_c(raw, on_s, off_s)
    if on_apply is None:
        return _empty(raw, flags, "no_step_near_catalog", a_on_probe, math.nan, on_s, off_s, on_s, off_s)
    off_apply, a_off_probe, _ = locate_off_c(raw, on_s, off_s, on_apply=on_apply, a_on=a_on_probe)
    if off_apply is None:
        return _empty(raw, flags, "no_step_near_catalog", a_on_probe, a_off_probe, on_apply, off_s, on_s, off_s)
    if off_apply <= on_apply:
        return _empty(raw, flags, "window_collapsed", a_on_probe, a_off_probe, on_apply, off_apply, on_s, off_s)

    floor_on = _bg_floor(raw, on_apply, -1)
    floor_off = _bg_floor(raw, off_apply, 1)
    on_left = walk_until_quiet(raw, on_apply, -1, floor_on, OUT_CAP_S)
    on_right = walk_until_quiet(raw, on_apply, 1, floor_on, IN_CAP_S)
    off_left = walk_until_quiet(raw, off_apply, -1, floor_off, IN_CAP_S)
    off_right = walk_until_quiet(raw, off_apply, 1, floor_off, OUT_CAP_S)
    if off_left - on_right < 8:
        mid = (on_apply + off_apply) // 2
        on_right = mid
        off_left = mid

    a_on = estimate_a_on_d(raw, on_left, on_right, off_left)
    a_off = estimate_a_off_d(raw, off_left, off_right, on_right)
    closed = closure_reason(a_on, a_off)
    if closed != "ok":
        return _empty(
            raw, flags, closed, a_on, a_off, on_apply, off_apply, on_s, off_s, floor_on, floor_off
        )

    a_star = combine_A(a_on, a_off)
    returned = raw.copy()
    returned[on_right:off_left] = raw[on_right:off_left] - a_star
    flags[on_right:off_left] = 1
    flags[on_left:on_right] = 2
    flags[off_left:off_right] = 2
    _interp_edges(returned, flags)

    return StepDResult(
        decision="correct",
        reason="settle_skip_or_hold_near_A",
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
        quiet_floor_on_nT=_finite(floor_on),
        quiet_floor_off_nT=_finite(floor_off),
        corrected=returned,
        flags=flags,
    )
