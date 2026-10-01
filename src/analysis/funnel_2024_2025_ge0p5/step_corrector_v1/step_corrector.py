"""Low-level sample, median and interpolation helpers used by D.1 dependencies.

Use correct_step_d1p1 as the sole publication entry point."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

DAY_S = 86400
SHOULDER_S = 60
LOCATE_WIN_S = 20
SETTLE_S = 40
SLACK_MAX_S = 120
SCAN_STEP_S = 5
A_MIN_NT = 0.4
CLOSURE_ABS_NT = 0.5
CLOSURE_REL = 0.35
EDGE_BLEND_S = 15
EDGE_MIN_S = 6
MAX_EDGE_S = 480
LEVEL_ABS_NT = 0.5
LEVEL_REL = 0.06
LEVEL_CAP_NT = 1.5
MIN_PLATEAU_S = 40
MIN_SAMPLES = 8


@dataclass(frozen=True)
class StepResult:
    decision: str
    reason: str
    A_on_nT: float
    A_off_nT: float
    A_used_nT: float
    closure_gap_nT: float
    on_apply_s: int
    off_apply_s: int
    on_delay_s: int
    off_delay_s: int
    blend_s: int
    corrected: np.ndarray
    flags: np.ndarray

    def summary(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("corrected")
        payload.pop("flags")
        payload["A_closure_gap_nT"] = payload["closure_gap_nT"]
        return payload


def _finite(value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def _median(z: np.ndarray, lo: int, hi: int) -> float:
    sl = z[max(0, int(lo)) : min(DAY_S, int(hi))]
    if sl.size == 0 or int(np.isfinite(sl).sum()) < MIN_SAMPLES:
        return math.nan
    return float(np.nanmedian(sl))


def _slack(on_s: int, off_s: int) -> int:
    span = max(1, int(off_s) - int(on_s))
    return max(25, min(SLACK_MAX_S, span // 2))


def _same_sign(jump: float, sign: float | None) -> bool:
    if sign is None or not math.isfinite(sign) or sign == 0.0:
        return True
    return math.isfinite(jump) and jump * sign > 0.0


def _pick_jump(
    pairs: list[tuple[int, float]],
    sign: float | None = None,
) -> tuple[int | None, float]:
    scored = [
        (delay, jump)
        for delay, jump in pairs
        if math.isfinite(jump) and _same_sign(jump, sign)
    ]
    if not scored:
        last = pairs[-1][1] if pairs else math.nan
        return None, last
    peak = max(abs(jump) for _, jump in scored)
    if peak < A_MIN_NT:
        return None, scored[-1][1]
    thresh = max(A_MIN_NT, 0.70 * peak)
    candidates = [(delay, jump) for delay, jump in scored if abs(jump) >= thresh]
    if not candidates:
        return None, scored[-1][1]
    return max(candidates, key=lambda item: abs(item[1]))


def _local_jump_on(z: np.ndarray, t: int) -> float:
    return _median(z, t, t + LOCATE_WIN_S) - _median(z, t - LOCATE_WIN_S, t)


def _local_jump_off(z: np.ndarray, t: int) -> float:
    return _median(z, t - LOCATE_WIN_S, t) - _median(z, t, t + LOCATE_WIN_S)


def _estimate_a_on(z: np.ndarray, on_s: int, on_apply: int, off_stop: int) -> float:
    pre = _median(z, on_s - SHOULDER_S, on_s)
    plat_lo = on_apply + SETTLE_S
    plat_hi = min(int(off_stop), plat_lo + SHOULDER_S)
    if plat_hi - plat_lo < MIN_SAMPLES:
        plat_hi = min(int(off_stop), on_apply + SETTLE_S + SHOULDER_S)
        plat_lo = max(on_apply, plat_hi - SHOULDER_S)
    return _median(z, plat_lo, plat_hi) - pre


def _estimate_a_off(z: np.ndarray, off_apply: int, on_apply: int) -> float:
    settle = min(SETTLE_S, max(0, off_apply - on_apply - MIN_SAMPLES))
    plateau = _median(z, off_apply - settle - SHOULDER_S, off_apply - settle)
    after = _median(z, off_apply + settle, off_apply + settle + SHOULDER_S)
    return plateau - after


def locate_on(z: np.ndarray, on_s: int, off_s: int) -> tuple[int | None, float, int]:
    slack = min(_slack(on_s, off_s), max(0, off_s - on_s - LOCATE_WIN_S - 1))
    pairs: list[tuple[int, float]] = []
    for delay in range(0, slack + 1, SCAN_STEP_S):
        pairs.append((delay, _local_jump_on(z, on_s + delay)))
    delay, probe = _pick_jump(pairs)
    if delay is None:
        return None, probe, slack
    apply_s = on_s + delay
    amp = _estimate_a_on(z, on_s, apply_s, off_s)
    if math.isfinite(amp) and abs(amp) >= A_MIN_NT:
        return apply_s, amp, slack
    return apply_s, probe if math.isfinite(probe) else amp, slack


def locate_off(
    z: np.ndarray,
    on_s: int,
    off_s: int,
    on_apply: int | None = None,
    a_on: float | None = None,
) -> tuple[int | None, float, int]:
    slack = min(_slack(on_s, off_s), max(0, off_s - on_s - LOCATE_WIN_S - 1))
    slack_back = slack
    if on_apply is not None:
        slack_back = min(slack_back, max(0, off_s - int(on_apply) - MIN_PLATEAU_S))

    def scan(delays: range) -> list[tuple[int, float]]:
        pairs: list[tuple[int, float]] = []
        for delay in delays:
            t = off_s + delay
            if t <= on_s or t >= DAY_S - LOCATE_WIN_S:
                continue
            pairs.append((delay, _local_jump_off(z, t)))
        return pairs

    delay, probe = _pick_jump(scan(range(0, slack + 1, SCAN_STEP_S)), sign=a_on)
    if delay is None:
        delay, probe = _pick_jump(
            scan(range(-SCAN_STEP_S, -slack_back - 1, -SCAN_STEP_S)),
            sign=a_on,
        )
    if delay is None:
        return None, probe, slack
    apply_s = off_s + delay
    amp = _estimate_a_off(z, apply_s, on_apply or on_s)
    if math.isfinite(amp) and abs(amp) >= A_MIN_NT:
        return apply_s, amp, slack
    return apply_s, probe if math.isfinite(probe) else amp, slack


def _closure(a_on: float, a_off: float) -> str:
    if not math.isfinite(a_on) or not math.isfinite(a_off):
        return "offset_unreliable"
    scale = max(abs(a_on), abs(a_off))
    if scale < A_MIN_NT:
        return "too_small"
    if abs(a_on - a_off) > max(CLOSURE_ABS_NT, CLOSURE_REL * scale):
        return "not_constant_step"
    return "ok"


def _level_tol(scale: float) -> float:
    return max(LEVEL_ABS_NT, min(LEVEL_CAP_NT, LEVEL_REL * abs(scale)))


def _near_level(value: float, target: float, scale: float) -> bool:
    if not math.isfinite(value) or not math.isfinite(target):
        return False
    return abs(value - target) <= _level_tol(scale)


def _walk_to_level(
    z: np.ndarray,
    start: int,
    direction: int,
    target: float,
    scale: float,
    limit: int,
) -> int:
    reached = int(start)
    for step in range(0, int(limit) + 1):
        t = int(start) + int(direction) * step
        if t < 10 or t > DAY_S - 10:
            break
        point = float(z[t]) if np.isfinite(z[t]) else math.nan
        band = _median(z, t, t + 8) if direction > 0 else _median(z, t - 8, t)
        reached = t
        if _near_level(point, target, scale) and _near_level(band, target, scale):
            return t
    return reached


def _interp_edges(returned: np.ndarray, flags: np.ndarray) -> None:
    fill = np.flatnonzero(flags == 2)
    if not len(fill):
        return
    trusted = (flags != 2) & np.isfinite(returned)
    anchors = np.flatnonzero(trusted)
    if not len(anchors):
        return
    returned[fill] = np.interp(
        fill.astype(float),
        anchors.astype(float),
        returned[anchors],
    )


def correct_step(
    z: np.ndarray,
    on_s: int,
    off_s: int,
) -> StepResult:
    """Return a catalog-window constant-shift correction or abstain."""
    raw = np.asarray(z, dtype=float).copy()
    if raw.shape[0] != DAY_S:
        raise ValueError("z must be a 86400-sample day")
    on_s = max(0, min(DAY_S - 2, int(on_s)))
    off_s = max(on_s + 2, min(DAY_S - 1, int(off_s)))
    flags = np.zeros(DAY_S, dtype=np.int8)
    empty = raw.copy()

    def abstain(
        reason: str,
        a_on: float,
        a_off: float,
        on_a: int,
        off_a: int,
    ) -> StepResult:
        return StepResult(
            decision="abstain",
            reason=reason,
            A_on_nT=_finite(a_on),
            A_off_nT=_finite(a_off),
            A_used_nT=math.nan,
            closure_gap_nT=(
                abs(a_on - a_off)
                if math.isfinite(a_on) and math.isfinite(a_off)
                else math.nan
            ),
            on_apply_s=on_a,
            off_apply_s=off_a,
            on_delay_s=max(0, on_a - on_s) if on_a else 0,
            off_delay_s=off_s - off_a if off_a else 0,
            blend_s=0,
            corrected=empty,
            flags=flags,
        )

    on_apply, a_on, _slack_on = locate_on(raw, on_s, off_s)
    if on_apply is None:
        return abstain("no_step_near_catalog", a_on, math.nan, on_s, off_s)
    off_apply, a_off, _slack_off = locate_off(
        raw, on_s, off_s, on_apply=on_apply, a_on=a_on
    )
    if off_apply is None:
        return abstain("no_step_near_catalog", a_on, a_off, on_apply, off_s)
    if off_apply <= on_apply:
        return abstain("window_collapsed", a_on, a_off, on_apply, off_apply)

    a_on_edge = _estimate_a_on(raw, on_s, on_apply, off_apply)
    a_off_edge = _estimate_a_off(raw, off_apply, on_apply)
    closed = _closure(a_on_edge, a_off_edge)
    if closed != "ok":
        return abstain(closed, a_on_edge, a_off_edge, on_apply, off_apply)

    pre = _median(raw, on_s - SHOULDER_S, on_s)
    mid = (on_apply + off_apply) // 2
    plat = _median(raw, mid - 30, mid + 30)
    a_mid = plat - pre
    if (
        math.isfinite(a_mid)
        and abs(a_mid) >= A_MIN_NT
        and (not math.isfinite(a_on_edge) or a_mid * a_on_edge > 0.0)
    ):
        a_on = a_mid
    else:
        a_on = a_on_edge
    a_off = a_off_edge
    if not math.isfinite(a_on) or abs(a_on) < A_MIN_NT:
        return abstain("too_small", a_on, a_off, on_apply, off_apply)

    on_left = _walk_to_level(raw, on_apply, -1, pre, a_on, MAX_EDGE_S)
    on_right = _walk_to_level(raw, on_apply, 1, plat, a_on, MAX_EDGE_S)
    off_left = _walk_to_level(raw, off_apply, -1, plat, a_on, MAX_EDGE_S)
    off_right = _walk_to_level(raw, off_apply, 1, pre, a_on, MAX_EDGE_S)
    if on_right - on_left < EDGE_MIN_S:
        pad = (EDGE_MIN_S - (on_right - on_left) + 1) // 2
        on_left = max(0, on_left - pad)
        on_right = min(off_apply, on_right + pad)
    if off_right - off_left < EDGE_MIN_S:
        pad = (EDGE_MIN_S - (off_right - off_left) + 1) // 2
        off_left = max(on_right, off_left - pad)
        off_right = min(DAY_S, off_right + pad)
    room = max(MIN_PLATEAU_S, off_apply - on_apply)
    max_each = max(EDGE_MIN_S, (room - MIN_PLATEAU_S) // 2)
    on_right = min(on_right, on_apply + max_each)
    off_left = max(off_left, off_apply - max_each)
    if off_left - on_right < MIN_PLATEAU_S:
        return abstain("window_too_short", a_on, a_off, on_apply, off_apply)

    returned = raw.copy()
    returned[on_right:off_left] = raw[on_right:off_left] - a_on
    flags[on_right:off_left] = 1
    flags[on_left:on_right] = 2
    flags[off_left:off_right] = 2
    _interp_edges(returned, flags)

    return StepResult(
        decision="correct",
        reason="constant_step",
        A_on_nT=a_on,
        A_off_nT=a_off,
        A_used_nT=a_on,
        closure_gap_nT=abs(a_on - a_off),
        on_apply_s=on_apply,
        off_apply_s=off_apply,
        on_delay_s=on_apply - on_s,
        off_delay_s=off_s - off_apply,
        blend_s=max(on_right - on_apply, off_apply - off_left, EDGE_MIN_S),
        corrected=returned,
        flags=flags,
    )
