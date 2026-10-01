"""Time-window and neighbouring-operation bounds used by D.1."""
from __future__ import annotations

DAY_S = 86400
NODE_WINDOW_S = 60
MAX_TRANSITION_S = 90


def is_minute_catalog_edge(catalog_s: int) -> bool:
    return int(catalog_s) % 60 == 0


def catalog_precision(catalog_s: int) -> str:
    return "minute" if is_minute_catalog_edge(catalog_s) else "second"


def node_window(catalog_s: int, *, nonzero_radius_s: int = 0) -> tuple[int, int]:
    """Half-open legal window for node_s.

    Minute-level: [catalog, catalog+60).
    Second-level: [catalog, catalog+1). The manuscript configuration uses nonzero_radius_s=0.
    """
    if type(nonzero_radius_s) is not int or nonzero_radius_s not in (0, 3, 5, 10, 30, 60, 300):
        raise ValueError("undeclared nonzero-second window")
    t0 = int(catalog_s)
    if catalog_precision(t0) == "minute":
        return t0, t0 + NODE_WINDOW_S
    return max(0, t0 - nonzero_radius_s), min(DAY_S, t0 + nonzero_radius_s + 1)


def is_legal_node(node_s: int | None, catalog_s: int) -> bool:
    if node_s is None:
        return False
    lo, hi = node_window(catalog_s)
    return lo <= int(node_s) < hi


def illegal_reason(node_s: int, catalog_s: int) -> str:
    if is_legal_node(node_s, catalog_s):
        return ""
    return "illegal_candidate"


def ban_windows(catalog_times: list[int] | tuple[int, ...]) -> list[tuple[int, int]]:
    out = []
    seen: set[tuple[int, int]] = set()
    for t in catalog_times:
        w = node_window(int(t))
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def in_ban(t: int, bans: list[tuple[int, int]]) -> bool:
    return any(lo <= int(t) < hi for lo, hi in bans)


def first_ban_after(t0: int, bans: list[tuple[int, int]]) -> int | None:
    starts = [lo for lo, hi in bans if lo > int(t0)]
    return min(starts) if starts else None


def stable_hard_stop(node_s: int, bans: list[tuple[int, int]], extra_stop: int | None = None) -> int:
    """First second the transition must not enter. No window expansion."""
    stop = min(DAY_S, int(node_s) + MAX_TRANSITION_S + 1)
    nxt = first_ban_after(node_s, bans)
    if nxt is not None:
        stop = min(stop, nxt)
    if extra_stop is not None:
        stop = min(stop, int(extra_stop))
    return max(int(node_s) + 1, stop)
