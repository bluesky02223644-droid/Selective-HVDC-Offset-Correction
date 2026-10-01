"""One in-memory synthetic API check; no real-data evaluation or writeback."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "src/analysis/funnel_2024_2025_ge0p5/step_corrector_d1p1"
sys.path.insert(0, str(CORE))
from step_corrector_d1p1 import D1Policy, correct_step_d1p1


def main():
    spec = json.loads((ROOT / "configs/workflow_v2_1.json").read_text(encoding="utf-8"))["scientific_spec"]
    t = np.arange(86400, dtype=float)
    on_catalog, off_catalog = 36000, 39600
    on_actual, off_actual, ramp_s = 36020, 39620, 30
    shape = np.clip((t - on_actual) / ramp_s, 0, 1) - np.clip((t - off_actual) / ramp_s, 0, 1)
    raw = 20000 + 0.0001 * t + 0.005 * np.sin(t / 13) + 3.0 * shape
    original = raw.copy()
    arguments = {k: v for k, v in spec.items() if k not in {"entry", "policy", "catalog_edge_fallback_ends"}}
    arguments["catalog_edge_fallback_ends"] = (on_actual + ramp_s, off_actual + ramp_s)
    result = correct_step_d1p1(raw, on_catalog, off_catalog,
        target_station="synthetic_target", peer_series={}, a_ref=3.0,
        policy=D1Policy(**spec["policy"]), **arguments)
    assert np.array_equal(raw, original), "Input array was changed"
    assert np.array_equal(result.corrected, original), "Shadow output must remain raw"
    assert not result.flags.any(), "Shadow output must retain zero flags"
    for name in ("step_corrector_d1p1", "timing_windows", "step_corrector_d", "step_corrector_c", "step_corrector"):
        Path(sys.modules[name].__file__).resolve().relative_to(ROOT)
    admission = result.evidence.get("candidate_audit", {})
    qc = result.evidence.get("final_qc", {})
    print(json.dumps({"example": "synthetic API check", "reason": result.reason,
        "candidate_available": admission.get("candidate_available", False),
        "candidate_qc_passed": qc.get("passed"), "qc_failures": qc.get("failures", []),
        "estimated_offset_nT": result.A_star_nT if np.isfinite(result.A_star_nT) else None,
        "input_unchanged": True, "final_shadow_output_unchanged": True,
        "returned_flags_all_zero": True, "imports_within_package": True}, indent=2))


if __name__ == "__main__":
    main()
