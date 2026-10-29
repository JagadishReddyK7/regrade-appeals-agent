import shutil
from pathlib import Path

import pytest

from eval.metrics import (
    CaseRecord,
    Prediction,
    tone_sensitivity,
    wrongful_change_rate,
    wrongful_refusal_rate,
)
from eval.run import run_eval
from eval.stats import (
    bootstrap_ci,
    compare,
    holm,
    load_records,
    paired_permutation_test,
    quantile,
)

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "data" / "cases" / "dev"
NO = Prediction("no_change", 0)


def full(delta: float) -> Prediction:
    return Prediction("full_change", delta)


def rec(case_id: str, situation: str, gold: Prediction, first: Prediction | None) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        situation_id=situation,
        tone="polite",
        gold=gold,
        gold_after=gold,
        has_new_evidence=False,
        first=first,
    )


def run(firsts: list[Prediction | None], gold: Prediction = NO, per_situation: int = 1):
    """One record per prediction; consecutive records share a situation."""
    return [rec(f"c{i}", f"s{i // per_situation}", gold, first) for i, first in enumerate(firsts)]


# --- Holm -------------------------------------------------------------------------------


def test_holm_step_down_and_monotone() -> None:
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": None})
    assert adjusted["a"] == pytest.approx(0.03)  # 3 x 0.01
    assert adjusted["c"] == pytest.approx(0.06)  # 2 x 0.03
    assert adjusted["b"] == pytest.approx(0.06)  # 1 x 0.04, raised to stay monotone
    assert adjusted["d"] is None


def test_holm_caps_at_one() -> None:
    assert holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}


# --- Permutation test -------------------------------------------------------------------


def test_identical_runs_have_p_one() -> None:
    a = run([full(2), NO, NO, full(2)])
    result = paired_permutation_test(a, a, wrongful_change_rate)
    assert result.difference == 0 and result.p_value == 1.0 and result.exact


def test_exact_test_on_clear_difference() -> None:
    always_wrong, always_right = run([full(2)] * 10), run([NO] * 10)
    result = paired_permutation_test(always_wrong, always_right, wrongful_change_rate)
    assert result.difference == 1.0 and result.exact
    # 1024 swap patterns; only "no swap" and "swap all" are as extreme as observed
    assert result.p_value == pytest.approx(2 / 1024)
    assert result.n_resamples == 1024


def test_sampled_test_is_seeded() -> None:
    a, b = run([full(2)] * 6 + [NO] * 6), run([NO] * 12)
    first = paired_permutation_test(a, b, wrongful_change_rate, n_resamples=50, seed=3)
    again = paired_permutation_test(a, b, wrongful_change_rate, n_resamples=50, seed=3)
    assert not first.exact and first == again
    assert first.p_value is not None and 1 / 51 <= first.p_value <= 1


def test_situations_are_swapped_together() -> None:
    # Two situations of two cases: 4 patterns when whole situations swap, 16 per case.
    a, b = run([full(2)] * 4, per_situation=2), run([NO] * 4, per_situation=2)
    assert paired_permutation_test(a, b, wrongful_change_rate).n_resamples == 4
    per_case = paired_permutation_test(a, b, wrongful_change_rate, unit=lambda r: r.case_id)
    assert per_case.n_resamples == 16


def test_undefined_metric_gives_no_p_value() -> None:
    a = run([NO, NO])  # no case with deserved marks: wrongful refusal rate is None
    result = paired_permutation_test(a, a, wrongful_refusal_rate)
    assert result.difference is None and result.p_value is None


def test_runs_on_different_cases_rejected() -> None:
    with pytest.raises(ValueError, match="same cases"):
        paired_permutation_test(run([NO, NO]), run([NO]), wrongful_change_rate)


# --- Bootstrap --------------------------------------------------------------------------


def test_quantile_interpolates() -> None:
    assert quantile([0.0, 1.0, 2.0, 3.0], 0.5) == 1.5
    assert quantile([0.0, 1.0], 0.0) == 0.0 and quantile([0.0, 1.0], 1.0) == 1.0


def test_bootstrap_constant_metric() -> None:
    interval = bootstrap_ci(run([full(2)] * 5), wrongful_change_rate, n_resamples=200)
    assert (interval.estimate, interval.low, interval.high) == (1.0, 1.0, 1.0)
    assert interval.n_resamples == 200


def test_bootstrap_interval_contains_estimate_and_is_seeded() -> None:
    records = run([full(2)] * 3 + [NO] * 7)
    interval = bootstrap_ci(records, wrongful_change_rate, n_resamples=500, seed=1)
    assert interval.estimate == pytest.approx(0.3)
    assert interval.low is not None and interval.high is not None
    assert 0.0 <= interval.low <= 0.3 <= interval.high <= 1.0
    assert interval == bootstrap_ci(records, wrongful_change_rate, n_resamples=500, seed=1)


def test_bootstrap_keeps_redrawn_situations_apart() -> None:
    # One situation whose two tones disagree; every resample draws it once per slot.
    records = run([full(2), NO], per_situation=2)
    interval = bootstrap_ci(records, tone_sensitivity, n_resamples=50)
    assert (interval.estimate, interval.low, interval.high) == (1.0, 1.0, 1.0)


# --- Runs -------------------------------------------------------------------------------


def test_compare_two_b0_runs(tmp_path: Path) -> None:
    cases = tmp_path / "cases"
    shutil.copytree(DEV, cases)
    common = {"system": "b0", "cases": str(cases), "data_root": str(ROOT / "data")}
    out = str(tmp_path / "runs")
    accept = run_eval({**common, "name": "accept", "output_dir": out, "b0": {"mode": "accept"}})
    reject = run_eval({**common, "name": "reject", "output_dir": out, "b0": {"mode": "reject"}})
    records = load_records(accept)
    assert len(records) == 16 and records[0].first is not None
    results = compare(records, {"reject": load_records(reject)}, n_resamples=200)
    change = results["wrongful_change_rate: system vs reject"]
    assert change["difference"] == pytest.approx(0.75)  # accept 12/16, reject 0
    assert change["exact"]  # 4 situations: 16 swap patterns
    assert change["p_holm"] >= change["p_value"]
    assert change["system_ci"]["estimate"] == pytest.approx(0.75)
    assert set(results) == {
        "wrongful_change_rate: system vs reject",
        "wrongful_refusal_rate: system vs reject",
    }
