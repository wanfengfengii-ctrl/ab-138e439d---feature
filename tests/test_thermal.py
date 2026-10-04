"""热补偿模式测试：与全枚举暴力联合最优对照，并覆盖决胜、不可行与字段错误。"""

from __future__ import annotations

import itertools
import random
import time

import pytest

from app.solver import (
    InfeasibleError,
    ValidationErrors,
    invert_payload,
    validate,
)


def brute_force_thermal_optimal(
    lengths, coeffs, t_min, t_max, strain_min, strain_max, windows_raw
):
    """全枚举参考：枚举 (tau, m_0..m_{n-1})，键 (M, S, tuple(m), tau) 取最小。

    窗约束按总应变 m_i + c_i*tau 回算；统一应变界与平滑指标只约束 m。
    """
    n = len(lengths)
    best = None
    for tau in range(t_min, t_max + 1):
        for ms in itertools.product(range(strain_min, strain_max + 1), repeat=n):
            ok = True
            for s, e, lo, hi in windows_raw:
                total = sum(
                    lengths[i] * (ms[i] + coeffs[i] * tau)
                    for i in range(s, e + 1)
                )
                if not (lo <= total <= hi):
                    ok = False
                    break
            if not ok:
                continue
            diffs = [ms[i] - ms[i - 1] for i in range(1, n)]
            m = max(abs(d) for d in diffs)
            s_abs = sum(abs(d) for d in diffs)
            key = (m, s_abs, ms, tau)
            if best is None or key < best[0]:
                best = (key, list(ms), tau)
    return best


def make_thermal_payload(
    lengths, strain_min, strain_max, windows_raw, coeffs, t_min, t_max
):
    return {
        "segment_lengths": lengths,
        "strain_bounds": {"min": strain_min, "max": strain_max},
        "windows": [
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": lo,
                "max_elongation": hi,
            }
            for s, e, lo, hi in windows_raw
        ],
        "thermal_compensation": {
            "coefficients": coeffs,
            "temperature_delta_bounds": {"min": t_min, "max": t_max},
        },
    }


def assert_thermal_recomputable(result, payload):
    """仅凭提交数据 + 响应复算每窗机械/热/总和，并检查机械未重复吸收热胀。"""
    lengths = payload["segment_lengths"]
    coeffs = payload["thermal_compensation"]["coefficients"]
    tau = result["thermal_compensation"]["temperature_delta"]
    ms = result["strains"]
    for check, win in zip(result["window_checks"], payload["windows"]):
        s = win["start_segment"] - 1
        e = win["end_segment"] - 1
        mech = sum(lengths[i] * ms[i] for i in range(s, e + 1))
        therm = tau * sum(lengths[i] * coeffs[i] for i in range(s, e + 1))
        assert check["mechanical_contribution"] == mech
        assert check["thermal_contribution"] == therm
        assert check["weighted_strain_sum"] == mech + therm
        assert win["min_elongation"] <= mech + therm <= win["max_elongation"]
        assert check["satisfied"] is True
    # 逐段总应变也可由提交数据复算。
    tc = result["thermal_compensation"]
    assert tc["thermal_strains"] == [tau * c for c in coeffs]
    assert tc["total_strains"] == [ms[i] + tau * coeffs[i] for i in range(len(ms))]
    # 平滑指标只关于机械应变。
    diffs = [ms[i] - ms[i - 1] for i in range(1, len(ms))]
    assert result["adjacent_diffs"] == diffs
    assert result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs)
    assert result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs)


@pytest.mark.parametrize("seed", range(8))
def test_random_thermal_matches_bruteforce(seed):
    rng = random.Random(400 + seed)
    n = 6
    lengths = [rng.randint(1, 4) for _ in range(n)]
    coeffs = [rng.randint(0, 3) for _ in range(n)]
    strain_min, strain_max = -2, 2
    true_tau = rng.randint(-2, 2)
    truth_m = [rng.randint(strain_min, strain_max) for _ in range(n)]
    windows_raw = []
    seen = set()
    while len(windows_raw) < 10:
        s = rng.randint(0, n - 1)
        e = rng.randint(s, n - 1)
        if (s, e) in seen:
            continue
        seen.add((s, e))
        total = sum(
            lengths[i] * (truth_m[i] + coeffs[i] * true_tau)
            for i in range(s, e + 1)
        )
        slack = rng.randint(0, 2)
        windows_raw.append((s, e, total - slack, total + slack))
    t_min, t_max = true_tau - 2, true_tau + 2  # 跨度 4
    payload = make_thermal_payload(
        lengths, strain_min, strain_max, windows_raw, coeffs, t_min, t_max
    )
    result = invert_payload(payload)
    expected = brute_force_thermal_optimal(
        lengths, coeffs, t_min, t_max, strain_min, strain_max, windows_raw
    )
    assert expected is not None
    assert result["strains"] == expected[1]
    assert result["thermal_compensation"]["temperature_delta"] == expected[2]
    assert_thermal_recomputable(result, payload)


def test_thermal_tie_prefers_smaller_delta():
    """三级完全相同时（这里所有 tau 下最优机械解恒为全 0），取区间较小温变。"""
    lengths = [2, 3, 1, 4, 2, 3]
    coeffs = [1, 0, 2, 1, 0, 1]
    # 窗界给得足够宽：任意 tau ∈ [-3,3]、m=0 均可行，且 m=0 是字典序最小的
    # M=S=0 解，与 tau 无关，因此三级键的机械部分恒同。
    windows = [
        {"start_segment": 1, "end_segment": 6, "min_elongation": -10**6,
         "max_elongation": 10**6}
    ]
    windows += [
        {"start_segment": i, "end_segment": i, "min_elongation": -10**6,
         "max_elongation": 10**6}
        for i in range(1, 7)
    ]
    windows += [
        {"start_segment": 1, "end_segment": 3, "min_elongation": -10**6,
         "max_elongation": 10**6}
    ]
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": 0, "max": 5},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": coeffs,
            "temperature_delta_bounds": {"min": -3, "max": 3},
        },
    }
    result = invert_payload(payload)
    assert result["strains"] == [0] * 6
    assert result["objectives"] == {"max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0}
    assert result["thermal_compensation"]["temperature_delta"] == -3
    assert_thermal_recomputable(result, payload)


def test_zero_coefficients_makes_delta_unidentifiable():
    """系数全 0 时温变不参与任何窗约束，联合结果取区间最小温变且热贡献全 0。"""
    lengths = [1] * 6
    windows = [
        {"start_segment": 1, "end_segment": 6, "min_elongation": 6, "max_elongation": 6}
    ]
    windows += [
        {"start_segment": i, "end_segment": i, "min_elongation": 1, "max_elongation": 1}
        for i in range(1, 7)
    ]
    windows += [
        {"start_segment": 1, "end_segment": 3, "min_elongation": 3, "max_elongation": 3}
    ]
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -10, "max": 10},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": [0] * 6,
            "temperature_delta_bounds": {"min": 2, "max": 9},
        },
    }
    result = invert_payload(payload)
    assert result["strains"] == [1] * 6
    assert result["thermal_compensation"]["temperature_delta"] == 2
    assert all(c["thermal_contribution"] == 0 for c in result["window_checks"])
    assert all(c["weighted_strain_sum"] == c["mechanical_contribution"]
               for c in result["window_checks"])


def _constant_total_strain_payload(t_min, t_max):
    """每段总应变被钉为 8，机械界 [-3,3]；c_i=1 时需 tau ∈ [5,11]。"""
    lengths = [2, 3, 1, 4, 2, 3]
    windows = [
        {"start_segment": 1, "end_segment": 6,
         "min_elongation": sum(lengths) * 8, "max_elongation": sum(lengths) * 8}
    ]
    windows += [
        {"start_segment": i, "end_segment": i,
         "min_elongation": lengths[i - 1] * 8, "max_elongation": lengths[i - 1] * 8}
        for i in range(1, 7)
    ]
    windows += [
        {"start_segment": 2, "end_segment": 4,
         "min_elongation": sum(lengths[1:4]) * 8,
         "max_elongation": sum(lengths[1:4]) * 8}
    ]
    return {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -3, "max": 3},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": [1] * 6,
            "temperature_delta_bounds": {"min": t_min, "max": t_max},
        },
    }


def test_thermal_rescues_mechanically_infeasible_instance():
    """旧模式不可行（机械界放不下总应变），启用热补偿后由热项承担共同偏移。"""
    payload = _constant_total_strain_payload(0, 20)
    legacy = {k: v for k, v in payload.items() if k != "thermal_compensation"}
    with pytest.raises(InfeasibleError):
        invert_payload(legacy)
    result = invert_payload(payload)
    # m + tau = 8；M=0 解 m 恒定，字典序最小 m=-3 对应最大可行 tau=11。
    assert result["strains"] == [-3] * 6
    assert result["thermal_compensation"]["temperature_delta"] == 11
    assert_thermal_recomputable(result, payload)
    for check in result["window_checks"]:
        assert check["min_elongation"] == check["max_elongation"]
        assert (
            check["mechanical_contribution"] + check["thermal_contribution"]
            == check["min_elongation"]
        )


def test_no_joint_explanation_is_infeasible():
    """输入合法但温变区间排除了全部解释 -> InfeasibleError（而非字段错误）。"""
    payload = _constant_total_strain_payload(0, 2)  # 需要 tau ∈ [5,11]
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


def test_omitting_thermal_keeps_legacy_shape():
    """省略 thermal_compensation 时响应不出现热补偿字段。"""
    payload = _constant_total_strain_payload(0, 20)
    legacy = {k: v for k, v in payload.items() if k != "thermal_compensation"}
    legacy["strain_bounds"] = {"min": -100, "max": 100}
    result = invert_payload(legacy)
    assert "thermal_compensation" not in result
    for check in result["window_checks"]:
        assert "mechanical_contribution" not in check
        assert "thermal_contribution" not in check


@pytest.mark.parametrize(
    "mutate,expected_path",
    [
        (lambda p: p["thermal_compensation"].pop("coefficients"),
         "thermal_compensation.coefficients"),
        (lambda p: p["thermal_compensation"].pop("temperature_delta_bounds"),
         "thermal_compensation.temperature_delta_bounds"),
        (lambda p: p["thermal_compensation"].update(coefficients=[1] * 5),
         "thermal_compensation.coefficients"),
        (lambda p: p["thermal_compensation"].update(coefficients=[1, 1, -1, 1, 1, 1]),
         "thermal_compensation.coefficients[2]"),
        (lambda p: p["thermal_compensation"].update(coefficients=[1, 1, 1.0, 1, 1, 1]),
         "thermal_compensation.coefficients[2]"),
        (lambda p: p["thermal_compensation"].update(coefficients=[1, True, 1, 1, 1, 1]),
         "thermal_compensation.coefficients[1]"),
        (lambda p: p["thermal_compensation"]["temperature_delta_bounds"].update(
            min=5, max=26),
         "thermal_compensation.temperature_delta_bounds"),
        (lambda p: p["thermal_compensation"]["temperature_delta_bounds"].update(
            min=9, max=8),
         "thermal_compensation.temperature_delta_bounds"),
        (lambda p: p["thermal_compensation"]["temperature_delta_bounds"].update(
            min="0", max=2),
         "thermal_compensation.temperature_delta_bounds.min"),
        (lambda p: p["thermal_compensation"].update(unknown=1),
         "thermal_compensation.unknown"),
    ],
)
def test_thermal_field_validation(mutate, expected_path):
    payload = _constant_total_strain_payload(0, 20)
    mutate(payload)
    with pytest.raises(ValidationErrors) as exc:
        validate(payload)
    assert any(e["field"] == expected_path for e in exc.value.fields), exc.value.fields


def test_thermal_object_not_object():
    payload = _constant_total_strain_payload(0, 20)
    payload["thermal_compensation"] = [1, 2, 3]
    with pytest.raises(ValidationErrors) as exc:
        validate(payload)
    assert any(e["field"] == "thermal_compensation" for e in exc.value.fields)


def test_span_boundary_20_is_legal_21_infeasible_as_validation():
    """跨度恰为 20 合法；21 为字段错误。"""
    payload = _constant_total_strain_payload(0, 20)
    invert_payload(payload)  # 不抛
    payload["thermal_compensation"]["temperature_delta_bounds"] = {"min": 0, "max": 21}
    with pytest.raises(ValidationErrors) as exc:
        validate(payload)
    assert any(
        e["field"] == "thermal_compensation.temperature_delta_bounds"
        for e in exc.value.fields
    )


def test_thermal_max_size_performance():
    """12 段、20 窗、21 个温变值的联合反演仍须在合理时间内完成。"""
    rng = random.Random(77)
    n = 12
    lengths = [rng.randint(1, 100) for _ in range(n)]
    coeffs = [rng.randint(0, 3) for _ in range(n)]
    true_tau = rng.randint(-3, 3)
    truth_m = [rng.randint(-3000, 3000) for _ in range(n)]
    windows = []
    # 12 个紧单段窗钉住每段总应变域，再加 8 个紧多段窗。
    for i in range(n):
        total = lengths[i] * (truth_m[i] + coeffs[i] * true_tau)
        windows.append(
            {"start_segment": i + 1, "end_segment": i + 1,
             "min_elongation": total - 5, "max_elongation": total + 5}
        )
    pairs = [(s, e) for s in range(n) for e in range(s + 1, n)]
    rng.shuffle(pairs)
    for s0, e0 in pairs[:8]:
        total = sum(
            lengths[i] * (truth_m[i] + coeffs[i] * true_tau)
            for i in range(s0, e0 + 1)
        )
        windows.append(
            {"start_segment": s0 + 1, "end_segment": e0 + 1,
             "min_elongation": total - 50, "max_elongation": total + 50}
        )
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -100000, "max": 100000},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": coeffs,
            "temperature_delta_bounds": {"min": -10, "max": 10},
        },
    }
    start = time.monotonic()
    result = invert_payload(payload)
    elapsed = time.monotonic() - start
    assert elapsed < 30.0
    assert_thermal_recomputable(result, payload)
