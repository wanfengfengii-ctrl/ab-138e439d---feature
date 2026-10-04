"""热补偿联合反演测试：

- 与全枚举暴力联合解（枚举机械应变 × 共同温变）对照四级字典序最优；
- 字段级校验（缺字段、系数数量/非负、温变区间非法/跨度过大）；
- 合法但无联合解释 -> INFEASIBLE；
- 每窗机械贡献/热贡献/总回算和均可由提交数据独立复核；
- 三级完全相同时取较小温变；
- 省略 thermal_compensation 时响应结构与旧模式完全一致。
"""

from __future__ import annotations

import itertools
import random
import time

import pytest

from app.solver import InfeasibleError, ValidationErrors, invert_payload, validate


def brute_force_optimal_thermal(
    lengths, strain_min, strain_max, windows_raw, coeff, t_min, t_max
):
    """全枚举参考：x ∈ 应变域、t ∈ 温变闭区间，按 (M, S, tuple(x), t) 取最小。"""
    n = len(lengths)
    best = None
    for t in range(t_min, t_max + 1):
        for xs in itertools.product(range(strain_min, strain_max + 1), repeat=n):
            ok = True
            for s, e, lo, hi in windows_raw:
                total = sum(
                    lengths[i] * (xs[i] + coeff[i] * t) for i in range(s, e + 1)
                )
                if not (lo <= total <= hi):
                    ok = False
                    break
            if not ok:
                continue
            diffs = [xs[i] - xs[i - 1] for i in range(1, n)]
            m = max(abs(d) for d in diffs)
            s_abs = sum(abs(d) for d in diffs)
            key = (m, s_abs, tuple(xs), t)
            if best is None or key < best[0]:
                best = (key, list(xs), diffs, t)
    return best


def make_thermal_payload(
    lengths, strain_min, strain_max, windows_raw, coeff, t_min, t_max
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
            "coefficients": list(coeff),
            "temperature_delta_bounds": {"min": t_min, "max": t_max},
        },
    }


def assert_thermal_checks(result, payload):
    """仅凭提交数据复算每个窗的机械贡献、热贡献与总回算和。"""
    lengths = payload["segment_lengths"]
    coeff = payload["thermal_compensation"]["coefficients"]
    t = result["thermal_compensation"]["temperature_delta"]
    strains = result["strains"]
    assert result["thermal_compensation"]["coefficients"] == coeff
    for check, win in zip(result["window_checks"], payload["windows"]):
        s = win["start_segment"] - 1
        e = win["end_segment"] - 1
        mech = sum(lengths[i] * strains[i] for i in range(s, e + 1))
        c_sum = sum(lengths[i] * coeff[i] for i in range(s, e + 1))
        heat = t * c_sum
        assert check["mechanical_weighted_sum"] == mech
        assert check["weighted_coefficient_sum"] == c_sum
        assert check["thermal_weighted_sum"] == heat
        assert check["weighted_strain_sum"] == mech + heat
        assert check["satisfied"] is True
        assert win["min_elongation"] <= mech + heat <= win["max_elongation"]


@pytest.mark.parametrize("seed", range(12))
def test_random_small_match_bruteforce(seed):
    rng = random.Random(500 + seed)
    n = 6
    lengths = [rng.randint(1, 4) for _ in range(n)]
    strain_min, strain_max = -2, 2
    coeff = [rng.randint(0, 2) for _ in range(n)]
    t_min, t_max = -1, 2  # 4 个候选温变
    truth_x = [rng.randint(strain_min, strain_max) for _ in range(n)]
    truth_t = rng.randint(t_min, t_max)
    windows_raw = []
    seen = set()
    while len(windows_raw) < 10:
        s = rng.randint(0, n - 1)
        e = rng.randint(s, n - 1)
        if (s, e) in seen:
            continue
        seen.add((s, e))
        total = sum(
            lengths[i] * (truth_x[i] + coeff[i] * truth_t) for i in range(s, e + 1)
        )
        slack = rng.randint(0, 2)
        windows_raw.append((s, e, total - slack, total + slack))
    payload = make_thermal_payload(
        lengths, strain_min, strain_max, windows_raw, coeff, t_min, t_max
    )
    result = invert_payload(payload)
    expected = brute_force_optimal_thermal(
        lengths, strain_min, strain_max, windows_raw, coeff, t_min, t_max
    )
    assert expected is not None
    (m, s_abs, xs_tuple, t), strains, diffs, chosen_t = expected
    assert result["strains"] == strains
    assert result["thermal_compensation"]["temperature_delta"] == chosen_t
    assert result["objectives"]["max_adjacent_diff"] == m
    assert result["objectives"]["sum_adjacent_abs_diff"] == s_abs
    assert_thermal_checks(result, payload)


@pytest.mark.parametrize("seed", range(5))
def test_bruteforce_infeasible(seed):
    """允许温变内无任何联合解释时，暴力与服务都应判不可行。"""
    rng = random.Random(600 + seed)
    n = 6
    lengths = [rng.randint(1, 3) for _ in range(n)]
    strain_min, strain_max = -1, 1
    coeff = [rng.randint(0, 2) for _ in range(n)]
    t_min, t_max = -2, 2
    # 单段窗把每段总应变钉为随机等式，再注入一个与任何 (x,t) 都矛盾的全长窗。
    truth_x = [rng.randint(strain_min, strain_max) for _ in range(n)]
    truth_t = rng.randint(t_min, t_max)
    windows_raw = []
    for i in range(n):
        total = lengths[i] * (truth_x[i] + coeff[i] * truth_t)
        windows_raw.append((i, i, total, total))
    # 单段窗已唯一确定每组 (x,t) 等价类；全长窗要求远超界的值。
    windows_raw.append((0, n - 1, 10**6, 10**6))
    while len(windows_raw) < 8:
        i = rng.randint(0, n - 1)
        total = lengths[i] * (truth_x[i] + coeff[i] * truth_t)
        windows_raw.append((i, i, total, total))
    assert (
        brute_force_optimal_thermal(
            lengths, strain_min, strain_max, windows_raw, coeff, t_min, t_max
        )
        is None
    )
    payload = make_thermal_payload(
        lengths, strain_min, strain_max, windows_raw, coeff, t_min, t_max
    )
    with pytest.raises(InfeasibleError):
        invert_payload(payload)


def test_thermal_stripped_so_mechanical_strain_is_zero():
    """观测完全由共同热胀造成：唯一可联合解释是 t=5、机械应变恒 0。"""
    lengths = [3, 5, 2, 4, 6, 1]
    coeff = [2, 2, 2, 2, 2, 2]  # 微应变/度
    t_true = 5
    windows = []
    spans = [(0, 5), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 2)]
    for s, e in spans:
        total = t_true * sum(lengths[i] * coeff[i] for i in range(s, e + 1))
        windows.append(
            {"start_segment": s + 1, "end_segment": e + 1,
             "min_elongation": total, "max_elongation": total}
        )
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -1, "max": 1},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": coeff,
            "temperature_delta_bounds": {"min": 0, "max": 10},
        },
    }
    result = invert_payload(payload)
    assert result["strains"] == [0] * 6
    assert result["adjacent_diffs"] == [0] * 5
    assert result["thermal_compensation"]["temperature_delta"] == 5
    assert_thermal_checks(result, payload)
    # 机械贡献必须为 0——共同热胀没有被机械应变重复吸收。
    assert all(c["mechanical_weighted_sum"] == 0 for c in result["window_checks"])


def test_zero_coefficients_picks_smallest_temperature():
    """系数全 0 时各温变三级解完全相同，必须取区间内最小温变。"""
    rng = random.Random(777)
    n = 6
    lengths = [rng.randint(1, 5) for _ in range(n)]
    truth = [rng.randint(-2, 2) for _ in range(n)]
    windows_raw = []
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    for s, e in pairs[:10]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows_raw.append((s, e, total - 1, total + 1))
    payload = make_thermal_payload(
        lengths, -5, 5, windows_raw, [0] * n, 3, 9
    )
    result = invert_payload(payload)
    assert result["thermal_compensation"]["temperature_delta"] == 3
    assert all(c["thermal_weighted_sum"] == 0 for c in result["window_checks"])


def test_temperature_tie_break_against_bruteforce():
    """构造两个温变三级指标完全相同的场景，验证取较小温变。"""
    # coeff 全 0 -> 温变与观测无关；(M,S,x) 对所有 t 相同。
    lengths = [1, 2, 3, 4, 5, 6]
    windows = []
    spans = [(0, 5), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (1, 4)]
    for s, e in spans:
        total = sum(lengths[i] for i in range(s, e + 1))  # 真值机械恒 1
        windows.append(
            {"start_segment": s + 1, "end_segment": e + 1,
             "min_elongation": total, "max_elongation": total}
        )
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -10, "max": 10},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": [0] * 6,
            "temperature_delta_bounds": {"min": -4, "max": 4},
        },
    }
    result = invert_payload(payload)
    assert result["strains"] == [1] * 6
    assert result["thermal_compensation"]["temperature_delta"] == -4


def test_omitted_thermal_keeps_legacy_response_shape():
    """省略 thermal_compensation 时结果与旧模式逐字节结构一致。"""
    payload = {
        "segment_lengths": [10, 12, 11, 13, 10, 14],
        "strain_bounds": {"min": -100, "max": 100},
        "windows": [
            {"start_segment": 1, "end_segment": 6, "min_elongation": 60, "max_elongation": 700},
            {"start_segment": 1, "end_segment": 1, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 2, "end_segment": 2, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 3, "end_segment": 3, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 4, "end_segment": 4, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 5, "end_segment": 5, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 6, "end_segment": 6, "min_elongation": 0, "max_elongation": 1000},
            {"start_segment": 1, "end_segment": 3, "min_elongation": 0, "max_elongation": 2000},
        ],
    }
    result = invert_payload(payload)
    assert "thermal_compensation" not in result
    for check in result["window_checks"]:
        assert "thermal_weighted_sum" not in check
        assert "mechanical_weighted_sum" not in check
        assert "weighted_coefficient_sum" not in check


def test_zero_temperature_matches_legacy_mode():
    """温变固定为 0 时热贡献恒 0，机械解必须与省略 thermal_compensation 完全一致。"""
    rng = random.Random(8888)
    n = 6
    lengths = [rng.randint(1, 6) for _ in range(n)]
    coeff = [rng.randint(0, 9) for _ in range(n)]
    truth = [rng.randint(-4, 4) for _ in range(n)]
    windows_raw = []
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    for s, e in pairs[:12]:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows_raw.append((s, e, total - 2, total + 2))
    base = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -8, "max": 8},
        "windows": [
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": lo,
                "max_elongation": hi,
            }
            for s, e, lo, hi in windows_raw
        ],
    }
    legacy = invert_payload(base)
    with_t0 = invert_payload(
        {
            **base,
            "thermal_compensation": {
                "coefficients": coeff,
                "temperature_delta_bounds": {"min": 0, "max": 0},
            },
        }
    )
    assert with_t0["strains"] == legacy["strains"]
    assert with_t0["adjacent_diffs"] == legacy["adjacent_diffs"]
    assert with_t0["objectives"] == legacy["objectives"]
    assert with_t0["thermal_compensation"]["temperature_delta"] == 0
    assert all(c["thermal_weighted_sum"] == 0 for c in with_t0["window_checks"])
    for a, b in zip(with_t0["window_checks"], legacy["window_checks"]):
        assert a["mechanical_weighted_sum"] == b["weighted_strain_sum"]
        assert a["weighted_strain_sum"] == b["weighted_strain_sum"]


@pytest.mark.parametrize("seed", range(4))
def test_n7_bruteforce_joint(seed):
    """n=7、3^7 机械枚举点 × 多个温变的全枚举联合对照。"""
    rng = random.Random(900 + seed)
    n = 7
    lengths = [rng.randint(1, 3) for _ in range(n)]
    strain_min, strain_max = -1, 1
    coeff = [rng.randint(0, 1) for _ in range(n)]
    t_min, t_max = -1, 1
    truth_x = [rng.randint(strain_min, strain_max) for _ in range(n)]
    truth_t = rng.randint(t_min, t_max)
    windows_raw = []
    pairs = [(s, e) for s in range(n) for e in range(s, n)]
    rng.shuffle(pairs)
    for s, e in pairs[:12]:
        total = sum(
            lengths[i] * (truth_x[i] + coeff[i] * truth_t) for i in range(s, e + 1)
        )
        slack = rng.randint(0, 1)
        windows_raw.append((s, e, total - slack, total + slack))
    payload = make_thermal_payload(
        lengths, strain_min, strain_max, windows_raw, coeff, t_min, t_max
    )
    result = invert_payload(payload)
    expected = brute_force_optimal_thermal(
        lengths, strain_min, strain_max, windows_raw, coeff, t_min, t_max
    )
    assert expected is not None
    (m, s_abs, xs_tuple, t), _, _, _ = expected
    assert result["strains"] == list(xs_tuple)
    assert result["thermal_compensation"]["temperature_delta"] == t
    assert result["objectives"]["max_adjacent_diff"] == m
    assert result["objectives"]["sum_adjacent_abs_diff"] == s_abs
    assert_thermal_checks(result, payload)


def test_explicit_zero_temperature_bound_single_value():
    """温变区间合法单点（含 0），跨度 0。"""
    windows = [
        {"start_segment": i, "end_segment": i, "min_elongation": 0, "max_elongation": 0}
        for i in range(1, 7)
    ]
    windows += [
        {"start_segment": 1, "end_segment": 6, "min_elongation": 0, "max_elongation": 0},
        {"start_segment": 1, "end_segment": 3, "min_elongation": 0, "max_elongation": 0},
    ]
    payload = {
        "segment_lengths": [1] * 6,
        "strain_bounds": {"min": -10, "max": 10},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": [0] * 6,
            "temperature_delta_bounds": {"min": 0, "max": 0},
        },
    }
    result = invert_payload(payload)
    assert result["thermal_compensation"]["temperature_delta"] == 0
    assert result["strains"] == [0] * 6


# --------------------------------------------------------------------------- #
# 字段级校验
# --------------------------------------------------------------------------- #
def _base_valid_payload():
    return {
        "segment_lengths": [1] * 6,
        "strain_bounds": {"min": -10, "max": 10},
        "windows": [
            {"start_segment": 1, "end_segment": 6, "min_elongation": 0, "max_elongation": 0}
        ]
        * 8,
    }


def _expect_fields(payload, expected_paths):
    with pytest.raises(ValidationErrors) as exc:
        validate(payload)
    paths = {f["field"] for f in exc.value.fields}
    for path in expected_paths:
        assert path in paths, f"missing field error for {path}; got {paths}"


def test_thermal_must_be_object():
    payload = _base_valid_payload()
    payload["thermal_compensation"] = []
    _expect_fields(payload, ["thermal_compensation"])


def test_thermal_missing_coefficients():
    payload = _base_valid_payload()
    payload["thermal_compensation"] = {"temperature_delta_bounds": {"min": 0, "max": 1}}
    _expect_fields(payload, ["thermal_compensation.coefficients"])


def test_thermal_missing_bounds():
    payload = _base_valid_payload()
    payload["thermal_compensation"] = {"coefficients": [0] * 6}
    _expect_fields(payload, ["thermal_compensation.temperature_delta_bounds"])


def test_thermal_coefficient_count_mismatch():
    payload = _base_valid_payload()
    payload["thermal_compensation"] = {
        "coefficients": [0, 0, 0],
        "temperature_delta_bounds": {"min": 0, "max": 1},
    }
    _expect_fields(payload, ["thermal_compensation.coefficients"])


def test_thermal_coefficient_negative_or_non_integer():
    payload = _base_valid_payload()
    payload["thermal_compensation"] = {
        "coefficients": [0, -1, 0, True, 1.5, 0],
        "temperature_delta_bounds": {"min": 0, "max": 1},
    }
    _expect_fields(
        payload,
        [
            "thermal_compensation.coefficients[1]",
            "thermal_compensation.coefficients[3]",
            "thermal_compensation.coefficients[4]",
        ],
    )


def test_thermal_bounds_reversed():
    payload = _base_valid_payload()
    payload["thermal_compensation"] = {
        "coefficients": [0] * 6,
        "temperature_delta_bounds": {"min": 3, "max": 1},
    }
    _expect_fields(payload, ["thermal_compensation.temperature_delta_bounds"])


def test_thermal_bounds_span_too_large():
    payload = _base_valid_payload()
    payload["thermal_compensation"] = {
        "coefficients": [0] * 6,
        "temperature_delta_bounds": {"min": 0, "max": 21},
    }
    _expect_fields(payload, ["thermal_compensation.temperature_delta_bounds"])


def test_thermal_bounds_span_exactly_20_is_valid():
    payload = _base_valid_payload()
    payload["thermal_compensation"] = {
        "coefficients": [0] * 6,
        "temperature_delta_bounds": {"min": -10, "max": 10},
    }
    lengths, smin, smax, windows, thermal = validate(payload)
    assert thermal.t_min == -10 and thermal.t_max == 10


def test_thermal_unknown_fields():
    payload = _base_valid_payload()
    payload["thermal_compensation"] = {
        "coefficients": [0] * 6,
        "temperature_delta_bounds": {"min": 0, "max": 1, "extra": 2},
        "bogus": 1,
    }
    _expect_fields(
        payload,
        [
            "thermal_compensation.bogus",
            "thermal_compensation.temperature_delta_bounds.extra",
        ],
    )


def test_thermal_max_size_performance():
    """12 段、20 窗、21 个候选温变下的联合反演耗时上限。"""
    rng = random.Random(4242)
    n = 12
    lengths = [rng.randint(1, 100) for _ in range(n)]
    coeff = [rng.randint(0, 5) for _ in range(n)]
    truth_x = [rng.randint(-500, 500) for _ in range(n)]
    t_true = rng.randint(-3, 3)
    windows = []
    starts = list(range(n))
    rng.shuffle(starts)
    for s0 in starts:
        e0 = rng.randint(s0, n - 1)
        total = sum(
            lengths[i] * (truth_x[i] + coeff[i] * t_true) for i in range(s0, e0 + 1)
        )
        windows.append(
            {"start_segment": s0 + 1, "end_segment": e0 + 1,
             "min_elongation": total - 200, "max_elongation": total + 200}
        )
    while len(windows) < 20:
        s0 = rng.randint(0, n - 1)
        e0 = rng.randint(s0, n - 1)
        total = sum(
            lengths[i] * (truth_x[i] + coeff[i] * t_true) for i in range(s0, e0 + 1)
        )
        windows.append(
            {"start_segment": s0 + 1, "end_segment": e0 + 1,
             "min_elongation": total - 100, "max_elongation": total + 100}
        )
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -100000, "max": 100000},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": coeff,
            "temperature_delta_bounds": {"min": -10, "max": 10},
        },
    }
    start = time.monotonic()
    result = invert_payload(payload)
    elapsed = time.monotonic() - start
    assert elapsed < 30.0
    assert all(c["satisfied"] for c in result["window_checks"])
    assert_thermal_checks(result, payload)
