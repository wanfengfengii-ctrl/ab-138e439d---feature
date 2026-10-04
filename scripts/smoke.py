"""对运行中的反演 API 做端到端冒烟（仅标准库）。

由 verify 一次性服务在 api 健康后执行。逐项回算每个观测窗的长度加权
应变和、两级平滑指标，全部由提交数据直接复核；任一不符即以非零退出。
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def request(method: str, path: str, body: object | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"  [FAIL] {message}")
        raise SystemExit(1)
    print(f"  [ok]   {message}")


def feasible_payload() -> dict:
    lengths = [10, 12, 11, 13, 10, 14]
    truth = [3, 5, 4, 2, 6, 1]
    spans = [(0, 5), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 2)]
    windows = []
    for s, e in spans:
        total = sum(lengths[i] * truth[i] for i in range(s, e + 1))
        windows.append(
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": total - 5,
                "max_elongation": total + 30,
            }
        )
    return {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -50, "max": 50},
        "windows": windows,
    }


def thermal_feasible_payload() -> tuple[dict, int]:
    """热补偿冒烟实例：逐段总应变被紧窗钉为 8，机械界 [-3,3]，系数全 1。

    m_i + τ = 8 且 m_i∈[-3,3] ⇒ τ∈[5,11]；三级最优取恒定机械序列（M=S=0）
    中字典序最小的 m=-3，对应 τ=11。返回 (payload, 期望温变)。
    """
    lengths = [2, 3, 1, 4, 2, 3]
    windows = [
        {
            "start_segment": 1,
            "end_segment": 6,
            "min_elongation": sum(lengths) * 8,
            "max_elongation": sum(lengths) * 8,
        }
    ]
    for i in range(1, 7):
        windows.append(
            {
                "start_segment": i,
                "end_segment": i,
                "min_elongation": lengths[i - 1] * 8,
                "max_elongation": lengths[i - 1] * 8,
            }
        )
    windows.append(
        {
            "start_segment": 2,
            "end_segment": 4,
            "min_elongation": sum(lengths[1:4]) * 8,
            "max_elongation": sum(lengths[1:4]) * 8,
        }
    )
    payload = {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -3, "max": 3},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": [1] * 6,
            "temperature_delta_bounds": {"min": 0, "max": 20},
        },
    }
    return payload, 11


def main() -> None:
    print(f"smoke against {BASE}")

    print("1) health")
    status, body = request("GET", "/health")
    check(status == 200, f"GET /health -> 200 (got {status})")
    check(body.get("status") == "ok", "health payload status == ok")

    print("2) successful inversion + 证据回算")
    payload = feasible_payload()
    status, body = request("POST", "/api/v1/invert", payload)
    check(status == 200, f"POST /api/v1/invert -> 200 (got {status}: {body})")
    check(body.get("code") == "OK", "response code == OK")
    result = body["result"]
    lengths = payload["segment_lengths"]
    strains = result["strains"]
    check(len(strains) == 6, "返回 6 段应变")
    check(
        all(payload["strain_bounds"]["min"] <= v <= payload["strain_bounds"]["max"]
            for v in strains),
        "每段应变均落入统一应变闭区间",
    )
    for check_item, win in zip(result["window_checks"], payload["windows"]):
        total = sum(
            lengths[i] * strains[i]
            for i in range(win["start_segment"] - 1, win["end_segment"])
        )
        check(
            check_item["weighted_strain_sum"] == total,
            f"窗[{win['start_segment']},{win['end_segment']}] 回算和 {total} 与响应一致",
        )
        check(
            win["min_elongation"] <= total <= win["max_elongation"],
            f"窗[{win['start_segment']},{win['end_segment']}] {total} ∈ 提交闭区间",
        )
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    check(
        result["adjacent_diffs"] == diffs,
        "adjacent_diffs 可由应变序列直接复核",
    )
    check(
        result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs),
        "一级指标 max_adjacent_diff 可由相邻差复核",
    )
    check(
        result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs),
        "二级指标 sum_adjacent_abs_diff 可由相邻差复核",
    )
    check(
        "thermal_compensation" not in result,
        "旧模式响应不包含 thermal_compensation 字段",
    )

    print("2b) thermal compensation + 机械/热分项回算")
    payload_t, expected_tau = thermal_feasible_payload()
    status, body = request("POST", "/api/v1/invert", payload_t)
    check(status == 200, f"热补偿反演 -> 200 (got {status}: {body})")
    check(body.get("code") == "OK", "热补偿响应 code == OK")
    result = body["result"]
    lengths = payload_t["segment_lengths"]
    coeffs = payload_t["thermal_compensation"]["coefficients"]
    tau = result["thermal_compensation"]["temperature_delta"]
    ms = result["strains"]
    t_bounds = payload_t["thermal_compensation"]["temperature_delta_bounds"]
    check(
        t_bounds["min"] <= tau <= t_bounds["max"],
        f"所选温变 {tau} 落在提交闭区间 [{t_bounds['min']},{t_bounds['max']}]",
    )
    for check_item, win in zip(result["window_checks"], payload_t["windows"]):
        s = win["start_segment"] - 1
        e = win["end_segment"] - 1
        mech = sum(lengths[i] * ms[i] for i in range(s, e + 1))
        therm = tau * sum(lengths[i] * coeffs[i] for i in range(s, e + 1))
        check(
            check_item["mechanical_contribution"] == mech,
            f"热模式窗[{win['start_segment']},{win['end_segment']}] 机械贡献 {mech} 可复算",
        )
        check(
            check_item["thermal_contribution"] == therm,
            f"热模式窗[{win['start_segment']},{win['end_segment']}] 热贡献 {therm} 可复算",
        )
        check(
            check_item["weighted_strain_sum"] == mech + therm,
            f"热模式窗[{win['start_segment']},{win['end_segment']}] 总和 = 机械+热",
        )
        check(
            win["min_elongation"] <= mech + therm <= win["max_elongation"],
            f"热模式窗[{win['start_segment']},{win['end_segment']}] 总和落入提交闭区间",
        )
    tc = result["thermal_compensation"]
    check(tc["temperature_delta"] == expected_tau, f"联合三级最优温变 == {expected_tau}")
    check(
        tc["thermal_strains"] == [tau * c for c in coeffs],
        "逐段热应变 = 温变 × 系数，可凭提交数据复算",
    )
    check(
        tc["total_strains"] == [ms[i] + tau * coeffs[i] for i in range(len(ms))],
        "逐段总应变 = 机械应变 + 热应变",
    )
    mech_diffs = [ms[i] - ms[i - 1] for i in range(1, len(ms))]
    check(
        result["adjacent_diffs"] == mech_diffs,
        "平滑指标只由机械应变相邻差复核（机械未吸收共同热胀）",
    )
    check(
        result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in mech_diffs),
        "热模式一级指标可由机械相邻差复核",
    )
    check(
        all(payload_t["strain_bounds"]["min"] <= v <= payload_t["strain_bounds"]["max"]
            for v in ms),
        "每段机械应变均落入统一应变闭区间",
    )

    print("2c) thermal: 无联合解释 -> INFEASIBLE")
    bad_t = dict(payload_t)
    bad_t["thermal_compensation"] = {
        "coefficients": coeffs,
        "temperature_delta_bounds": {"min": expected_tau + 5, "max": expected_tau + 10},
    }
    status, body = request("POST", "/api/v1/invert", bad_t)
    check(status == 409, f"温变区间排除全部解释 -> 409 (got {status})")
    check(body.get("code") == "INFEASIBLE", "错误码 == INFEASIBLE")

    print("2d) thermal: 字段错误 -> 422")
    bad_t = dict(payload_t)
    bad_t["thermal_compensation"] = {
        "coefficients": coeffs[:-1],  # 系数数量与段数不符
        "temperature_delta_bounds": {"min": 0, "max": 21},  # 跨度非法
    }
    status, body = request("POST", "/api/v1/invert", bad_t)
    check(status == 422, f"热补偿非法输入 -> 422 (got {status})")
    check(body.get("code") == "INVALID_INPUT", "错误码 == INVALID_INPUT")
    paths = {f["field"] for f in body.get("fields", [])}
    check("thermal_compensation.coefficients" in paths, "指出系数数量不符")
    check(
        "thermal_compensation.temperature_delta_bounds" in paths,
        "指出温变区间跨度非法",
    )

    print("3) conflicting windows -> INFEASIBLE")
    bad = feasible_payload()
    bad["windows"][0] = {
        "start_segment": 1,
        "end_segment": 6,
        "min_elongation": 10**9,
        "max_elongation": 10**9,
    }
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 409, f"冲突观测 -> 409 (got {status})")
    check(body.get("code") == "INFEASIBLE", "错误码 == INFEASIBLE")

    print("4) invalid input -> 字段错误")
    bad = feasible_payload()
    bad["segment_lengths"] = [1, 2, 3]
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 422, f"非法输入 -> 422 (got {status})")
    check(body.get("code") == "INVALID_INPUT", "错误码 == INVALID_INPUT")
    check(bool(body.get("fields")), "返回明确 fields 列表")

    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
