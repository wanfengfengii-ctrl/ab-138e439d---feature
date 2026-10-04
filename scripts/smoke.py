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


def thermal_payload() -> dict:
    """纯热胀场景：共同温变 5、系数 2，总应变 10/段，机械应变恒 0。

    单段等式窗 + 窄机械应变界 [-1,1] 使唯一联合解释为 t=5、x_i=0，
    可严格验证热胀没有被机械应变重复吸收。
    """
    lengths = [3, 5, 2, 4, 6, 1]
    coeff = [2, 2, 2, 2, 2, 2]
    t_true = 5
    spans = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 5), (0, 2)]
    windows = []
    for s, e in spans:
        total = t_true * sum(lengths[i] * coeff[i] for i in range(s, e + 1))
        windows.append(
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": total,
                "max_elongation": total,
            }
        )
    return {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -1, "max": 1},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": coeff,
            "temperature_delta_bounds": {"min": 0, "max": 10},
        },
    }


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

    print("5) thermal compensation 成功 + 机械/热贡献分列回算")
    thermal = thermal_payload()
    status, body = request("POST", "/api/v1/invert", thermal)
    check(status == 200, f"热补偿反演 -> 200 (got {status}: {body})")
    check(body.get("code") == "OK", "热补偿响应 code == OK")
    result = body["result"]
    t_info = result.get("thermal_compensation")
    check(isinstance(t_info, dict), "返回 thermal_compensation 对象")
    chosen_t = t_info.get("temperature_delta")
    check(chosen_t == 5, f"联合反演选出共同温变 5 (got {chosen_t})")
    check(
        t_info.get("coefficients") == thermal["thermal_compensation"]["coefficients"],
        "回显逐段热系数与提交一致",
    )
    lengths = thermal["segment_lengths"]
    coeff = thermal["thermal_compensation"]["coefficients"]
    strains = result["strains"]
    check(all(v == 0 for v in strains), "热胀被完全剥离：机械应变恒 0")
    check(
        result["objectives"]
        == {"max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0},
        "机械应变两级平滑指标均为 0（热胀未被机械重复吸收）",
    )
    for check_item, win in zip(result["window_checks"], thermal["windows"]):
        s = win["start_segment"] - 1
        e = win["end_segment"] - 1
        mech = sum(lengths[i] * strains[i] for i in range(s, e + 1))
        c_sum = sum(lengths[i] * coeff[i] for i in range(s, e + 1))
        heat = chosen_t * c_sum
        check(
            check_item["weighted_coefficient_sum"] == c_sum,
            f"窗[{s + 1},{e + 1}] 窗内 ΣL·c = {c_sum} 可由提交数据复算",
        )
        check(
            check_item["mechanical_weighted_sum"] == mech == 0,
            f"窗[{s + 1},{e + 1}] 机械贡献 = 0",
        )
        check(
            check_item["thermal_weighted_sum"] == heat,
            f"窗[{s + 1},{e + 1}] 热贡献 = t·ΣL·c = {heat}",
        )
        total = mech + heat
        check(
            check_item["weighted_strain_sum"] == total,
            f"窗[{s + 1},{e + 1}] 总回算和 = 机械 + 热 = {total}",
        )
        check(
            win["min_elongation"] <= total <= win["max_elongation"],
            f"窗[{s + 1},{e + 1}] 总回算和 {total} 落入提交闭区间",
        )

    print("6) thermal compensation 字段错误 -> 422")
    bad = thermal_payload()
    bad["thermal_compensation"]["coefficients"] = [1, 2, 3]  # 数量不符
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 422, f"系数数量不符 -> 422 (got {status})")
    check(
        any(
            f["field"] == "thermal_compensation.coefficients"
            for f in body.get("fields", [])
        ),
        "字段级错误定位到 thermal_compensation.coefficients",
    )
    bad = thermal_payload()
    bad["thermal_compensation"]["temperature_delta_bounds"] = {"min": 0, "max": 21}
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 422, f"温变区间跨度过大 -> 422 (got {status})")

    print("7) thermal compensation 合法但无联合解释 -> 409")
    bad = thermal_payload()
    bad["windows"][0] = {
        "start_segment": 1,
        "end_segment": 6,
        "min_elongation": 10**9,
        "max_elongation": 10**9,
    }
    status, body = request("POST", "/api/v1/invert", bad)
    check(status == 409, f"联合不可行 -> 409 (got {status})")
    check(body.get("code") == "INFEASIBLE", "错误码 == INFEASIBLE")

    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
