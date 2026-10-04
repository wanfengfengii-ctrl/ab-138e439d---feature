"""API 层测试：健康检查、成功反演、422 字段错误、409 不可行、坏 JSON。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def sample_payload():
    return {
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


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_schema_hint_documents_fields():
    resp = client.get("/api/v1/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert "window_checks" in body["success_response"]
    assert "INFEASIBLE" in body["errors"]


def test_invert_success_and_recompute():
    resp = client.post("/api/v1/invert", json=sample_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["code"] == "OK"
    result = body["result"]
    assert len(result["strains"]) == 6
    assert all(-100 <= v <= 100 for v in result["strains"])
    # 全部回算和均可由提交数据复核且落在提交区间内。
    payload = sample_payload()
    lengths = payload["segment_lengths"]
    strains = result["strains"]
    for check, win in zip(result["window_checks"], payload["windows"]):
        total = sum(
            lengths[i] * strains[i]
            for i in range(win["start_segment"] - 1, win["end_segment"])
        )
        assert check["weighted_strain_sum"] == total
        assert win["min_elongation"] <= total <= win["max_elongation"]
        assert check["satisfied"] is True
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    assert result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs)
    assert result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs)


def test_invert_invalid_fields_422():
    payload = sample_payload()
    payload["segment_lengths"] = [1, 2]  # 少于 6 段
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "INVALID_INPUT"
    assert any(f["field"] == "segment_lengths" for f in body["fields"])


def test_invert_infeasible_409():
    payload = sample_payload()
    payload["windows"][0] = {
        "start_segment": 1,
        "end_segment": 6,
        "min_elongation": 10**9,
        "max_elongation": 10**9,
    }
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "INFEASIBLE"


def test_bad_json_400():
    resp = client.post(
        "/api/v1/invert",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400
    assert resp.json()["code"] == "INVALID_INPUT"


# --------------------------------------------------------------------------- #
# 热补偿模式
# --------------------------------------------------------------------------- #
def thermal_payload():
    lengths = [10, 12, 11, 13, 10, 14]
    coeff = [2, 3, 1, 2, 3, 1]
    t_true = 4
    mech = [3, -1, 2, 0, -2, 1]
    spans = [(0, 5), (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (0, 2)]
    windows = []
    for s, e in spans:
        total = sum(
            lengths[i] * (mech[i] + coeff[i] * t_true) for i in range(s, e + 1)
        )
        windows.append(
            {
                "start_segment": s + 1,
                "end_segment": e + 1,
                "min_elongation": total - 3,
                "max_elongation": total + 3,
            }
        )
    return {
        "segment_lengths": lengths,
        "strain_bounds": {"min": -50, "max": 50},
        "windows": windows,
        "thermal_compensation": {
            "coefficients": coeff,
            "temperature_delta_bounds": {"min": 0, "max": 10},
        },
    }


def test_thermal_success_and_recompute():
    payload = thermal_payload()
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    chosen_t = result["thermal_compensation"]["temperature_delta"]
    assert 0 <= chosen_t <= 10
    assert result["thermal_compensation"]["coefficients"] == payload[
        "thermal_compensation"
    ]["coefficients"]
    lengths = payload["segment_lengths"]
    coeff = payload["thermal_compensation"]["coefficients"]
    strains = result["strains"]
    for check, win in zip(result["window_checks"], payload["windows"]):
        s, e = win["start_segment"] - 1, win["end_segment"] - 1
        mech_sum = sum(lengths[i] * strains[i] for i in range(s, e + 1))
        c_sum = sum(lengths[i] * coeff[i] for i in range(s, e + 1))
        heat = chosen_t * c_sum
        assert check["mechanical_weighted_sum"] == mech_sum
        assert check["weighted_coefficient_sum"] == c_sum
        assert check["thermal_weighted_sum"] == heat
        assert check["weighted_strain_sum"] == mech_sum + heat
        assert check["satisfied"] is True
        assert win["min_elongation"] <= mech_sum + heat <= win["max_elongation"]
    # 平滑指标只描述机械应变。
    diffs = [strains[i] - strains[i - 1] for i in range(1, 6)]
    assert result["objectives"]["max_adjacent_diff"] == max(abs(d) for d in diffs)
    assert result["objectives"]["sum_adjacent_abs_diff"] == sum(abs(d) for d in diffs)


def test_thermal_invalid_coefficient_count_422():
    payload = thermal_payload()
    payload["thermal_compensation"]["coefficients"] = [1, 2, 3]
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "INVALID_INPUT"
    assert any(
        f["field"] == "thermal_compensation.coefficients" for f in body["fields"]
    )


def test_thermal_illegal_bounds_422():
    payload = thermal_payload()
    payload["thermal_compensation"]["temperature_delta_bounds"] = {"min": 0, "max": 21}
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert any(
        "temperature_delta_bounds" in f["field"] for f in body["fields"]
    )


def test_thermal_joint_infeasible_409():
    """合法但任何允许温变都解释不了时返回 409 INFEASIBLE。"""
    payload = thermal_payload()
    # 全长窗要求巨量伸长，温变区间内的热+机械贡献都达不到。
    payload["windows"][0] = {
        "start_segment": 1,
        "end_segment": 6,
        "min_elongation": 10**9,
        "max_elongation": 10**9,
    }
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 409
    assert resp.json()["code"] == "INFEASIBLE"
