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


def _thermal_payload(t_min=0, t_max=20):
    """每段总应变钉为 8、机械界 [-3,3]、c_i=1；需 ΔT∈[5,11]，最优 m=-3, ΔT=11。"""
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


def test_invert_thermal_success_and_recompute():
    payload = _thermal_payload()
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    assert result["strains"] == [-3] * 6
    tc = result["thermal_compensation"]
    assert tc["temperature_delta"] == 11
    assert tc["thermal_strains"] == [11] * 6
    assert tc["total_strains"] == [8] * 6
    lengths = payload["segment_lengths"]
    coeffs = payload["thermal_compensation"]["coefficients"]
    for check, win in zip(result["window_checks"], payload["windows"]):
        s, e = win["start_segment"] - 1, win["end_segment"] - 1
        mech = sum(lengths[i] * result["strains"][i] for i in range(s, e + 1))
        therm = 11 * sum(lengths[i] * coeffs[i] for i in range(s, e + 1))
        assert check["mechanical_contribution"] == mech
        assert check["thermal_contribution"] == therm
        assert check["weighted_strain_sum"] == mech + therm
        assert check["satisfied"] is True


def test_invert_thermal_no_joint_explanation_409():
    resp = client.post("/api/v1/invert", json=_thermal_payload(0, 2))
    assert resp.status_code == 409
    assert resp.json()["code"] == "INFEASIBLE"


def test_invert_thermal_invalid_fields_422():
    payload = _thermal_payload()
    payload["thermal_compensation"]["coefficients"] = [1] * 5  # 数量不符
    payload["thermal_compensation"]["temperature_delta_bounds"] = {"min": 0, "max": 21}
    resp = client.post("/api/v1/invert", json=payload)
    assert resp.status_code == 422
    paths = {f["field"] for f in resp.json()["fields"]}
    assert "thermal_compensation.coefficients" in paths
    assert "thermal_compensation.temperature_delta_bounds" in paths
