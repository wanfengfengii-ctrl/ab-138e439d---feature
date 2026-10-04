"""海缆整数微应变联合反演 HTTP API。"""

from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .solver import InfeasibleError, ValidationErrors, invert_payload

app = FastAPI(
    title="Submarine Cable Integer Strain Inversion API",
    version="1.0.0",
    description=(
        "对多个重叠标距的累计伸长量观测窗做精确整数联合反演，"
        "依次最小化相邻段最大应变差、相邻差绝对值之和与应变序列字典序。"
    ),
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/schema")
async def schema_hint() -> dict:
    """返回输入输出字段约定，便于调用方自查。"""
    return {
        "request": {
            "segment_lengths": "6..12 个正整数，按顺序排列的缆段长度",
            "strain_bounds": {"min": "整数微应变闭区间下端", "max": "上端（仅约束机械应变）"},
            "windows": [
                {
                    "start_segment": "起始段（1 基，含）",
                    "end_segment": "结束段（1 基，含，不小于 start_segment）",
                    "min_elongation": "累计伸长量闭区间下端（整数，机械+热）",
                    "max_elongation": "上端（整数，不小于下端）",
                }
            ],
            "window_count": "8..20",
            "thermal_compensation": "(可选) 缺省即旧模式；启用后共同温变在"
                                    " temperature_delta_bounds 内联合寻优",
            "thermal_compensation_fields": {
                "coefficients": "逐段非负整数系数 c_i，数量必须等于段数；"
                                "总应变 x_i = 机械应变 m_i + c_i·ΔT",
                "temperature_delta_bounds": {
                    "min": "共同整数温变闭区间下端",
                    "max": "上端（max-min 不得超过 20）",
                },
            },
        },
        "success_response": {
            "segment_count": "int",
            "strains": "逐段整数机械微应变 m_i（字典序最优；统一界与平滑指标只约束它）",
            "adjacent_diffs": "逐相邻段机械应变差，可直接复核两级平滑指标",
            "objectives": {
                "max_adjacent_diff": "第一级指标 = max(|adjacent_diffs|)",
                "sum_adjacent_abs_diff": "第二级指标 = sum(|adjacent_diffs|)",
            },
            "window_checks": [
                {
                    "mechanical_contribution": "(热补偿模式) = Σ 窗内长度×机械应变",
                    "thermal_contribution": "(热补偿模式) = ΔT·Σ 窗内长度×c_i",
                    "weighted_strain_sum": "= 机械贡献 + 热贡献（旧模式即长度×应变和），"
                                           "须落入提交的闭区间",
                    "total_length": "= Σ 窗内段长",
                    "satisfied": "min<=weighted_strain_sum<=max",
                }
            ],
            "thermal_compensation": {
                "temperature_delta": "联合三级最优所选共同温变；三级完全相同时取较小值",
                "coefficients": "回显提交的逐段系数，便于独立复核",
                "thermal_strains": "= temperature_delta × coefficients[i]",
                "total_strains": "= strains[i] + thermal_strains[i]（观测总应变）",
            },
            "criteria_order": [
                "max_adjacent_diff",
                "sum_adjacent_abs_diff",
                "lexicographic",
            ],
        },
        "errors": {
            "INVALID_INPUT": {"fields": [{"field": "字段路径", "message": "原因"}]},
            "INFEASIBLE": "观测窗彼此冲突，不存在满足全部闭区间的整数（机械应变, 温变）组合",
        },
    }


@app.post("/api/v1/invert")
async def invert(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={
                "code": "INVALID_INPUT",
                "message": "request body must be valid JSON",
                "fields": [{"field": ".", "message": "invalid JSON body"}],
            },
        )
    try:
        result = invert_payload(payload)
    except ValidationErrors as exc:
        return JSONResponse(
            status_code=422,
            content={
                "code": "INVALID_INPUT",
                "message": "input validation failed",
                "fields": exc.fields,
            },
        )
    except InfeasibleError as exc:
        return JSONResponse(
            status_code=409,
            content={"code": "INFEASIBLE", "message": str(exc)},
        )
    return JSONResponse(status_code=200, content={"code": "OK", "result": result})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.environ.get("APP_HOST", "0.0.0.0"),
        port=int(os.environ.get("APP_PORT", "8000")),
    )
