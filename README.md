# 海缆连续缆段整数微应变联合反演服务

海缆检修组只能取得多个**重叠标距**的累计伸长量。本服务把 6–12 段缆段的
整数微应变作为未知量，对 8–20 个由「连续起止段 + 累计伸长量闭区间」组成的
观测窗做**精确整数联合反演**，避免逐窗均摊导致彼此矛盾的局部结论。

## 数学模型与优选准则

设第 i 段长度为 L_i、待求整数微应变为 x_i。每个观测窗 [s,e] 要求

    Σ_{i=s..e} L_i · x_i ∈ [min_elongation, max_elongation]

且每段都满足统一应变闭区间 `strain_bounds.min ≤ x_i ≤ strain_bounds.max`。
所有运算与比较均为 Python 任意精度**整数**，无任何浮点参与。

可行解依次最小化（三级字典序，前一级最优后才比较下一级）：

1. `max_adjacent_diff` = max_k |x_k − x_{k−1}|
2. `sum_adjacent_abs_diff` = Σ_k |x_k − x_{k−1}|
3. 应变序列 (x_0, …, x_{n−1}) 的字典序

实现方法（`app/solver.py`，仅标准库）：

- 引入前缀和 P_i = Σ_{j<i} L_j·x_j，观测窗与应变界全部化为 P 上的
  **差分约束**，以 Floyd–Warshall 最长路闭包检测正环（冲突）并导出
  每段 x_i 的紧整数域；
- |x_k−x_{k−1}|≤M 同样是差分约束，并入闭包缩域；
- 对线性窗约束做整数界传播 + MRV/折半回溯判定可行性；
- 最小 M、最小 S（辅助变量 e_k≥|Δ_k| 精确松弛）与字典序最优序列
  均利用「可行性关于阈值单调」二分钉死。

## 热补偿（可选 thermal_compensation）

海缆温度变化使累计伸长同时包含热胀与机械应变。启用热补偿后另引入一个
所有段**共同的整数温变 ΔT** 与逐段非负整数系数 c_i：

    每段观测总应变 x_i = m_i + c_i·ΔT          （m_i 为机械应变）
    观测窗 [s,e]：Σ_{i=s..e} L_i·(m_i + c_i·ΔT) ∈ [min, max]

- 观测窗仍按段长加权；**统一应变界与两级平滑指标只约束机械应变 m_i**；
- ΔT 在 `temperature_delta_bounds` 给出的共同整数闭区间内联合寻优，
  区间跨度（max−min）不得超过 20（至多 21 个取值）；
- 求解时枚举每个 ΔT，将窗界平移 −ΔT·Σ_{s..e} L_i·c_i 后退化为纯机械
  反演，在全部允许温变上联合取三级最优；联合比较键为
  `(M, S, (m_0,…,m_{n−1}), ΔT)`，故**三级完全相同时取较小温变**；
- 省略 `thermal_compensation` 时，请求、响应、三级裁决与失败语义完全不变。

调用方仅凭提交数据即可复核：

    thermal_contribution_w = ΔT · Σ_{i=s..e} L_i·c_i
    mechanical_contribution_w = Σ_{i=s..e} L_i·m_i
    weighted_strain_sum = mechanical + thermal ∈ [min, max]

即机械应变不会重复吸收共同热胀；响应还给出逐段 `thermal_strains`
（= ΔT·c_i）与 `total_strains`（= m_i + c_i·ΔT）。

## 运行（Docker Compose）

```bash
# 宿主机端口可配置（默认 8000）
HOST_PORT=9000 docker compose up --build -d api
curl -s http://localhost:9000/health
```

### 一次性验证服务 verify

`verify` 服务会等待 `api` 健康检查通过，然后依次运行：

1. 单元测试（pytest，含与全枚举暴力解的三级最优性对照，覆盖旧模式与热补偿模式）
2. 构建检查（`compileall`）与 `pip check`
3. 新旧两种模式的反演 API 冒烟（旧模式成功回算、热补偿机械/热分项回算、
   `INFEASIBLE` 含无联合解释、`INVALID_INPUT` 含热补偿字段错误）

完成后**自行退出**，退出码即结论（0 为全部通过）：

```bash
docker compose build
docker compose up --abort-on-container-exit --exit-code-from verify verify
echo "verify exit code: $?"
```

## API

### `GET /health`

返回 `{"status": "ok"}`，供容器与 Compose 健康检查使用。

### `GET /api/v1/schema`

返回请求/响应/错误码的字段约定。

### `POST /api/v1/invert`

请求体：

```json
{
  "segment_lengths": [10, 12, 11, 13, 10, 14],
  "strain_bounds": {"min": -100, "max": 100},
  "windows": [
    {"start_segment": 1, "end_segment": 6,
     "min_elongation": 100, "max_elongation": 400}
  ],
  "thermal_compensation": {
    "coefficients": [1, 2, 1, 0, 2, 1],
    "temperature_delta_bounds": {"min": -10, "max": 10}
  }
}
```

- `segment_lengths`：6–12 个正整数，按顺序排列。
- `strain_bounds`：统一应变闭区间（整数微应变）；启用热补偿时只约束机械应变。
- `windows`：8–20 个观测窗；段号 1 基且含端点，`start ≤ end`，
  `min_elongation ≤ max_elongation`。
- `thermal_compensation`：**可选**。
  - `coefficients`：逐段非负整数系数 c_i，数量必须等于段数；
  - `temperature_delta_bounds`：共同整数温变闭区间，跨度不得超过 20。
  - 省略该对象即旧模式，请求/响应/裁决/失败语义保持不变。

成功（200）：

```json
{
  "code": "OK",
  "result": {
    "segment_count": 6,
    "strains": [3, 5, 4, 2, 6, 1],
    "adjacent_diffs": [2, -1, -2, 4, -5],
    "objectives": {
      "max_adjacent_diff": 5,
      "sum_adjacent_abs_diff": 14
    },
    "window_checks": [
      {
        "index": 0,
        "start_segment": 1,
        "end_segment": 6,
        "total_length": 70,
        "min_elongation": 100,
        "max_elongation": 400,
        "weighted_strain_sum": 234,
        "satisfied": true
      }
    ],
    "criteria_order": ["max_adjacent_diff",
                       "sum_adjacent_abs_diff", "lexicographic"]
  }
}
```

每个 `weighted_strain_sum` 都能用提交的 `segment_lengths` 与返回的
`strains` 直接复核并确认落在提交闭区间内；两级平滑指标可用
`adjacent_diffs` 直接复核。

启用热补偿时（示意，字段齐全）：

```json
{
  "code": "OK",
  "result": {
    "segment_count": 6,
    "strains": [-3, -3, -3, -3, -3, -3],
    "adjacent_diffs": [0, 0, 0, 0, 0],
    "objectives": {"max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0},
    "window_checks": [
      {
        "index": 0, "start_segment": 1, "end_segment": 6, "total_length": 15,
        "min_elongation": 120, "max_elongation": 120,
        "mechanical_contribution": -45,
        "thermal_contribution": 165,
        "weighted_strain_sum": 120,
        "satisfied": true
      }
    ],
    "thermal_compensation": {
      "temperature_delta": 11,
      "coefficients": [1, 1, 1, 1, 1, 1],
      "thermal_strains": [11, 11, 11, 11, 11, 11],
      "total_strains": [8, 8, 8, 8, 8, 8]
    },
    "criteria_order": ["max_adjacent_diff",
                       "sum_adjacent_abs_diff", "lexicographic"]
  }
}
```

- `strains` 在热补偿模式下是**机械应变** m_i（统一界与平滑指标只对它成立）；
- 每窗 `mechanical_contribution + thermal_contribution = weighted_strain_sum`，
  三项都能仅凭提交数据（长度、系数、所选温变）复算；
- `thermal_strains[i] = temperature_delta × coefficients[i]`，
  `total_strains[i] = strains[i] + thermal_strains[i]`。

错误：

- `400`：请求体不是合法 JSON。
- `422 INVALID_INPUT`：字段错误，`fields[]` 逐条给出 `field` 与 `message`，
  包括 `thermal_compensation` 字段缺失、系数数量与段数不符、系数为负/非整数，
  以及温变区间非整数/上下端颠倒/跨度超过 20。
- `409 INFEASIBLE`：输入合法但不存在满足全部闭区间的解释——旧模式下是无可行
  整数应变序列；热补偿模式下是**全部允许温变与机械应变**均无联合可行解。

## 本地开发（无 Docker）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/python scripts/smoke.py
```
