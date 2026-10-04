# 海缆连续缆段整数微应变联合反演服务

海缆检修组只能取得多个**重叠标距**的累计伸长量。本服务把 6–12 段缆段的
整数微应变作为未知量，对 8–20 个由「连续起止段 + 累计伸长量闭区间」组成的
观测窗做**精确整数联合反演**，避免逐窗均摊导致彼此矛盾的局部结论。

## 数学模型与优选准则

设第 i 段长度为 L_i、待求整数**机械**微应变为 x_i。每个观测窗 [s,e] 要求

    Σ_{i=s..e} L_i · x_i ∈ [min_elongation, max_elongation]

且每段都满足统一应变闭区间 `strain_bounds.min ≤ x_i ≤ strain_bounds.max`。
所有运算与比较均为 Python 任意精度**整数**，无任何浮点参与。

### 可选：共同温变热补偿（thermal_compensation）

海缆温度变化使累计伸长同时包含热胀与机械应变。可选提交
`thermal_compensation` 后，引入一个所有段**共同**的整数温变 t：

    每段总应变 = x_i（机械） + c_i·t（热），c_i = coefficients[i]

观测窗 [s,e] 约束变为

    Σ_{i=s..e} L_i·x_i + t·(Σ_{i=s..e} L_i·c_i) ∈ [lo, hi]

即对每个候选 t 把窗界整体平移 −t·C_w（C_w = 窗内 Σ L_i·c_i）；观测窗仍按
段长加权，**统一应变界与三级平滑指标只约束机械应变 x**，热胀由 t·C_w 单独
列示，不被机械应变重复吸收。

联合裁决：先在每个候选 t 内取得机械序列的三级最优，再在全部允许温变间
**联合**取 (M, S, x, t) 字典序最优——三级完全相同时取较小温变。实现上把
温变集合视为隐式析取：最小 M、最小 S 与逐段字典序钉压都在所有候选温变上
联合二分/探测，候选温变按升序探测并随钉压逐段裁剪。

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

## 运行（Docker Compose）

```bash
# 宿主机端口可配置（默认 8000）
HOST_PORT=9000 docker compose up --build -d api
curl -s http://localhost:9000/health
```

### 一次性验证服务 verify

`verify` 服务会等待 `api` 健康检查通过，然后依次运行：

1. 单元测试（pytest，含与全枚举暴力解的三级最优性对照；热补偿模式另与
   「机械应变 × 共同温变」全枚举联合解对照）
2. 构建检查（`compileall`）与 `pip check`
3. 一组反演 API 冒烟（新旧两种模式：成功回算、`INFEASIBLE`、`INVALID_INPUT`；
   热补偿模式额外逐窗复核机械贡献、热贡献与总回算和）

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
  ]
}
```

- `segment_lengths`：6–12 个正整数，按顺序排列。
- `strain_bounds`：统一应变闭区间（整数微应变，仅约束机械应变）。
- `windows`：8–20 个观测窗；段号 1 基且含端点，`start ≤ end`，
  `min_elongation ≤ max_elongation`。
- `thermal_compensation`（可选，省略时请求/响应/裁决/失败语义完全不变）：
  - `coefficients`：逐段非负整数热系数 c_i，数量必须与段数相同；
  - `temperature_delta_bounds`：共同整数温变闭区间
    `{min, max}`，跨度 `max - min ≤ 20`。

启用热补偿的请求示例：

```json
{
  "segment_lengths": [3, 5, 2, 4, 6, 1],
  "strain_bounds": {"min": -1, "max": 1},
  "windows": [
    {"start_segment": 1, "end_segment": 1,
     "min_elongation": 30, "max_elongation": 30}
  ],
  "thermal_compensation": {
    "coefficients": [2, 2, 2, 2, 2, 2],
    "temperature_delta_bounds": {"min": 0, "max": 10}
  }
}
```

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

启用热补偿时（200）：

```json
{
  "code": "OK",
  "result": {
    "segment_count": 6,
    "strains": [0, 0, 0, 0, 0, 0],
    "adjacent_diffs": [0, 0, 0, 0, 0],
    "objectives": {"max_adjacent_diff": 0, "sum_adjacent_abs_diff": 0},
    "window_checks": [
      {
        "index": 0,
        "start_segment": 1,
        "end_segment": 1,
        "total_length": 3,
        "min_elongation": 30,
        "max_elongation": 30,
        "mechanical_weighted_sum": 0,
        "weighted_coefficient_sum": 6,
        "thermal_weighted_sum": 30,
        "weighted_strain_sum": 30,
        "satisfied": true
      }
    ],
    "thermal_compensation": {
      "temperature_delta": 5,
      "coefficients": [2, 2, 2, 2, 2, 2],
      "temperature_delta_bounds": {"min": 0, "max": 10}
    },
    "criteria_order": ["max_adjacent_diff",
                       "sum_adjacent_abs_diff", "lexicographic"]
  }
}
```

- `strains`/`adjacent_diffs`/两级 `objectives` 始终只描述**机械**应变；
- 每个观测窗分列三项，仅凭提交数据即可独立复核：
  - `mechanical_weighted_sum` = Σ 窗内 L_i·`strains[i]`；
  - `weighted_coefficient_sum` = Σ 窗内 L_i·`coefficients[i]`；
  - `thermal_weighted_sum` = `temperature_delta` · `weighted_coefficient_sum`；
  - `weighted_strain_sum`（总回算和）= 机械贡献 + 热贡献，须落入提交闭区间；
- `thermal_compensation.temperature_delta` 为联合裁决选出的共同温变；
  三级完全相同时取较小温变。

错误：

- `400`：请求体不是合法 JSON。
- `422 INVALID_INPUT`：字段错误，`fields[]` 逐条给出 `field` 与 `message`
  （含 `thermal_compensation` 对象缺失字段、系数数量不符/含负数或非整数、
  温变区间非整数/上下界颠倒/跨度超过 20 等字段级定位）。
- `409 INFEASIBLE`：输入合法但观测窗彼此冲突（含与统一应变界冲突），
  不存在满足全部闭区间的整数应变序列；启用热补偿时表示允许温变区间内
  **不存在任何联合（机械应变，共同温变）解释**。

## 本地开发（无 Docker）

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
API_BASE_URL=http://127.0.0.1:8000 .venv/bin/python scripts/smoke.py
```
