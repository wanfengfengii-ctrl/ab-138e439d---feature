"""海缆连续缆段整数微应变联合反演核心（仅使用 Python 标准库）。

输入
----
- 6..12 段按顺序排列的缆段长度（正整数）
- 统一应变闭区间 [strain_min, strain_max]（整数微应变）
- 8..20 个观测窗，每窗给出连续起止段（1 基，含端点）与累计伸长量闭区间
- 可选 thermal_compensation：逐段非负整数热系数 c_i 与共同整数温变闭区间
  [t_min, t_max]（跨度不超过 20）

模型（所有比较与运算均为 Python 任意精度精确整数）
--------------------------------------------------
待求逐段**机械**应变为 x_0..x_{n-1}（整数微应变）。

无热补偿时观测窗 [s,e] 约束：Σ_{i=s..e} L_i·x_i ∈ [lo, hi]。
启用热补偿后另引入共同温变 t ∈ [t_min, t_max]，每段观测总应变为
x_i + c_i·t，观测窗约束变为

    Σ_{i=s..e} L_i·x_i ∈ [lo - t·C_w, hi - t·C_w]，
    C_w = Σ_{i=s..e} L_i·c_i

即对每个候选 t 把窗界整体平移 -t·C_w；统一应变界与三级平滑指标始终
**只约束机械应变 x**，热胀不被机械应变重复吸收。

每段还须满足 strain_min ≤ x_i ≤ strain_max。

引入前缀和 P_0=0, P_i = Σ_{j<i} L_j·x_j，则
    P_{e+1} - P_s ∈ [lo, hi]
    P_{i+1} - P_i ∈ [L_i·strain_min, L_i·strain_max]
全部是 P 上的**差分约束**，其最长路闭包（Floyd-Warshall）：
- 出现正环 ⇒ 整体不可行（可靠的充分必要判定，针对松弛后的实值域；
  整数可行性仍以下层整数搜索为准）；
- 给出每个 P_i 的精确上下界，进而导出每段 x_i 的紧整数域：
      x_i ≥ ceil( D[i][i+1] / L_i )
      x_i ≤ floor( -D[i+1][i] / L_i )

优选准则（字典序三级，逐级不可放宽；均只针对机械应变 x）
1. 最小化相邻段最大应变差 M = max_k |x_k - x_{k-1}|
   —— 在 x 空间就是差分约束 x_k - x_{k-1} ∈ [-M, M]，同样并入最长路闭包。
2. 在 1 的最优解中最小化 S = Σ_k |x_k - x_{k-1}|
   —— 辅助变量 e_k ≥ |Δ_k|，以 Σ e_k ≤ S 的线性松弛精确等价判定。
3. 在 1、2 的最优解中取应变序列 (x_0,...,x_{n-1}) 字典序最小者。
启用热补偿时，先在每个候选共同温变 t 内取得上述三级最优机械序列，再在
全部允许温变间联合取 (M, S, x, t) 字典序最优——三级完全相同时取较小温变。

求解：最长路闭包缩域 + 通用整数界传播 + MRV/折半回溯；
最小 M、最小 S 及字典序各值均以「可行性关于阈值单调」二分得到。
"""

from __future__ import annotations

from dataclasses import dataclass

MIN_SEGMENTS = 6
MAX_SEGMENTS = 12
MIN_WINDOWS = 8
MAX_WINDOWS = 20
MAX_TEMPERATURE_SPAN = 20


class ValidationErrors(ValueError):
    """输入字段错误（可一次性包含多个字段问题）。"""

    def __init__(self, fields: list[dict[str, str]]):
        self.fields = fields
        super().__init__("; ".join(f"{f['field']}: {f['message']}" for f in fields))


class InfeasibleError(ValueError):
    """全部输入合法，但观测窗彼此冲突（含统一应变界），无可行解。"""


@dataclass(frozen=True)
class Window:
    start: int  # 0 基，含端点
    end: int  # 0 基，含端点
    lo: int  # 累计伸长量（长度加权总应变和）闭区间下端
    hi: int  # 闭区间上端


@dataclass(frozen=True)
class ThermalCompensation:
    coefficients: tuple[int, ...]  # 逐段非负整数热系数 c_i
    t_min: int  # 共同整数温变闭区间下端
    t_max: int  # 上端（t_max - t_min <= MAX_TEMPERATURE_SPAN）


# --------------------------------------------------------------------------- #
# 输入校验
# --------------------------------------------------------------------------- #
def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _err(fields: list[dict[str, str]], field: str, message: str) -> None:
    fields.append({"field": field, "message": message})


def _validate_lengths(payload: dict, fields: list[dict[str, str]]) -> list[int] | None:
    raw = payload.get("segment_lengths")
    if raw is None:
        _err(fields, "segment_lengths", "field is required")
        return None
    if not isinstance(raw, list):
        _err(fields, "segment_lengths", "must be an array of positive integers")
        return None
    lengths: list[int] = []
    for i, item in enumerate(raw):
        if not _is_int(item) or item < 1:
            _err(fields, f"segment_lengths[{i}]", "must be a positive integer")
        else:
            lengths.append(item)
    if not (MIN_SEGMENTS <= len(raw) <= MAX_SEGMENTS):
        _err(
            fields,
            "segment_lengths",
            f"must contain between {MIN_SEGMENTS} and {MAX_SEGMENTS} segments"
            f" (got {len(raw)})",
        )
    return lengths if len(lengths) == len(raw) else None


def _validate_strain_bounds(
    payload: dict, fields: list[dict[str, str]]
) -> tuple[int, int] | None:
    raw = payload.get("strain_bounds")
    if raw is None:
        _err(fields, "strain_bounds", "field is required")
        return None
    if not isinstance(raw, dict):
        _err(fields, "strain_bounds", 'must be an object {"min": int, "max": int}')
        return None
    for key in raw:
        if key not in ("min", "max"):
            _err(fields, f"strain_bounds.{key}", "unknown field")
    lo = raw.get("min")
    hi = raw.get("max")
    ok = True
    if not _is_int(lo):
        _err(fields, "strain_bounds.min", "must be an integer")
        ok = False
    if not _is_int(hi):
        _err(fields, "strain_bounds.max", "must be an integer")
        ok = False
    if ok and lo > hi:
        _err(fields, "strain_bounds", "min must be less than or equal to max")
        return None
    return (lo, hi) if ok else None


def _validate_windows(
    payload: dict, n: int | None, fields: list[dict[str, str]]
) ->list[Window] | None:
    raw = payload.get("windows")
    if raw is None:
        _err(fields, "windows", "field is required")
        return None
    if not isinstance(raw, list):
        _err(fields, "windows", "must be an array of observation windows")
        return None
    if not (MIN_WINDOWS <= len(raw) <= MAX_WINDOWS):
        _err(
            fields,
            "windows",
            f"must contain between {MIN_WINDOWS} and {MAX_WINDOWS} windows"
            f" (got {len(raw)})",
        )
    windows: list[Window] = []
    all_ok = True
    for i, item in enumerate(raw):
        prefix = f"windows[{i}]"
        if not isinstance(item, dict):
            _err(fields, prefix, "must be an object")
            all_ok = False
            continue
        ok = True
        for key in ("start_segment", "end_segment", "min_elongation", "max_elongation"):
            if key not in item:
                _err(fields, f"{prefix}.{key}", "field is required")
                ok = False
            elif not _is_int(item[key]):
                _err(fields, f"{prefix}.{key}", "must be an integer")
                ok = False
        for key in item:
            if key not in ("start_segment", "end_segment", "min_elongation", "max_elongation"):
                _err(fields, f"{prefix}.{key}", "unknown field")
                ok = False
        if not ok:
            all_ok = False
            continue
        s = item["start_segment"]
        e = item["end_segment"]
        lo = item["min_elongation"]
        hi = item["max_elongation"]
        if n is not None:
            if not (1 <= s <= n):
                _err(fields, f"{prefix}.start_segment", f"must be between 1 and {n}")
                ok = False
            if not (1 <= e <= n):
                _err(fields, f"{prefix}.end_segment", f"must be between 1 and {n}")
                ok = False
            if ok and s > e:
                _err(fields, f"{prefix}.start_segment", "must be <= end_segment")
                ok = False
        if lo > hi:
            _err(
                fields,
                f"{prefix}.min_elongation",
                "must be less than or equal to max_elongation",
            )
            ok = False
        if ok and n is not None:
            windows.append(Window(start=s - 1, end=e - 1, lo=lo, hi=hi))
        else:
            all_ok = False
    return windows if all_ok else None


_ALLOWED_TOP_LEVEL = {"segment_lengths", "strain_bounds", "windows", "thermal_compensation"}


def _validate_thermal(
    payload: dict, n: int | None, fields: list[dict[str, str]]
) -> ThermalCompensation | None:
    """校验可选 thermal_compensation；省略时返回 None（保持旧语义）。"""
    raw = payload.get("thermal_compensation")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        _err(
            fields,
            "thermal_compensation",
            'must be an object {"coefficients": [non-negative int], '
            '"temperature_delta_bounds": {"min": int, "max": int}}',
        )
        return None
    for key in raw:
        if key not in ("coefficients", "temperature_delta_bounds"):
            _err(fields, f"thermal_compensation.{key}", "unknown field")
    coeff_ok = True
    coefficients: tuple[int, ...] | None = None
    coeff_raw = raw.get("coefficients")
    if coeff_raw is None:
        _err(fields, "thermal_compensation.coefficients", "field is required")
        coeff_ok = False
    elif not isinstance(coeff_raw, list):
        _err(
            fields,
            "thermal_compensation.coefficients",
            "must be an array of non-negative integers",
        )
        coeff_ok = False
    else:
        values: list[int] = []
        for i, item in enumerate(coeff_raw):
            if not _is_int(item) or item < 0:
                _err(
                    fields,
                    f"thermal_compensation.coefficients[{i}]",
                    "must be a non-negative integer",
                )
                coeff_ok = False
            else:
                values.append(item)
        if n is not None and len(coeff_raw) != n:
            _err(
                fields,
                "thermal_compensation.coefficients",
                f"must contain exactly {n} coefficients, one per segment"
                f" (got {len(coeff_raw)})",
            )
            coeff_ok = False
        coefficients = tuple(values) if coeff_ok else None

    t_lo = t_hi = None
    bounds_ok = True
    bounds_raw = raw.get("temperature_delta_bounds")
    if bounds_raw is None:
        _err(fields, "thermal_compensation.temperature_delta_bounds", "field is required")
        bounds_ok = False
    elif not isinstance(bounds_raw, dict):
        _err(
            fields,
            "thermal_compensation.temperature_delta_bounds",
            'must be an object {"min": int, "max": int}',
        )
        bounds_ok = False
    else:
        for key in bounds_raw:
            if key not in ("min", "max"):
                _err(
                    fields,
                    f"thermal_compensation.temperature_delta_bounds.{key}",
                    "unknown field",
                )
                bounds_ok = False
        t_lo = bounds_raw.get("min")
        t_hi = bounds_raw.get("max")
        if not _is_int(t_lo):
            _err(fields, "thermal_compensation.temperature_delta_bounds.min", "must be an integer")
            bounds_ok = False
        if not _is_int(t_hi):
            _err(fields, "thermal_compensation.temperature_delta_bounds.max", "must be an integer")
            bounds_ok = False
        if bounds_ok and t_lo > t_hi:
            _err(
                fields,
                "thermal_compensation.temperature_delta_bounds",
                "min must be less than or equal to max",
            )
            bounds_ok = False
        if bounds_ok and t_hi - t_lo > MAX_TEMPERATURE_SPAN:
            _err(
                fields,
                "thermal_compensation.temperature_delta_bounds",
                f"closed interval span must not exceed {MAX_TEMPERATURE_SPAN}"
                f" (got {t_hi - t_lo})",
            )
            bounds_ok = False
    if coeff_ok and bounds_ok and coefficients is not None:
        assert t_lo is not None and t_hi is not None
        return ThermalCompensation(coefficients=coefficients, t_min=t_lo, t_max=t_hi)
    return None


def validate(
    payload: object,
) -> tuple[list[int], int, int, list[Window], ThermalCompensation | None]:
    """校验并归一化输入；非法时抛 ValidationErrors。"""
    fields: list[dict[str, str]] = []
    if not isinstance(payload, dict):
        raise ValidationErrors(
            [{"field": ".", "message": "request body must be a JSON object"}]
        )
    for key in payload:
        if key not in _ALLOWED_TOP_LEVEL:
            _err(fields, key, "unknown field")
    lengths = _validate_lengths(payload, fields)
    bounds = _validate_strain_bounds(payload, fields)
    n = len(lengths) if lengths is not None else None
    windows = _validate_windows(payload, n, fields)
    thermal = _validate_thermal(payload, n, fields)
    if fields:
        raise ValidationErrors(fields)
    assert lengths is not None and bounds is not None and windows is not None
    return lengths, bounds[0], bounds[1], windows, thermal


# --------------------------------------------------------------------------- #
# 差分约束最长路闭包
# --------------------------------------------------------------------------- #
def _longest_path_closure(
    node_count: int, edges: list[tuple[int, int, int]]
) -> list[list[int | None]] | None:
    """约束 P_v >= P_u + w 的最长路闭包；存在正环返回 None。

    D[i][j] = P_j - P_i 的最强（最大）下界；None 表示尚无路径（-∞）。
    D[i][i] > 0 即正环，矛盾。以 None 为哨兵可兼容任意精度输入。
    """
    d: list[list[int | None]] = [[None] * node_count for _ in range(node_count)]
    for i in range(node_count):
        d[i][i] = 0
    for u, v, w in edges:
        if d[u][v] is None or w > d[u][v]:
            d[u][v] = w
    for k in range(node_count):
        dk = d[k]
        for i in range(node_count):
            dik = d[i][k]
            if dik is None:
                continue
            di = d[i]
            for j in range(node_count):
                if dk[j] is None:
                    continue
                cand = dik + dk[j]
                if di[j] is None or cand > di[j]:
                    di[j] = cand
    for i in range(node_count):
        if d[i][i] is not None and d[i][i] > 0:
            return None
    return d


def _ceil_div(a: int, b: int) -> int:
    """数学上的 ceil(a / b)（b>0），纯整数。"""
    return -((-a) // b)


def _prefix_closure(
    n: int, lengths: list[int], strain_min: int, strain_max: int,
    windows: list[Window],
) -> list[list[int]] | None:
    """前缀和 P 上的差分约束闭包（窗 + 每段应变界）。"""
    edges: list[tuple[int, int, int]] = []
    for i, length in enumerate(lengths):  # P_{i+1} - P_i ∈ [L·smin, L·smax]
        edges.append((i, i + 1, length * strain_min))
        edges.append((i + 1, i, -length * strain_max))
    for win in windows:  # P_{e+1} - P_s ∈ [lo, hi]
        edges.append((win.start, win.end + 1, win.lo))
        edges.append((win.end + 1, win.start, -win.hi))
    return _longest_path_closure(n + 1, edges)


def _x_bounds_from_prefix(
    n: int,
    lengths: list[int],
    strain_min: int,
    strain_max: int,
    p_closure: list[list[int]],
) -> tuple[list[int], list[int]]:
    """由 P 的最强界导出每段 x_i 的紧整数域。"""
    los: list[int] = []
    his: list[int] = []
    for i, length in enumerate(lengths):
        forward = p_closure[i][i + 1]
        backward = p_closure[i + 1][i]
        # i -> i+1 与 i+1 -> i 均为直接边，闭包后必非 None。
        assert forward is not None and backward is not None
        lo = max(strain_min, _ceil_div(forward, length))
        hi = min(strain_max, (-backward) // length)
        los.append(lo)
        his.append(hi)
    return los, his


def _x_diff_closure(
    n: int, los: list[int], his: list[int], m: int
) -> tuple[list[int], list[int]] | None:
    """并入一元域与 |x_k-x_{k-1}|<=M 后的 x 最长路闭包（锚点节点 n）。"""
    anchor = n
    edges: list[tuple[int, int, int]] = []
    for i in range(n):  # x_i >= anchor + lo_i；anchor >= x_i - hi_i
        edges.append((anchor, i, los[i]))
        edges.append((i, anchor, -his[i]))
    for k in range(1, n):
        edges.append((k - 1, k, -m))  # x_k >= x_{k-1} - M
        edges.append((k, k - 1, -m))  # x_{k-1} >= x_k - M
    closure = _longest_path_closure(n + 1, edges)
    if closure is None:
        return None
    out_lo = [closure[anchor][i] if closure[anchor][i] is not None else los[i] for i in range(n)]
    out_hi = [-closure[i][anchor] if closure[i][anchor] is not None else his[i] for i in range(n)]
    for i in range(n):
        if out_lo[i] > out_hi[i]:
            return None
    return out_lo, out_hi


# --------------------------------------------------------------------------- #
# 通用整数界传播 + 回溯（x 空间）
# --------------------------------------------------------------------------- #
# 约束：sum_j w_j * var_j ∈ [lo, hi]，端点可为 None。
Constraint = tuple[tuple[tuple[int, int], ...], int | None, int | None]


def _propagate(
    los: list[int], his: list[int], constraints: list[Constraint]
) -> tuple[list[int], list[int]] | None:
    """对所有线性区间约束反复做一元界传播；矛盾返回 None。"""
    los = list(los)
    his = list(his)
    while True:
        changed = False
        for terms, a, b in constraints:
            for j, w in terms:
                # 即使域已为单点也不能跳过：后加入的约束可能与其冲突。
                others_min = 0
                others_max = 0
                for j2, w2 in terms:
                    if j2 == j:
                        continue
                    if w2 > 0:
                        others_min += w2 * los[j2]
                        others_max += w2 * his[j2]
                    else:
                        others_min += w2 * his[j2]
                        others_max += w2 * los[j2]
                if b is not None:  # w*x_j <= b - others_min
                    c = b - others_min
                    if w > 0:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                    else:
                        v = _ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                if a is not None:  # w*x_j >= a - others_max
                    c = a - others_max
                    if w > 0:
                        v = _ceil_div(c, w)
                        if v > los[j]:
                            los[j] = v
                            changed = True
                    else:
                        v = c // w
                        if v < his[j]:
                            his[j] = v
                            changed = True
                if los[j] > his[j]:
                    return None
        if not changed:
            return los, his


def _search(
    los: list[int], his: list[int], constraints: list[Constraint]
) -> tuple[int, ...] | None:
    """找一个可行赋值（MRV 变量序，域折半分支）；无可行解返回 None。"""
    narrowed = _propagate(los, his, constraints)
    if narrowed is None:
        return None
    los, his = narrowed
    best = -1
    best_size = 0
    for j in range(len(los)):
        if los[j] != his[j]:
            size = his[j] - los[j]
            if best == -1 or size < best_size:
                best = j
                best_size = size
    if best == -1:
        return tuple(los)
    j = best
    mid = (los[j] + his[j]) // 2
    lo2, hi2 = list(los), list(his)
    hi2[j] = mid
    result = _search(lo2, hi2, constraints)
    if result is not None:
        return result
    lo3, hi3 = list(los), list(his)
    lo3[j] = mid + 1
    return _search(lo3, hi3, constraints)


# --------------------------------------------------------------------------- #
# 反演
# --------------------------------------------------------------------------- #
def _window_constraints(n: int, lengths: list[int], windows: list[Window]) -> list[Constraint]:
    cons: list[Constraint] = []
    for win in windows:
        terms = tuple(
            (i, lengths[i]) for i in range(win.start, win.end + 1)
        )
        cons.append((terms, win.lo, win.hi))
    return cons


def _diff_constraints(n: int, m: int) -> list[Constraint]:
    """|x_k - x_{k-1}| <= M（亦供通用传播，与差分闭包一致）。"""
    cons: list[Constraint] = []
    for k in range(1, n):
        cons.append((((k, 1), (k - 1, -1)), -m, m))
    return cons


def _abs_sum_constraints(n: int, m: int, s_lo: int | None, s_hi: int | None) -> list[Constraint]:
    """e_k >= x_k-x_{k-1}、e_k >= x_{k-1}-x_k、0<=e_k<=M、Σe_k∈[s_lo,s_hi]。"""
    cons: list[Constraint] = []
    for k in range(1, n):
        e_var = n + k - 1
        cons.append((((e_var, 1), (k, -1), (k - 1, 1)), 0, None))
        cons.append((((e_var, 1), (k, 1), (k - 1, -1)), 0, None))
        cons.append((((e_var, 1),), 0, m))
    cons.append((tuple((n + k - 1, 1) for k in range(1, n)), s_lo, s_hi))
    return cons


def _solve_mechanical(
    lengths: list[int],
    strain_min: int,
    strain_max: int,
    windows: list[Window],
) -> tuple[list[int], list[int], int, int]:
    """求给定观测窗下机械应变的三级最优解。

    输入合法但无任何可行机械序列时抛 InfeasibleError。
    """
    n = len(lengths)
    window_cons = _window_constraints(n, lengths, windows)

    # 前缀和差分闭包：与 M 无关，只算一次。
    p_closure = _prefix_closure(n, lengths, strain_min, strain_max, windows)
    if p_closure is None:
        raise InfeasibleError("observation windows are mutually inconsistent")
    base_x_lo, base_x_hi = _x_bounds_from_prefix(
        n, lengths, strain_min, strain_max, p_closure
    )
    full_range = strain_max - strain_min

    def feasible(m: int, s_hi: int | None = None, pin: tuple[int, int] | None = None) -> bool:
        """|Δ|<=m（若给 s_hi 则 Σ|Δ|<=s_hi），可附加 x_var<=t 的钉压。"""
        bounds = _x_diff_closure(n, base_x_lo, base_x_hi, m)
        if bounds is None:
            return False
        x_lo, x_hi = bounds
        constraints = list(window_cons)
        constraints += _diff_constraints(n, m)
        if s_hi is not None:
            x_lo = x_lo + [0] * (n - 1)
            x_hi = x_hi + [m] * (n - 1)
            constraints += _abs_sum_constraints(n, m, None, s_hi)
        if pin is not None:
            var, t = pin
            constraints.append((((var, 1),), None, t))
        return _search(list(x_lo), list(x_hi), constraints) is not None

    # 阶段 0：无平滑约束（M = 全量程）下的整数可行性。
    if not feasible(full_range):
        raise InfeasibleError("observation windows are mutually inconsistent")

    # 阶段 1：二分最小可行 M。
    m_lo, m_hi = 0, full_range
    while m_lo < m_hi:
        mid = (m_lo + m_hi) // 2
        if feasible(mid):
            m_hi = mid
        else:
            m_lo = mid + 1
    best_m = m_lo

    # 阶段 2：二分最小可行 S = Σ|Δ_k|。
    s_lo, s_hi = 0, (n - 1) * best_m
    while s_lo < s_hi:
        mid = (s_lo + s_hi) // 2
        if feasible(best_m, s_hi=mid):
            s_hi = mid
        else:
            s_lo = mid + 1
    best_s = s_lo

    # 阶段 3：逐段钉死字典序最小值（x_0, x_1, ... 顺序；可行性关于阈值单调）。
    bounds = _x_diff_closure(n, base_x_lo, base_x_hi, best_m)
    assert bounds is not None
    x_lo, x_hi = bounds
    x_lo = x_lo + [0] * (n - 1)
    x_hi = x_hi + [best_m] * (n - 1)
    constraints = list(window_cons)
    constraints += _diff_constraints(n, best_m)
    constraints += _abs_sum_constraints(n, best_m, best_s, best_s)

    strains: list[int] = []
    for var in range(n):
        t_lo, t_hi = x_lo[var], x_hi[var]
        while t_lo < t_hi:
            mid = (t_lo + t_hi) // 2
            trial = constraints + [(((var, 1),), None, mid)]
            if _search(list(x_lo), list(x_hi), trial) is not None:
                t_hi = mid
            else:
                t_lo = mid + 1
        strains.append(t_lo)
        x_lo[var] = x_hi[var] = t_lo
        narrowed = _propagate(x_lo, x_hi, constraints)
        assert narrowed is not None  # 已选最优值必然可行
        x_lo, x_hi = narrowed

    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    return strains, diffs, best_m, best_s


def _shift_windows_for_temperature(
    windows: list[Window], weighted_coeff: list[int], t: int
) -> list[Window]:
    """窗 [lo,hi] 扣除热贡献 t·C_w 后得到机械加权和的闭区间。"""
    return [
        Window(
            start=win.start,
            end=win.end,
            lo=win.lo - t * weighted_coeff[idx],
            hi=win.hi - t * weighted_coeff[idx],
        )
        for idx, win in enumerate(windows)
    ]


def _solve_joint_thermal(
    lengths: list[int],
    strain_min: int,
    strain_max: int,
    windows: list[Window],
    thermal: ThermalCompensation,
    weighted_coeff: list[int],
) -> tuple[list[int], list[int], int, int, int]:
    """在全部允许温变上联合取得 (M, S, x, t) 字典序最优。

    温变集合是隐式析取：阈值判定 = 是否存在某个候选温变可行。候选温变始终
    按升序探测，末级平局（三级完全相同）自然落在较小温变上。
    无任何联合可行解释时抛 InfeasibleError。
    """
    n = len(lengths)
    temperatures = list(range(thermal.t_min, thermal.t_max + 1))
    full_range = strain_max - strain_min

    # 每个温变：窗界平移、机械窗约束、前缀闭包导出的机械域，只计算一次。
    shifted: dict[int, list[Window]] = {}
    window_cons_cache: dict[int, list[Constraint]] = {}
    base_bounds: dict[int, tuple[list[int], list[int]] | None] = {}
    diff_bounds_cache: dict[tuple[int, int], tuple[list[int], list[int]] | None] = {}

    for t in temperatures:
        shifted_t = _shift_windows_for_temperature(windows, weighted_coeff, t)
        shifted[t] = shifted_t
        window_cons_cache[t] = _window_constraints(n, lengths, shifted_t)
        closure = _prefix_closure(n, lengths, strain_min, strain_max, shifted_t)
        if closure is None:
            base_bounds[t] = None
        else:
            base_bounds[t] = _x_bounds_from_prefix(
                n, lengths, strain_min, strain_max, closure
            )

    def diff_bounds(t: int, m: int) -> tuple[list[int], list[int]] | None:
        key = (t, m)
        if key not in diff_bounds_cache:
            bounds0 = base_bounds[t]
            diff_bounds_cache[key] = (
                None if bounds0 is None else _x_diff_closure(n, bounds0[0], bounds0[1], m)
            )
        return diff_bounds_cache[key]

    def probe(
        candidates: list[int],
        m: int,
        s_hi: int | None = None,
        pins: list[tuple[int, int]] | None = None,
        cap: tuple[int, int] | None = None,
    ) -> int | None:
        """候选温变（按列表顺序）中是否存在可行者；返回首个可行温变。

        pins：已钉死的前段变量等式 (var, value)；cap：x_var <= value 的
        字典序钉压（与 pins 分开，保证可行性关于 cap 阈值单调）。
        """
        for t in candidates:
            narrowed = diff_bounds(t, m)
            if narrowed is None:
                continue
            x_lo, x_hi = list(narrowed[0]), list(narrowed[1])
            constraints = list(window_cons_cache[t])
            constraints += _diff_constraints(n, m)
            if s_hi is not None:
                x_lo += [0] * (n - 1)
                x_hi += [m] * (n - 1)
                constraints += _abs_sum_constraints(n, m, None, s_hi)
            if pins:
                for var, value in pins:
                    x_lo[var] = x_hi[var] = value
                    constraints.append((((var, 1),), value, value))
            if cap is not None:
                var, value = cap
                constraints.append((((var, 1),), None, value))
            if _search(x_lo, x_hi, constraints) is not None:
                return t
        return None

    # 阶段 0：任一温变在 M=全量程下可行？
    if probe(temperatures, full_range) is None:
        raise InfeasibleError(
            "no joint (mechanical strain, common temperature delta) assignment "
            "satisfies all observation windows within the submitted bounds"
        )

    # 阶段 1：联合二分最小 M（阈值可行 = 存在某个温变可行）。
    m_lo, m_hi = 0, full_range
    while m_lo < m_hi:
        mid = (m_lo + m_hi) // 2
        if probe(temperatures, mid) is not None:
            m_hi = mid
        else:
            m_lo = mid + 1
    best_m = m_lo

    # M* 下幸存的温变（升序）。
    candidates = [t for t in temperatures if probe([t], best_m) is not None]
    assert candidates

    # 阶段 2：联合二分最小 S = Σ|Δ_k|。
    s_lo, s_hi = 0, (n - 1) * best_m
    while s_lo < s_hi:
        mid = (s_lo + s_hi) // 2
        if probe(candidates, best_m, s_hi=mid) is not None:
            s_hi = mid
        else:
            s_lo = mid + 1
    best_s = s_lo
    candidates = [
        t for t in candidates if probe([t], best_m, s_hi=best_s) is not None
    ]
    assert candidates

    # 阶段 3：在幸存温变集合上联合钉死机械序列字典序最小值。
    # 每段探针区间取各温变 |Δ|<=M 域的并集；钉压可行阈值由二分钉死，
    # 随后裁掉与已选前缀不相容的温变。
    pinned: list[tuple[int, int]] = []
    strains: list[int] = []
    for var in range(n):
        union_lo = min(diff_bounds(t, best_m)[0][var] for t in candidates)
        union_hi = max(diff_bounds(t, best_m)[1][var] for t in candidates)
        t_lo, t_hi = union_lo, union_hi
        while t_lo < t_hi:
            mid = (t_lo + t_hi) // 2
            if (
                probe(candidates, best_m, s_hi=best_s, pins=pinned, cap=(var, mid))
                is not None
            ):
                t_hi = mid
            else:
                t_lo = mid + 1
        strains.append(t_lo)
        pinned.append((var, t_lo))
        candidates = [
            t
            for t in candidates
            if probe([t], best_m, s_hi=best_s, pins=pinned) is not None
        ]
        assert candidates

    # 整条机械序列钉死后，幸存温变中取最小者（三级完全相同的平局规则）。
    chosen_t = candidates[0]
    diffs = [strains[i] - strains[i - 1] for i in range(1, n)]
    return strains, diffs, best_m, best_s, chosen_t


def invert_payload(payload: object) -> dict:
    """完整反演，返回可直接复核的结果字典；校验失败/不可行抛对应异常。"""
    lengths, strain_min, strain_max, windows, thermal = validate(payload)

    if thermal is None:
        # 旧模式：请求/响应/裁决/失败语义完全不变。
        strains, diffs, best_m, best_s = _solve_mechanical(
            lengths, strain_min, strain_max, windows
        )
        return _build_result(lengths, windows, strains, diffs, best_m, best_s)

    # 热补偿：C_w = Σ_{i=s..e} L_i·c_i，窗内总加权和 = 机械和 + t·C_w。
    weighted_coeff = [
        sum(lengths[i] * thermal.coefficients[i] for i in range(win.start, win.end + 1))
        for win in windows
    ]
    strains, diffs, best_m, best_s, chosen_t = _solve_joint_thermal(
        lengths, strain_min, strain_max, windows, thermal, weighted_coeff
    )
    return _build_result(
        lengths,
        windows,
        strains,
        diffs,
        best_m,
        best_s,
        thermal=thermal,
        temperature_delta=chosen_t,
        weighted_coeff=weighted_coeff,
    )


def _build_result(
    lengths: list[int],
    windows: list[Window],
    strains: list[int],
    diffs: list[int],
    best_m: int,
    best_s: int,
    thermal: ThermalCompensation | None = None,
    temperature_delta: int | None = None,
    weighted_coeff: list[int] | None = None,
) -> dict:
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    window_checks = []
    for idx, win in enumerate(windows):
        weighted_sum = 0
        for i in range(win.start, win.end + 1):
            weighted_sum += lengths[i] * strains[i]
        total_length = prefix[win.end + 1] - prefix[win.start]
        check = {
            "index": idx,
            "start_segment": win.start + 1,
            "end_segment": win.end + 1,
            "total_length": total_length,
            "min_elongation": win.lo,
            "max_elongation": win.hi,
            "weighted_strain_sum": weighted_sum,
            "satisfied": win.lo <= weighted_sum <= win.hi,
        }
        if thermal is not None:
            assert temperature_delta is not None and weighted_coeff is not None
            # 启用热补偿时窗界已扣除热贡献；这里把热贡献加回，逐项分列以便复核：
            #   总回算和 = 机械贡献 + 热贡献（= t·Σ_{窗内} L_i·c_i）
            thermal_sum = temperature_delta * weighted_coeff[idx]
            observed_total = weighted_sum + thermal_sum
            check["mechanical_weighted_sum"] = weighted_sum
            check["weighted_coefficient_sum"] = weighted_coeff[idx]
            check["thermal_weighted_sum"] = thermal_sum
            check["weighted_strain_sum"] = observed_total
            check["satisfied"] = win.lo <= observed_total <= win.hi
        window_checks.append(check)
    # 两级平滑指标均由相邻差直接复核（重算以自证，不直接采用搜索内部值）。
    recomputed_m = max(abs(d) for d in diffs)
    recomputed_s = sum(abs(d) for d in diffs)
    assert recomputed_m == best_m and recomputed_s == best_s
    result = {
        "segment_count": len(lengths),
        "strains": strains,
        "adjacent_diffs": diffs,
        "objectives": {
            "max_adjacent_diff": recomputed_m,
            "sum_adjacent_abs_diff": recomputed_s,
        },
        "window_checks": window_checks,
        "criteria_order": [
            "max_adjacent_diff",
            "sum_adjacent_abs_diff",
            "lexicographic",
        ],
    }
    if thermal is not None:
        # strains/adjacent_diffs/两级指标均只描述机械应变；
        # 热贡献由所选共同温变与提交系数、段长即可独立复算。
        result["thermal_compensation"] = {
            "temperature_delta": temperature_delta,
            "coefficients": list(thermal.coefficients),
            "temperature_delta_bounds": {"min": thermal.t_min, "max": thermal.t_max},
        }
    return result
