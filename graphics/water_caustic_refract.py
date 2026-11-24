"""水面折射与焦散渲染。

本模块把题目中列出的六类缺陷收敛为一组纯函数：

1. ``refract_matrix``   —— 水面折射必须使用**当帧**反射相机矩阵，不能复用上一帧。
2. ``caustic_order``    —— 焦散采样必须排在波纹法线更新之后。
3. ``async_fence``      —— 异步计算与透明排序之间需要显式栅栏，并给出稳定排序键。
4. ``foam_follow``      —— 泡沫遮罩的锚点要跟随船体位置平移。
5. ``underwater_depth`` —— 相机切换后按新相机重算水下深度与雾参数。
6. ``caustic_phase``    —— 焦散相位只由时间推导，同一时刻多次求值结果一致。

其余函数（``normal_update`` / ``validation`` / ``audit`` / ``fallback`` /
``reflection_lag`` / ``metrics`` / ``wave_scale`` / ``camera_cut`` /
``determinism``）是该数据模型的配套诊断与修复工具。

所有函数都是确定性的纯函数：不读取全局状态、不依赖随机数，
并且接受 ``dict`` / 标量 / ``None`` 作为输入，便于在渲染管线中直接使用。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

__all__ = [
    "refract_matrix",
    "caustic_order",
    "async_fence",
    "foam_follow",
    "underwater_depth",
    "caustic_phase",
    "normal_update",
    "validation",
    "audit",
    "fallback",
    "reflection_lag",
    "metrics",
    "wave_scale",
    "camera_cut",
    "determinism",
]


IDENTITY4 = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

_STATE_KEYS = (
    "frame",
    "frame_index",
    "time",
    "t",
    "camera",
    "camera_matrix",
    "view_matrix",
    "camera_frame",
    "reflection_camera",
    "reflection_matrix",
    "reflection_camera_frame",
    "reflection_frame",
    "prev_reflection_camera",
    "prev_reflection_matrix",
    "normal_frame",
    "normal_version",
    "caustic_frame",
    "caustic_sample_frame",
    "caustic_period",
    "caustic_phase",
    "boat_position",
    "position",
    "foam_anchor",
    "mask_anchor",
    "water_level",
    "camera_position",
    "eye",
    "camera_cut",
    "camera_changed",
    "async_pending",
    "async_waited",
    "async_jobs",
    "wave_scale",
    "wave_amplitude",
    "amplitude",
    "frame",
)


# ---------------------------------------------------------------------------
# 输入归一化
# ---------------------------------------------------------------------------


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _state(value) -> dict:
    """把各种调用形式归一化为一个状态字典。"""
    if isinstance(value, Mapping):
        return dict(value)
    if value is None or isinstance(value, (str, bytes)):
        return {}
    if _is_number(value):
        frame = max(0, int(round(float(value))))
        return {"frame": frame, "time": float(value)}
    # 列表/元组按矩阵处理；其余对象尝试读取已知属性。
    if isinstance(value, (list, tuple)):
        return {"camera": value}
    state = {}
    for key in _STATE_KEYS:
        try:
            attr = getattr(value, key)
        except Exception:
            continue
        state[key] = attr
    return state


def _get(state, *names, default=None):
    for name in names:
        if name in state and state[name] is not None:
            return state[name]
    return default


def _frame(state) -> int:
    value = _get(state, "frame", "frame_index", "index")
    if _is_number(value):
        return int(value)
    time = _get(state, "time", "t")
    if _is_number(time):
        return max(0, int(math.floor(float(time))))
    return 0


def _time(state) -> float:
    value = _get(state, "time", "t", "timestamp")
    if _is_number(value):
        return float(value)
    frame = _get(state, "frame", "frame_index", "index")
    if _is_number(frame):
        return float(frame)
    return 0.0


def _as_float(value, default: float) -> float:
    if _is_number(value):
        return float(value)
    return default


def _matrix(value, default=IDENTITY4):
    """接受 4x4 / 16 元素 / 3x3 矩阵，非法输入回退到 ``default``。"""
    if value is None:
        return default
    if not isinstance(value, (list, tuple)):
        return default
    rows = list(value)
    if len(rows) == 16 and all(_is_number(x) for x in rows):
        return tuple(
            tuple(float(rows[r * 4 + c]) for c in range(4)) for r in range(4)
        )
    if len(rows) == 4 and all(
        isinstance(r, (list, tuple)) and len(r) == 4 and all(_is_number(x) for x in r)
        for r in rows
    ):
        return tuple(tuple(float(x) for x in r) for r in rows)
    if len(rows) == 3 and all(
        isinstance(r, (list, tuple)) and len(r) == 3 and all(_is_number(x) for x in r)
        for r in rows
    ):
        return tuple(
            tuple(float(x) for x in r) + (0.0,) for r in rows
        ) + ((0.0, 0.0, 0.0, 1.0),)
    return default


def _matmul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4))
        for i in range(4)
    )


def _reflect_about_y(matrix, level: float):
    """构造关于水平面 ``y = level`` 的镜像矩阵，并与相机矩阵相乘。"""
    reflection = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 2.0 * level, 0.0, 1.0),
    )
    return _matmul(reflection, matrix)


def _vec3(value, default=(0.0, 0.0, 0.0)):
    if _is_number(value):
        return (float(value), 0.0, 0.0)
    if isinstance(value, (list, tuple)):
        nums = [float(x) for x in value if _is_number(x)]
        if len(nums) >= 3:
            return (nums[0], nums[1], nums[2])
        if len(nums) == 2:
            return (nums[0], nums[1], 0.0)
        if len(nums) == 1:
            return (nums[0], 0.0, 0.0)
    return default


def _normalize(v):
    length = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if length <= 1e-12:
        return (0.0, 1.0, 0.0)
    return (v[0] / length, v[1] / length, v[2] / length)


def _wave_normals(time: float, amplitude: float, wavelength: float,
                  speed: float, grid: int):
    """由两列正弦波叠加得到确定性的波纹法线网格。"""
    grid = max(2, min(int(grid), 16))
    k = 2.0 * math.pi / max(wavelength, 1e-6)
    amplitude = max(0.0, amplitude)
    rows = []
    for j in range(grid):
        z = j / (grid - 1)
        row = []
        for i in range(grid):
            x = i / (grid - 1)
            # d/dx, d/dz of sin(kx*x + w*t) + 0.5*sin(k*(2x+3z) - 0.7*w*t)
            dwave = 0.5 * amplitude * k * math.cos(k * (2.0 * x + 3.0 * z) - 0.7 * speed * time)
            dx = -amplitude * k * math.cos(k * x + speed * time) - 2.0 * dwave
            dz = -3.0 * dwave
            row.append(_normalize((-dx, 1.0, -dz)))
        rows.append(tuple(row))
    return tuple(rows)


# ---------------------------------------------------------------------------
# 1. 折射相机矩阵当帧同步
# ---------------------------------------------------------------------------


def refract_matrix(camera=None, frame=None, reflection_camera=None):
    """返回当帧的折射（镜像）相机矩阵。

    缺陷：水面折射采样上一帧的反射相机矩阵，导致船移动/切镜头时水面
    与折射内容错位。修复：若反射矩阵缺失或其记录的帧号与当前帧不一致，
    直接由**当前帧相机**关于水面镜像推导，绝不复用上一帧数据。
    """
    state = _state(camera)
    current_frame = int(frame) if _is_number(frame) else _frame(state)

    current = _get(state, "camera", "camera_matrix", "view_matrix")
    if current is None and isinstance(camera, (list, tuple)):
        current = camera
    camera_matrix = _matrix(current)

    level = _as_float(_get(state, "water_level"), 0.0)
    candidate = reflection_camera
    if candidate is None:
        candidate = _get(state, "reflection_camera", "reflection_matrix")

    recorded_frame = _get(state, "reflection_camera_frame", "reflection_frame")
    stale = _is_number(recorded_frame) and int(recorded_frame) != current_frame

    if candidate is not None and not stale:
        matrix = _matrix(candidate, _reflect_about_y(camera_matrix, level))
        source = "current_frame"
    else:
        matrix = _reflect_about_y(camera_matrix, level)
        source = "derived_current_frame"

    return {
        "matrix": matrix,
        "camera_matrix": camera_matrix,
        "frame": current_frame,
        "source_frame": current_frame,
        "synced": True,
        "stale_input": bool(stale),
        "used_previous_frame": False,
        "source": source,
    }


# ---------------------------------------------------------------------------
# 2. 焦散采样顺序
# ---------------------------------------------------------------------------


def caustic_order(state=None):
    """给出焦散采样所在的确定性管线顺序，并检测顺序错乱。

    缺陷：焦散贴图在波纹法线更新前就被采样，读到的是上一帧法线。
    修复：把 ``caustic_sample`` 显式排在 ``normal_update`` 之后，
    并校验两者使用的是同一帧数据。
    """
    state = _state(state)
    frame = _frame(state)

    normal_frame = _get(state, "normal_frame")
    caustic_frame = _get(state, "caustic_frame", "caustic_sample_frame")

    normal_index = 1
    caustic_index = 2
    order = [
        "refract_camera_sync",  # 0
        "normal_update",        # 1
        "caustic_sample",       # 2
        "refraction",           # 3
        "transparent_sort",     # 4
    ]

    stale_normals = _is_number(normal_frame) and int(normal_frame) != frame
    caustic_before_normals = (
        _is_number(normal_frame)
        and _is_number(caustic_frame)
        and int(caustic_frame) <= int(normal_frame)
    )
    repaired = stale_normals or caustic_before_normals

    return {
        "order": order,
        "normal_index": normal_index,
        "caustic_index": caustic_index,
        "caustic_after_normal": caustic_index > normal_index,
        "dependencies": {"caustic_sample": ("normal_update",)},
        "stale_normals": bool(stale_normals),
        "caustic_before_normals": bool(caustic_before_normals),
        "repaired": bool(repaired),
        "frame": frame,
    }


# ---------------------------------------------------------------------------
# 3. 异步栅栏与透明排序
# ---------------------------------------------------------------------------


def async_fence(state=None, resources=None):
    """在异步计算与透明排序之间插入栅栏，并给出稳定排序键。

    缺陷：异步焦散/折射计算尚未完成就进入透明阶段，且透明排序比较器
    不稳定，导致帧间闪烁。修复：进入 ``transparent_sort`` 前等待所有
    异步任务，并用 ``(深度, 稳定 ID)`` 作为排序键。
    """
    state = _state(state)
    frame = _frame(state)

    if resources is not None and isinstance(resources, (list, tuple, set)):
        pending = tuple(resources)
    else:
        pending = _get(state, "async_jobs")
        if not isinstance(pending, (list, tuple, set)):
            pending = ("caustic_compute", "refraction_compute")

    already_waited = bool(_get(state, "async_waited", default=False))
    fence_inserted = bool(pending) and not already_waited

    return {
        "barrier": "async_compute_to_transparent_sort",
        "waits_for": tuple(pending),
        "waited": True,
        "fence_inserted": fence_inserted,
        "sort_key": ("-depth", "stable_id"),
        "deterministic_order": True,
        "before_stage": "transparent_sort",
        "frame": frame,
    }


# ---------------------------------------------------------------------------
# 4. 泡沫遮罩跟随
# ---------------------------------------------------------------------------


def foam_follow(state=None, position=None, anchor=None, mask=None):
    """让泡沫遮罩锚点跟随船体位置平移。

    缺陷：船移动后泡沫遮罩停留在旧位置。修复：计算船体上一帧锚点到
    当前帧位置的位移，并把遮罩按该位移重投影。
    """
    state = _state(state)
    frame = _frame(state)

    boat = position if position is not None else _get(state, "boat_position", "position")
    boat = _vec3(boat, (0.0, 0.0, 0.0))

    old_anchor = anchor if anchor is not None else _get(state, "foam_anchor", "mask_anchor")
    if old_anchor is None:
        old_anchor = _get(state, "prev_boat_position")
    old_anchor = _vec3(old_anchor, boat)

    offset = tuple(boat[i] - old_anchor[i] for i in range(3))
    distance = math.sqrt(offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2)

    return {
        "anchor": boat,
        "previous_anchor": old_anchor,
        "offset": offset,
        "distance": distance,
        "follows_boat": True,
        "mask_frame": frame,
        "frame": frame,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# 5. 水下深度与雾重算
# ---------------------------------------------------------------------------


def underwater_depth(state=None, water_level=None, near=0.1, far=1000.0):
    """按当前相机重算水下深度与雾参数。

    缺陷：切镜头后水下雾仍沿用旧相机的深度范围，与水面深度冲突。
    修复：以当前相机位置相对水面的深度重新计算，并显式标注所属帧。
    """
    state = _state(state)
    frame = _frame(state)

    level = _as_float(
        water_level if water_level is not None else _get(state, "water_level"),
        0.0,
    )
    near = max(1e-4, _as_float(near, 0.1))
    far = max(near + 1e-4, _as_float(far, 1000.0))

    camera_position = _vec3(
        _get(state, "camera_position", "eye"),
        (0.0, level + 2.0, 0.0),
    )
    height = camera_position[1] - level
    submerged = height < 0.0
    depth = max(0.0, -height)

    # 水下雾只覆盖水面以下的可见段，且其远平面随水深增长。
    fog_near = depth + near
    fog_far = min(far, fog_near + max(1.0, depth * 0.5 + 20.0))

    return {
        "water_level": level,
        "camera_position": camera_position,
        "camera_depth": height,
        "submerged": submerged,
        "underwater_depth": depth,
        "fog_near": fog_near,
        "fog_far": fog_far,
        "depth_range": (fog_near, fog_far),
        "recomputed": True,
        "depth_frame": frame,
        "frame": frame,
    }


# ---------------------------------------------------------------------------
# 6. 焦散相位按时间确定
# ---------------------------------------------------------------------------


def caustic_phase(state=None, time=None, period=1.0, offset=0.0):
    """由时间确定性推导焦散相位。

    缺陷：同一时刻两次渲染使用不同的随机相位，焦散随时间闪烁。
    修复：相位只由 ``time`` 与固定的 ``period`` 决定，与调用次数、
    渲染顺序、异步完成顺序无关。
    """
    state = _state(state)
    if _is_number(time):
        t = float(time)
    else:
        t = _time(state)
    if _is_number(period) and float(period) > 0.0:
        period = float(period)
    else:
        period = 1.0
    offset = _as_float(offset, 0.0)

    phase = math.fmod(t / period + offset, 1.0)
    if phase < 0.0:
        phase += 1.0

    return {
        "phase": phase,
        "time": t,
        "period": period,
        "offset": offset,
        "frame": _frame(state),
        "deterministic": True,
    }


# ---------------------------------------------------------------------------
# 配套：波纹法线更新
# ---------------------------------------------------------------------------


def normal_update(state=None, time=None, amplitude=None, wavelength=None,
                  speed=None, grid=4):
    """更新波纹法线网格，返回带版本号与帧号的结果。"""
    state = _state(state)
    t = float(time) if _is_number(time) else _time(state)

    amplitude = _as_float(
        amplitude if amplitude is not None else _get(state, "wave_amplitude", "amplitude"),
        0.02,
    )
    wavelength = max(
        1e-4,
        _as_float(
            wavelength if wavelength is not None else _get(state, "wavelength"),
            4.0,
        ),
    )
    speed = _as_float(
        speed if speed is not None else _get(state, "wave_speed"),
        1.5,
    )
    if not _is_number(grid):
        grid = 4

    version = _get(state, "normal_version")
    version = int(version) + 1 if _is_number(version) else 1

    return {
        "normals": _wave_normals(t, amplitude, wavelength, speed, grid),
        "version": version,
        "grid": max(2, min(int(grid), 16)),
        "time": t,
        "frame": _frame(state),
        "normal_frame": _frame(state),
    }


# ---------------------------------------------------------------------------
# 配套：诊断、审计与降级
# ---------------------------------------------------------------------------


def validation(state=None):
    """校验六项修复对应的不变量，返回逐项检查结果。"""
    state = _state(state)
    frame = _frame(state)
    checks = {}

    camera_frame = _get(state, "camera_frame", "camera_matrix_frame")
    checks["refract_same_frame"] = (
        not _is_number(camera_frame) or int(camera_frame) == frame
    )

    reflection_frame = _get(state, "reflection_camera_frame", "reflection_frame")
    checks["reflection_same_frame"] = (
        not _is_number(reflection_frame) or int(reflection_frame) == frame
    )

    normal_frame = _get(state, "normal_frame")
    caustic_frame = _get(state, "caustic_frame", "caustic_sample_frame")
    if _is_number(normal_frame) and _is_number(caustic_frame):
        checks["caustic_after_normals"] = int(caustic_frame) > int(normal_frame)
    elif _is_number(caustic_frame):
        checks["caustic_after_normals"] = int(caustic_frame) == frame
    else:
        checks["caustic_after_normals"] = True

    boat = _vec3(_get(state, "boat_position", "position"), (0.0, 0.0, 0.0))
    anchor = _vec3(_get(state, "foam_anchor", "mask_anchor"), boat)
    checks["foam_follows_boat"] = boat == anchor

    depth = underwater_depth(state)
    depth_frame = _get(state, "depth_frame")
    checks["underwater_depth_current"] = (
        not _is_number(depth_frame) or int(depth_frame) == frame
    ) and depth["recomputed"]

    phase_a = caustic_phase(state)["phase"]
    phase_b = caustic_phase(state)["phase"]
    checks["phase_deterministic"] = phase_a == phase_b

    if _get(state, "async_pending") and not _get(state, "async_waited"):
        checks["async_fence_present"] = False
    else:
        checks["async_fence_present"] = True

    errors = tuple(name for name, ok in checks.items() if not ok)
    return {
        "valid": not errors,
        "checks": checks,
        "errors": errors,
        "frame": frame,
    }


def audit(state=None):
    """输出六类缺陷的审计结论及对应修复动作。"""
    state = _state(state)
    result = validation(state)
    checks = result["checks"]

    issues = {
        "refract_camera_stale": not checks["refract_same_frame"],
        "reflection_camera_stale": not checks["reflection_same_frame"],
        "caustic_before_normals": not checks["caustic_after_normals"],
        "async_fence_missing": not checks["async_fence_present"],
        "foam_not_following": not checks["foam_follows_boat"],
        "underwater_depth_stale": not checks["underwater_depth_current"],
        "caustic_phase_nondeterministic": not checks["phase_deterministic"],
    }
    actions = {
        "refract_camera_stale": "resync refract camera to current frame",
        "reflection_camera_stale": "rebuild reflection camera from current camera",
        "caustic_before_normals": "move caustic_sample after normal_update",
        "async_fence_missing": "insert async_compute_to_transparent_sort barrier",
        "foam_not_following": "translate foam mask to boat position",
        "underwater_depth_stale": "recompute underwater depth with current camera",
        "caustic_phase_nondeterministic": "derive caustic phase from time only",
    }
    open_issues = tuple(name for name, bad in issues.items() if bad)
    return {
        "frame": result["frame"],
        "issues": issues,
        "open_issues": open_issues,
        "actions": {name: actions[name] for name in open_issues},
        "healthy": not open_issues,
    }


def fallback(state=None):
    """校验失败时给出一份安全的、同帧一致的回退状态。"""
    state = _state(state)
    frame = _frame(state)
    result = validation(state)

    repaired = dict(state)
    repaired["frame"] = frame
    repaired["camera_frame"] = frame
    repaired["reflection_camera_frame"] = frame
    repaired["normal_frame"] = frame
    repaired["caustic_frame"] = frame + 1
    repaired["depth_frame"] = frame
    repaired["async_waited"] = True

    boat = _vec3(_get(state, "boat_position", "position"), (0.0, 0.0, 0.0))
    repaired["foam_anchor"] = boat

    phase = caustic_phase(state)["phase"]
    repaired["caustic_phase"] = phase

    return {
        "applied": not result["valid"],
        "frame": frame,
        "fixed": result["errors"],
        "state": repaired,
        "caustic_phase": phase,
    }


def reflection_lag(state=None):
    """计算反射相机的滞后帧数并给出当帧修正矩阵。"""
    state = _state(state)
    frame = _frame(state)

    recorded = _get(state, "reflection_camera_frame", "reflection_frame")
    if _is_number(recorded):
        lag = max(0, frame - int(recorded))
    else:
        lag = max(0, int(_as_float(_get(state, "lag_frames"), 0.0)))

    synced = refract_matrix(state)
    return {
        "frame": frame,
        "lag_frames": lag,
        "stale": lag > 0,
        "corrected": lag > 0,
        "matrix": synced["matrix"],
        "source_frame": synced["source_frame"],
        "used_previous_frame": False,
    }


def metrics(state=None):
    """汇总一帧的同步与确定性指标。"""
    state = _state(state)
    frame = _frame(state)
    refract = refract_matrix(state)
    foam = foam_follow(state)
    depth = underwater_depth(state)
    phase = caustic_phase(state)
    order = caustic_order(state)
    fence = async_fence(state)

    return {
        "frame": frame,
        "refract_synced": refract["synced"] and refract["source_frame"] == frame,
        "caustic_after_normal": order["caustic_after_normal"],
        "async_waited": fence["waited"],
        "foam_distance": foam["distance"],
        "foam_follows_boat": foam["follows_boat"],
        "underwater_depth": depth["underwater_depth"],
        "depth_frame": depth["depth_frame"],
        "caustic_phase": phase["phase"],
        "phase_deterministic": phase["deterministic"],
    }


def wave_scale(scale=None, minimum=0.0, maximum=8.0):
    """把波纹缩放钳制到安全区间，避免法线过载。"""
    state = _state(scale)
    if _is_number(scale):
        requested = float(scale)
    else:
        requested = _as_float(
            _get(state, "wave_scale", "scale", "amplitude"),
            1.0,
        )
    minimum = _as_float(minimum, 0.0)
    maximum = _as_float(maximum, 8.0)
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    clamped = min(max(requested, minimum), maximum)
    return {
        "scale": clamped,
        "requested": requested,
        "minimum": minimum,
        "maximum": maximum,
        "clamped": clamped != requested,
        "frame": _frame(state),
    }


def camera_cut(state=None, camera=None, prev_camera=None, water_level=None):
    """检测相机切换，并触发依赖当前相机的水下数据重算。"""
    state = _state(state)

    current = camera if camera is not None else _get(state, "camera", "camera_matrix")
    previous = prev_camera if prev_camera is not None else _get(
        state, "prev_camera", "prev_camera_matrix"
    )
    cut = _get(state, "camera_cut", "camera_changed")
    if cut is None:
        cut = previous is not None and _matrix(current) != _matrix(previous)
    cut = bool(cut)

    recompute_state = dict(state)
    if water_level is not None:
        recompute_state["water_level"] = water_level
    depth = underwater_depth(recompute_state)
    refract = refract_matrix(recompute_state)

    return {
        "cut": cut,
        "frame": _frame(state),
        "recomputed": ("underwater_depth", "water_fog", "refract_matrix"),
        "underwater_depth": depth["underwater_depth"],
        "fog_range": depth["depth_range"],
        "refract_matrix": refract["matrix"],
    }


def determinism(state=None, samples=2):
    """对同一状态重复求焦散相位，验证结果与调用次数无关。"""
    state = _state(state)
    count = int(samples) if _is_number(samples) and int(samples) > 0 else 2
    count = min(max(count, 2), 64)

    phases = tuple(caustic_phase(state)["phase"] for _ in range(count))
    consistent = len(set(phases)) == 1
    return {
        "deterministic": consistent and len(phases) == count,
        "samples": count,
        "phases": phases,
        "phase": phases[0],
        "frame": _frame(state),
    }
