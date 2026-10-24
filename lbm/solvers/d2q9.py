"""D2Q9 求解器。

streaming 用 pull 形式，逐方向切片写入目标 buffer，不使用 ``torch.roll``：

    f_new_i(x) = f_post_i(x - c_i)          源点是流体
    f_new_i(x) = f_post_opp_i(x)            源点是固体（半程反弹）

两种边界模式：

``boundary="periodic"``（默认，也是所有验证算例用的）
    x、y 都周期。壁面靠域内固体行实现。

``boundary="inlet_outlet"``
    x 不再周期：左边界 Zou/He 速度入口（给定 ``u_inf``），右边界
    Zou/He 密度出口（给定 ``rho_inf``）。y 仍按周期处理。

两种模式下固体都用半程反弹。
"""

from __future__ import annotations

import warnings

import torch

from ..collide.bgk import BGK
from ..collide.mrt import MRT
from ..collide.trt import TRT
from ..stencils.d2q9 import D2Q9
from .base import BaseSolver

# magic parameter 的常用取值，壁面位置无关性最好
DEFAULT_LAMBDA = 3.0 / 16.0


def axis_slices(shift: int, n: int) -> list[tuple[slice, slice]]:
    """把"沿某个轴移动 shift 格并周期回绕"拆成若干 (目标切片, 源切片)。"""
    if shift == 0:
        return [(slice(0, n), slice(0, n))]
    if shift > 0:
        return [
            (slice(shift, n), slice(0, n - shift)),
            (slice(0, shift), slice(n - shift, n)),
        ]
    s = -shift
    return [
        (slice(0, n - s), slice(s, n)),
        (slice(n - s, n), slice(0, s)),
    ]


def open_slices(shift: int, n: int) -> list[tuple[slice, slice]]:
    """非周期方向的移位：够不到的地方不写，留给边界条件填。"""
    if shift == 0:
        return [(slice(0, n), slice(0, n))]
    if shift > 0:
        return [(slice(shift, n), slice(0, n - shift))]
    s = -shift
    return [(slice(0, n - s), slice(s, n))]


def write_shifted(
    dst: torch.Tensor,
    src: torch.Tensor,
    sx: int,
    sy: int,
    wrap_x: bool = True,
) -> None:
    """``dst[x, y] = src[x - sx, y - sy]``。

    y 方向永远周期回绕；x 方向由 ``wrap_x`` 决定。``wrap_x=False`` 时
    目标里"没被写到"的那一列保持原值，随后由 Zou/He 覆盖——这正是
    入口/出口要的语义。
    """
    nx, ny = src.shape[-2], src.shape[-1]
    x_pairs = axis_slices(sx, nx) if wrap_x else open_slices(sx, nx)
    for dx, sxx in x_pairs:
        for dy, syy in axis_slices(sy, ny):
            dst[dx, dy] = src[sxx, syy]


class D2Q9Solver(BaseSolver):
    spec = D2Q9
    model = "D2Q9"

    def __init__(
        self,
        shape,
        solid=None,
        *,
        boundary: str = "periodic",
        u_inf: float = 0.0,
        rho_inf: float = 1.0,
        s_e: float = 1.9,
        **kwargs,
    ) -> None:
        if boundary not in ("periodic", "inlet_outlet"):
            raise ValueError(
                f"boundary 只支持 'periodic' / 'inlet_outlet'，收到 {boundary!r}"
            )
        self._boundary = boundary
        self._u_inf = float(u_inf)
        self._rho_inf = float(rho_inf)
        self._s_e = float(s_e)
        if boundary == "inlet_outlet":
            if not (-1.0 < self._u_inf < 1.0):
                raise ValueError(f"u_inf 必须在 (-1,1) 内，收到 {u_inf}")
            if self._rho_inf <= 0.0:
                raise ValueError(f"rho_inf 必须为正，收到 {rho_inf}")
        # inlet/outlet 模式下 x 不周期，streaming 不能回绕
        self._wrap_x = boundary == "periodic"
        super().__init__(shape, solid, **kwargs)

    def _setup(self) -> None:
        if not self._wrap_x:
            # Zou/He 只作用在整列流体上；边界列一旦有固体就冲突了
            if bool(self.solid[0, :].any()) or bool(self.solid[-1, :].any()):
                raise ValueError("inlet_outlet 模式要求 x=0 与 x=Nx-1 两列全为流体")

        # 每个方向预计算一张"目标点的源点是不是固体"的 mask。
        # mask[x] = solid[x - c_i]，x 方向的回绕方式必须和 streaming 一致。
        self._wall_mask: list[torch.Tensor] = []
        self._has_wall: list[bool] = []

        for i in range(self.spec.q):
            cx, cy = self.spec.c[i]
            m = torch.zeros_like(self.solid)
            write_shifted(m, self.solid, cx, cy, wrap_x=self._wrap_x)
            self._wall_mask.append(m)
            # i=0 时反射项和正常项完全一样，跳过
            self._has_wall.append(i != 0 and bool(m.any().item()))

        self.collision = self._make_collision()

    def _make_collision(self):
        match self.collision_kind:
            case "bgk":
                if self.Lambda is not None:
                    warnings.warn(
                        "collision='bgk' 时 Lambda 无意义，已忽略",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    self.Lambda = None
                self.collision_info = {"collision": "bgk"}
                return BGK(self.tau, self._st.w, self._st.c)

            case "trt":
                # Λ 由外部指定；不指定时用 3/16（magic parameter 的常用取值）
                lam = DEFAULT_LAMBDA if self.Lambda is None else self.Lambda
                self.Lambda = lam
                op = TRT.from_lambda(
                    self.tau, lam, self._st.w, self._st.c, self._st.opp
                )
                self.collision_info = {
                    "collision": "trt",
                    "Lambda": lam,
                    "tau_minus": op.tau_minus,
                }
                return op

            case "mrt":
                # Λ 由外部指定；不指定时用 3/16（magic parameter 的常用取值）
                lam = DEFAULT_LAMBDA if self.Lambda is None else self.Lambda
                self.Lambda = lam
                op = MRT(
                    self.nu,
                    lam,
                    self._st.w,
                    self._st.c,
                    self.spec.c,
                    s_e=self._s_e,
                )
                self.collision_info = {
                    "collision": "mrt",
                    "Lambda": lam,
                    "s_nu": op.s_nu,
                    "s_e": op.s_e,
                    "s_q": op.s_q,
                }
                return op

            case _:
                raise RuntimeError(f"未知碰撞模型: {self.collision_kind!r}")

    def _stream(self, f_post: torch.Tensor, dst: torch.Tensor) -> None:
        dst[0].copy_(f_post[0])
        for i in range(1, self.spec.q):
            cx, cy = self.spec.c[i]
            write_shifted(dst[i], f_post[i], cx, cy, wrap_x=self._wrap_x)
            if self._has_wall[i]:
                # 注意：这里必须先算出一个临时张量再拷回。
                # torch.where(..., out=dst[i]) 在 MPS 上「输出与输入别名」时
                # 会算出错误结果（CPU 上正确），别为了省一次分配改成 out=。
                dst[i].copy_(
                    torch.where(
                        self._wall_mask[i],
                        f_post[self.spec.opp[i]],
                        dst[i],
                    )
                )

        # 固体节点上的值没有意义，清掉以免后续求和时出现奇怪量级。
        if self._has_solid:
            dst.masked_fill_(self.solid, 0.0)

    # ------------------------------------------------------------------
    # Zou/He 入口出口
    # ------------------------------------------------------------------
    def _apply_open_boundary(self, dst: torch.Tensor) -> None:
        """左边界速度入口 + 右边界密度出口。

        必须在 streaming 之后调用：此时 cx<=0 的分量已经正确地从域内流进来，
        cx>0 的分量因为 wrap_x=False 而空缺，正好由 Zou/He 补上。

        D2Q9 方向编号（本项目）::

            i : (cx, cy)     opp
            1 : ( 1,  0) E     3 W
            2 : ( 0, -1) S     4 N
            5 : ( 1,  1) NE    7 SW
            6 : ( 1, -1) SE    8 NW

        西边界未知量是 1/5/6（cx>0），东边界未知量是 3/7/8（cx<0）。
        """
        if self._boundary != "inlet_outlet":
            return

        u_in = self._u_inf
        rho_in = self._rho_inf

        # ---- 西边界：速度入口 u = (u_in, 0) ----
        w = dst[:, 0, :]
        f0, f2, f3, f4 = w[0], w[2], w[3], w[4]
        f7, f8 = w[7], w[8]

        rho_w = (f0 + f2 + f4 + 2.0 * (f3 + f7 + f8)) / (1.0 - u_in)
        # 先算完所有右端项再写回，w[1]/w[5]/w[6] 不参与任何右端项
        w[1] = f3 + (2.0 / 3.0) * rho_w * u_in
        w[5] = f7 + (1.0 / 6.0) * rho_w * u_in + 0.5 * (f2 - f4)
        w[6] = f8 + (1.0 / 6.0) * rho_w * u_in - 0.5 * (f2 - f4)

        # ---- 东边界：密度出口 rho = rho_in ----
        e = dst[:, -1, :]
        g0, g1, g2, g4 = e[0], e[1], e[2], e[4]
        g5, g6 = e[5], e[6]

        u_e = -1.0 + (g0 + g2 + g4 + 2.0 * (g1 + g5 + g6)) / rho_in
        e[3] = g1 - (2.0 / 3.0) * rho_in * u_e
        e[7] = g5 - (1.0 / 6.0) * rho_in * u_e - 0.5 * (g2 - g4)
        e[8] = g6 - (1.0 / 6.0) * rho_in * u_e + 0.5 * (g2 - g4)

    def _extra_meta(self) -> dict:
        meta = {"boundary": self._boundary}
        meta.update(getattr(self, "collision_info", {}))
        if self._boundary == "inlet_outlet":
            meta["u_inf"] = self._u_inf
            meta["rho_inf"] = self._rho_inf
        return meta
