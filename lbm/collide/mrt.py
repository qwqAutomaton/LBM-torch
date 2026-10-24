"""MRT（Multi-Relaxation-Time）碰撞 + Guo forcing。

矩按 Lallemand & Luo 的 D2Q9 顺序排列：

    rho, e, epsilon, jx, qx, jy, qy, pxx, pxy

LL 原矩阵的列顺序是
``(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)``，
而本项目的方向表（见 ``docs/physics.md``）是
``(0,0), (1,0), (0,-1), (-1,0), (0,1), (1,1), (1,-1), (-1,-1), (-1,1)``。
``ll_matrix()`` 按速度向量重排列，行顺序保持不变。

松弛矩阵是对角的：

    S = diag(0, s_e, s_eps, 0, s_q, 0, s_q, s_nu, s_nu)

    s_nu = 1/tau = 1/(cs^2*nu + 0.5)
    s_q  = 1/(0.5 + Lambda/(tau - 0.5))      # magic parameter
    s_e = s_eps = 1.9                         # Lallemand-Luo 的稳定取值

**能量模式不能跟着魔法参数走。** 如果把 ``s_e`` / ``s_eps`` 也设成 ``s_q``，
在 ``tau = 0.53`` 时 ``s_q ≈ 0.148``（``tau_q ≈ 6.75``），声学/能量模式几乎
不衰减；带 Zou/He 入口出口的绕流会从出口开始指数增长，几千步内 rho 到 inf
（见 ``docs/validation.md``）。``s_q`` 只负责壁面位置，``s_e`` / ``s_eps``
取 1.9 把能量模式压住。TRT 正好是 ``s_e = s_eps = s_nu``、``s_q = s_minus``
的特例，所以这套取法在低粘度下和 TRT 一样稳。

密度和两个动量分量是守恒矩（松弛率 0），``pxx`` / ``pxy`` 用 ``s_nu``
决定粘度。逆矩阵在构造时算好并缓存（先用 float64 求逆再转 compute dtype），
每步只多两次 9x9 的 ``tensordot``。

力项在矩空间施加 ``(I - S/2) M F*``，其中 ``F*_i = w_i [3(c_i-u) +
9(c_i·u)c_i] · F`` 是不带半格点因子的 Guo 源项。动量行的松弛率为 0，
所以每步动量严格增加 ``F``，宏观速度满足 ``u = (n + 1/2) F``，与
BGK / TRT 的约定一致。

最后把 ``f_post`` 的密度和动量投影回目标值（``w_i [Δρ + 3 c_i·Δj]``）。
``M^{-1} M`` 在 float32 下不是精确恒等，不做这一步的话守恒量会每步漂移，
长时间算例里表现为 ``uy`` 线性增长；精确算术下该修正恒为 0。
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..equilibrium import equilibrium_with_cu
from .base import CollisionOperator

# 矩的名字，顺序就是变换矩阵的行
MOMENTS = ("rho", "e", "epsilon", "jx", "qx", "jy", "qy", "pxx", "pxy")

# Lallemand & Luo 的 D2Q9 矩矩阵，列顺序见 _LL_VELOCITIES
LL_M = (
    (1, 1, 1, 1, 1, 1, 1, 1, 1),
    (-4, -1, -1, -1, -1, 2, 2, 2, 2),
    (4, -2, -2, -2, -2, 1, 1, 1, 1),
    (0, 1, 0, -1, 0, 1, -1, -1, 1),
    (0, -2, 0, 2, 0, 1, -1, -1, 1),
    (0, 0, 1, 0, -1, 1, 1, -1, -1),
    (0, 0, -2, 0, 2, 1, 1, -1, -1),
    (0, 1, -1, 1, -1, 0, 0, 0, 0),
    (0, 0, 0, 0, 0, 1, -1, 1, -1),
)

# LL_M 的列对应的速度，顺序与 LL 原文一致
_LL_VELOCITIES = (
    (0, 0),
    (1, 0),
    (0, 1),
    (-1, 0),
    (0, -1),
    (1, 1),
    (-1, 1),
    (-1, -1),
    (1, -1),
)


def ll_matrix(velocities: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """按给定速度顺序对 LL 矩阵做列置换，返回本项目的 9x9 矩矩阵。"""
    if len(velocities) != len(_LL_VELOCITIES):
        raise ValueError(
            f"MRT 只支持 D2Q9 的 {len(_LL_VELOCITIES)} 个速度，"
            f"收到 {len(velocities)} 个"
        )
    index = {v: i for i, v in enumerate(_LL_VELOCITIES)}
    perm = []
    for v in velocities:
        key = tuple(int(x) for x in v)
        if key not in index:
            raise ValueError(f"MRT 不支持速度 {key!r}，只支持 D2Q9 的九方向集")
        perm.append(index[key])
    if len(set(perm)) != len(perm):
        raise ValueError("MRT 的速度集有重复，无法与 LL 矩阵对应")
    return tuple(tuple(row[j] for j in perm) for row in LL_M)


class MRT(CollisionOperator):
    def __init__(
        self,
        nu: float,
        Lambda: float,
        w: torch.Tensor,
        c: Sequence[torch.Tensor],
        velocities: Sequence[Sequence[int]],
        s_e: float = 1.9,
    ) -> None:
        if nu <= 0.0:
            raise ValueError(f"nu 必须为正，当前 {nu}")
        if Lambda <= 0.0:
            raise ValueError(f"Lambda 必须为正，当前 {Lambda}")

        self.nu = float(nu)
        self.tau = 3.0 * self.nu + 0.5
        self.s_nu = 1.0 / self.tau
        self.Lambda = float(Lambda)
        # 魔法参数只约束 ghost（能量通量）模式，决定半程反弹的壁面位置
        self.s_q = 1.0 / (0.5 + self.Lambda / (self.tau - 0.5))
        # 能量模式必须快速松弛，否则低粘度 + Zou/He 出口会指数发散
        self.s_e = s_e
        if not 0.0 < self.s_q < 2.0:
            raise ValueError(
                "Lambda/(tau - 0.5) 给出的 s_q 必须落在 (0,2)，"
                f"当前 s_nu={self.s_nu:.6g}, s_q={self.s_q:.6g}"
            )

        self.w = w
        self.c = tuple(c)

        matrix = torch.tensor(ll_matrix(velocities), dtype=torch.float64)
        inverse = torch.linalg.inv(matrix)
        self.M = matrix.to(dtype=w.dtype, device=w.device)
        self.M_inv = inverse.to(dtype=w.dtype, device=w.device)

        s = torch.zeros(len(MOMENTS), dtype=w.dtype, device=w.device)
        s[1] = s[2] = self.s_e
        s[4] = s[6] = self.s_q
        s[7] = s[8] = self.s_nu
        self.s = s.view(-1, 1, 1)

    def __call__(
        self,
        f: torch.Tensor,
        rho: torch.Tensor,
        u: Sequence[torch.Tensor],
        force: tuple[float, ...] | None = None,
    ) -> torch.Tensor:
        feq, cu = equilibrium_with_cu(self.w, self.c, rho, u)

        m = torch.tensordot(self.M, f, dims=1)
        m_eq = torch.tensordot(self.M, feq, dims=1)
        m_post = m - self.s * (m - m_eq)

        if force is not None:
            src = 0.0
            for d, cd in enumerate(self.c):
                # (c_d - u_d)/cs^2 + (c·u)/cs^4 * c_d
                src = src + (3.0 * (cd - u[d]) + 9.0 * cu * cd) * force[d]
            m_src = torch.tensordot(self.M, self.w * src, dims=1)
            m_post = m_post + (1.0 - 0.5 * self.s) * m_src

        f_post = torch.tensordot(self.M_inv, m_post, dims=1)

        # float32 下 M_inv @ (M f) 不是精确的恒等映射，守恒矩每步会漂移
        # （周期性通道里表现为 uy 随步数线性增长）。把密度与动量投影回
        # m_post 的目标值；精确算术下这个修正恒为 0。
        target_jx = m[3] + (force[0] if force is not None else 0.0)
        target_jy = m[5] + (force[1] if force is not None else 0.0)
        drho = m[0] - f_post.sum(dim=0)
        djx = target_jx - (f_post * self.c[0]).sum(dim=0)
        djy = target_jy - (f_post * self.c[1]).sum(dim=0)
        f_post = f_post + self.w * (
            drho + 3.0 * (self.c[0] * djx + self.c[1] * djy)
        )

        return f_post
