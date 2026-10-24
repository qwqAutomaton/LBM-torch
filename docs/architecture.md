# 项目结构与接口

## 目录

```
LBM-torch/
├── docs/                       # 本目录
├── lbm/
│   ├── __init__.py             # 导出 create_solver、Diagnostics、progress、run_steps、__version__
│   ├── stencils/
│   │   ├── __init__.py
│   │   ├── base.py             # LatticeSpec / StencilTensors / make_tensors / CS2
│   │   ├── d2q9.py             # 方向表（c/w/opp、VELOCITY_NAMES）
│   │   ├── d3q19.py            # 只是常量表，供未来使用
│   │   └── d3q27.py            # 张量积生成，只是常量表
│   ├── equilibrium.py          # feq(rho, u) —— 所有模型共用
│   ├── macros.py               # 从 f 求 rho/u，含 Guo 的 F/2 修正
│   ├── collide/
│   │   ├── __init__.py
│   │   ├── base.py             # CollisionOperator 协议
│   │   ├── bgk.py              # BGK(tau) + Guo 源项
│   │   ├── trt.py              # TRT(tau+, tau-)
│   │   └── mrt.py              # MRT（LL 矩基 + 列置换），D2Q9
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseSolver：双缓冲、IO、诊断、step 循环
│   │   ├── d2q9.py             # D2Q9Solver：_stream 与 Zou/He 在这里
│   │   ├── d3q19.py            # 只有元数据，initialize/step 抛 NotImplementedError
│   │   ├── d3q27.py            # 同上
│   │   └── factory.py          # create_solver(model=...)
│   ├── diagnostics.py          # Diagnostics 数据类
│   ├── progress.py             # tqdm 进度条 + run_steps 分块推进
│   └── io/
│       ├── __init__.py
│       ├── npz.py              # 唯一真正实现的输出（含 read_* 读接口）
│       ├── h5.py               # stub：存 PDF
│       └── vtk.py              # stub：存宏观量
├── examples/
│   ├── poiseuille.py           # 端到端示例，也是 validation 的入口
│   ├── karman.py               # 圆柱绕流（Zou/He 入口出口 + 涡量动画）
│   └── bench.py                # MLUPS 性能基准
├── tests/
│   ├── conftest.py             # device fixture：cpu / mps / cuda
│   ├── test_stencils.py
│   ├── test_equilibrium.py
│   ├── test_collision.py
│   ├── test_streaming.py
│   ├── test_forcing.py
│   ├── test_io.py
│   ├── test_factory.py         # 工厂与 D3 stub
│   ├── test_trt.py             # TRT 退化 / 守恒 / 低粘度稳定性
│   └── test_poiseuille.py      # 标 slow
├── pyproject.toml
└── README.md
```

`lbm/` 里不放任何 `if device == 'mps'` 之类的分支，设备差异统一在
`solver.__init__` 里处理一次（dtype 校验、`torch.device` 解析）。

## 模块职责划分

`equilibrium.py` 和 `macros.py` 是与模型无关的，用 `(Q,1,1)` / `(1,Nx,Ny)`
广播就能同时给 D2Q9 和 D3Q19 用。可以放通用实现，因为这里的 `Nx` 维度
不影响 `Q` 的方向语义。

`collide/` 只关心"给我 `f, rho, u, force`，还你 `f_post`"。
TRT 把方向按对称/反对称分组，用 `__init__` 里预置的 `opp` 索引完成拆分；
MRT 在构造时按速度向量对 LL 矩矩阵做列置换、求逆并缓存，两者都不影响外部接口。

`solvers/` 是真正特化的地方。`_stream()` 每个模型写一份，
因为不同 `c_i` 的切片组合确实不一样，抽象出来的通用版要么慢要么难读。
按用户的要求，重复写三遍是可以接受的成本。

`BaseSolver` 只放与维度无关的部分：构造校验、双缓冲管理、`step()` 循环、
`diagnose()`、IO 调用。**不**放 streaming，也不放碰撞。

`stencils/base.py` 负责两件事：`LatticeSpec.check()` 在构造 solver 时校验
c/w/opp 的自洽性（对合、一阶矩、二阶矩），`make_tensors()` 把常量搬到
device 并预展开成 `(Q,1,1)`，打包成 `StencilTensors`。`progress.py` 是
纯 python 层：tqdm 进度条与 `run_steps()` 分块推进，不参与计算图。

## 公开 API

```python
def create_solver(
    model: str,                          # "D2Q9" | "D3Q19" | "D3Q27"（大小写/空格/下划线不敏感）
    shape: tuple[int, ...],              # (Nx, Ny) 或 (Nx, Ny, Nz)
    solid: torch.Tensor | None = None,   # bool，形状同 shape；None 表示全流体
    *,
    nu: float,                           # 运动粘度，tau = 3*nu + 0.5
    Lambda: float | None = None,         # TRT magic parameter，None 时取 3/16
    collision: str = "trt",              # "trt" | "bgk" | "mrt"（MRT 仅 D2Q9）
    force: tuple[float, ...] | torch.Tensor | None = None,
    device: str | torch.device = "cpu",
    compute_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype = torch.float16,
    compile: bool = False,
    # 以下三个只由 D2Q9Solver 接受：
    boundary: str = "periodic",          # "periodic" | "inlet_outlet"
    u_inf: float = 0.0,                  # 左边界入口速度（仅 inlet_outlet）
    rho_inf: float = 1.0,                # 右边界出口密度（仅 inlet_outlet）
) -> BaseSolver
```

工厂对 `"D3Q19"` / `"D3Q27"` 正常返回对应的类实例，是**实例化成功、调用即抛错**，
不是工厂直接抛错。这样 `model` 拼写错误（比如 `"D2Q8"`）和"未实现"是两种
不同的错误，排查时不会混。实际抛错的位置是 `initialize()` / `step()` /
`macro()` 里的 `_check_implemented()`；`diagnose()` / `save_npz()` /
`dump_setup()` 不做这个检查。

构造期校验（都在 `BaseSolver.__init__` / `D2Q9Solver.__init__` 里一次做完）：

- `nu > 0`（`tau = 3*nu + 0.5` 必然 `> 0.5`）；`collision ∈ {"bgk","trt","mrt"}`；
  `Lambda > 0`（TRT / MRT）；`collision="bgk"` 又传 `Lambda` 时发
  `RuntimeWarning` 并把 `Lambda` 置 `None`
- MRT 还要求 `s_q = 1/(0.5 + Lambda/(tau-0.5))` 落在 `(0,2)`；`s_e = s_eps = 1.9` 是固定值
- `shape` 维数必须等于 stencil 维数、每维为正；`solid` 形状必须与 `shape` 一致
- `compute_dtype` 只接受 `float32` / `float64`；MPS 上给 `float64` 直接报错；
  指定了不可用的 `cuda` / `mps` 报 `RuntimeError`
- `output_dtype` 当前**不校验**，只影响 `save_npz()` 里的 `.numpy()`，
  非法 dtype 会在保存时报错
- `boundary="inlet_outlet"` 时还要求：`u_inf ∈ (-1,1)`、`rho_inf > 0`，
  且 `x=0` 与 `x=Nx-1` 两列全为流体

Solver 的实例方法：

```python
initialize(rho=1.0, u=None) -> None
    # rho: float 或 shape 张量；u: tuple 或 (dim, *shape) 张量
    # 若 u is None 则零速度；feq 写进 pdf[0]，pdf[1] 清零，固体节点置 0
    # 同时把 step_count 与 cur_idx 归零

set_force(force) -> None
    # 更新体积力，可随时调用，影响后续步

step(delta_step: int = 1) -> None
    # Python 层循环调用 self._advance(src, dst)，再翻转 cur_idx

macro() -> tuple[Tensor, Tensor, Tensor]
    # 返回 (rho, ux, uy)，完整场，在 device 上，未做 CPU 拷贝
    # 供验证脚本做剖面分析用；不要放进热循环

diagnose() -> Diagnostics
    # 只统计流体节点；有 NaN 时 rho_min/rho_max/max_velocity 都返回 NaN

save_npz(directory, prefix="pdf") -> Path
    # 写 <directory>/<prefix>_<step:08d>.npz，内含当前 buffer 的 pdf

dump_setup(directory, *, total_steps, save_interval, extra=None) -> tuple[Path, Path]
    # 写 metadata.json + solid.npz，在模拟循环之前调用一次
```

模块级辅助函数（与 solver 一起从 `lbm` 导出）：

```python
progress(total, desc=None, **kwargs)        # contextmanager；tqdm 缺失时退化为空壳
run_steps(solver, total_steps, *, desc=None, chunk=500, **kwargs) -> None
    # 把 total_steps 分块喂给 solver.step()，只为让进度条有机会刷新
```

只读属性：`model`、`shape`、`device`、`compute_dtype`、`output_dtype`、
`nu`、`tau`、`step_count`、`spec`，以及 `Lambda`、`force`、`collision`
（当前碰撞算子实例）、`collision_info`（写进 metadata 的碰撞参数）。

`step_count` 从 0 开始，`step(n)` 之后是 `n`。它同时决定保存文件名，
所以不要让用户手动传步数。

## 内部数据流

```
step(delta_step):
    for _ in range(delta_step):
        src  = self.pdf[self.cur_idx]          # (Q, Nx, Ny)，只读
        dst  = self.pdf[1 - self.cur_idx]      # (Q, Nx, Ny)，写
        self._advance(src, dst)                # compile=True 时编译的就是它
        self.cur_idx = 1 - self.cur_idx
        self._step_count += 1

_advance(src, dst):                            # 纯函数式：不碰 cur_idx/count
    rho, u = macroscopic(src, self._st.c, self.force)   # 含 F/2 修正
    f_post = self.collision(src, rho, u, self.force)   # 含 Guo 源项
    self._stream(f_post, dst)                      # 逐方向切片写入 + 反射覆盖
    self._apply_open_boundary(dst)                 # 已实现完整 Zou/He
```

顺序不能调换：`_apply_open_boundary` 必须在 `_stream` 之后，
因为入口出口要修正的是"刚被拉进来、还缺方向"的分布函数。

`macros` 用 `F/2` 修正、`collision` 用修正后的 `u`，两者必须一致。

注意 `_step_count` 和 `cur_idx` 的更新都留在了 `step()` 里，**不进入** `_advance`。
原因见下一节。

## 双缓冲与 cur_idx

`self.pdf` 是一个 `(2, Q, Nx, Ny)` 的张量，`self.cur_idx ∈ {0, 1}`。

不写 `pdf[0] = pdf[1]`，也不做 `pdf_a, pdf_b = pdf_b, pdf_a` 这种引用交换。
固定一块 `(2, ...)` 的显存、只翻转一个 int，是为了让 `torch.compile`
的输入签名稳定——引用交换会让每次调用的 tensor 身份变化，触发反复重新编译。

**关于"两个图变体"**：一开始的设计是把 `cur_idx` 当作 Python int 传进去，
`torch.compile` 会按它的值各生成一份图，缓存里稳定是两个变体——这在功能上
没有问题。实际实现换了个更省事的写法：把 `src` / `dst` 两块视图作为参数
传给 `_advance(src, dst)`，两块视图的 shape/dtype/stride 完全一致，所以
`cur_idx` 根本不进入被编译的函数，只生成**一份**图。两种都行，
但如果以后把 `cur_idx` 塞回去，记住它会让缓存变成两份，这是预期行为，
不要为了"只编译一次"改用 tensor 索引——那样会引入一次额外的 gather。

同理，`step(delta_step)` 不要被编译。`delta_step` 每次可能不同，
编译多步循环会让缓存膨胀。

## compile 开关

```python
if compile:
    self._advance = torch.compile(self._advance)
```

失败就让它抛，不 try/except、不自动回退。用户要求如此，也确实合理：
静默回退会让人以为在跑编译版，实际上跑的是 eager，性能数据全错。

编译前的实现必须满足：
- 没有 `.item()`、`.cpu()`、`.numpy()`
- 没有依赖张量值的 `if` / `while`
- 所有常量张量（`c`、`w`、`opp` 索引、固体 mask）在 `__init__` 里预置到 device 上

如果 `torch.compile` 报 graph break，先修代码，不要加 `fullgraph=False` 绕过。

## streaming 的实现要点

对方向 `i`，目标 `dst[i]` 是 `f_post[i]` 在 `(-cx, -cy)` 方向的移位，
外加反射覆盖。移位是环绕的。

以 `c = (+1, +1)` 为例：

```
dst[i, 1:, 1:] = f_post[i, :-1, :-1]
dst[i, 0, 1:]  = f_post[i, -1, :-1]
dst[i, 1:, 0]  = f_post[i, :-1, -1]
dst[i, 0, 0]   = f_post[i, -1, -1]
```

`c = (0, 0)` 时直接 `dst[0] = f_post[0]`。

实现上没有手写这些切片，而是 `write_shifted()` 用 `axis_slices()`（周期）
或 `open_slices()`（`inlet_outlet` 的 x 方向）拼出同样的组合。`wrap_x=False`
时目标里没被写到的列保持原值，随后由 `_apply_open_boundary()` 填——这正是
入口/出口需要的语义。

反射覆盖：在 `__init__` 里对每个方向预计算一个 `wall_mask_i`（`(Nx,Ny)` bool），
含义是"目标节点 x 的源点 x-c_i 是固体"。`_stream` 里：

```
if self._has_wall[i]:                        # 预计算的 Python bool
    dst[i].copy_(torch.where(wall_mask_i, f_post[opp[i]], dst[i]))
```

必须是"先算临时张量再 `copy_`"，不能用 `torch.where(..., out=dst[i])`：
输出与输入别名时 MPS 上会算错（CPU 上正确），详见 `pitfalls.md`。

`_has_wall[i]` 是预存的 Python bool，没有壁面的方向整支跳过，
Poiseuille 里 `cy = 0` 的 5 个方向都不会进 `where`。

不要用 `f_post[solid_mask]` 这种布尔索引，它会触发 device→host 同步。

更进一步的优化（v1 不做，记在这里）：几何固定时可以把壁面 link 的扁平索引
预计算出来，用 `index_put_` 只改边界那几十个点。等真的觉得慢了再说。

## 精度

```
compute_dtype: float32（默认）/ float64（仅 CPU、CUDA）
output_dtype:  float16（默认）/ float32
```

构造时校验：MPS 上给 `float64` 直接报错（MPS 不支持），不要等运行时才炸。

`output_dtype` 只在写文件时生效，不改变计算 buffer。
转换路径固定为 `detach().cpu().to(output_dtype).numpy()`，
顺序别反，MPS 上直接 `float16 -> numpy` 容易出意外。

## 输出契约

保存前调用 `dump_setup(directory, total_steps=..., save_interval=...)`，
写两个文件：

`metadata.json`，至少包含

```
model, shape, nu, tau, Lambda, force, device,
compute_dtype, output_dtype, total_steps, save_interval,
initial_rho / initial_u（描述字符串，未 initialize 时为 null）,
created_at, lbm_version
```

子类通过 `_extra_meta()` 还会补上运行参数：`collision`（"trt"/"bgk"/"mrt"）、
TRT 的 `tau_minus`、MRT 的 `s_nu` / `s_e` / `s_q`、`boundary`，
`inlet_outlet` 时再加 `u_inf` / `rho_inf`。
`extra=` 传入的键在最后覆盖。

`solid.npz`，键名 `solid`，值是 CPU 上的 bool 数组 `(Nx,Ny)`。

每步保存 `pdf_<step:08d>.npz`，键名 `pdf`，形状 `(Q, Nx, Ny)`，
**只存当前 buffer**。文件里不放宏观量，因为宏观量随时可以从 pdf 重构，
放了反而容易出现"pdf 和 rho 不同步"的诡异 bug。

`h5.py` / `vtk.py` 里只写函数签名加 `raise NotImplementedError`，
不要写半成品。签名按"存什么"定，而不是照抄 npz：

- `write_pdf_h5(path, pdf, **kwargs)`：和 npz 一样存 PDF
- `write_vtk(path, rho, u, **kwargs)`：存宏观量（密度 + 速度场），给 ParaView 用

`npz.py` 同时提供 `read_pdf()` / `read_solid()` / `read_metadata()`，
测试与后续分析脚本用它们读回。

## 诊断

`Diagnostics` 是 dataclass：`step, total_mass, max_velocity, rho_min, rho_max`。

统计范围只到流体节点。检查 NaN 时用：

```
if not torch.isfinite(rho_fluid).all():
    全部返回 NaN
```

`max_velocity` 也要参与这个判断——只检查密度、不检查速度，会出现
"密度正常但速度已经发散"的窗口期，看起来一切正常，其实已经坏了。

`diagnose()` 会触发同步与归约，成本远超单个时间步，**不要每步调用**，
按 `save_interval` 调一次即可。它不在 `_advance` 里，不进编译图，
也不影响热循环。
