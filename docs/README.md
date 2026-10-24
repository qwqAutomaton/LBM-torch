# LBM on PyTorch —— 开发文档

这个目录是项目从零开始的实施依据。用途是先把 D2Q9 跑通、验证正确，
之后再改造成滑移边界 + 可变粘度 + 高 Knudsen 的论文版本。

所以整体的取舍是：**数值正确性和"换模型不改代码"优先，性能第二**。
性能只要在目标硬件上跑得动即可（M2 / MPS 做小算例，4060Ti / CUDA 做中等规模）。

## 文档索引

| 文件 | 内容 |
|---|---|
| `physics.md` | 格子单位、D2Q9 方向表、平衡态、BGK/TRT/MRT + Guo、pull streaming、半程反弹、周期与 Zou/He 边界、有效壁面与解析解 |
| `architecture.md` | 目录结构、模块职责、公开 API、数据流、双缓冲、IO 契约 |
| `plan.md` | P0~P8 实施顺序，每阶段的验收标准 |
| `validation.md` | 守恒/不动点/力响应测试、Poiseuille 算例参数与容差、收敛阶 |
| `pitfalls.md` | 数值、PyTorch、MPS/CUDA、性能上的已知坑 |
| `research_mrt.md` | MRT 调研与集成指导：矩空间公式（本项目方向序）、参数 SOP、验证清单、陷阱 |

## 一页纸设计决定

| 项 | 决定 |
|---|---|
| 模型 | D2Q9 实现；D3Q19 / D3Q27 只留类和工厂入口，`initialize` / `step` 抛 `NotImplementedError` |
| 架构 | 特化 + 工厂。三个 solver 各自实现 streaming，共享 python 层面的工具函数 |
| 布局 | `pdf: (2, Q, Nx, Ny)`，`solid: (Nx, Ny)` bool |
| 双缓冲 | 维护 `cur_idx ∈ {0,1}`，每步读 `pdf[cur_idx]` 写 `pdf[1-cur_idx]`，然后翻转索引。不做整块 copy |
| 碰撞 | `CollisionOperator` 接口；TRT（`Λ=3/16`）为默认，BGK / MRT 可切换 |
| streaming | pull，不用 `torch.roll`，逐方向切片写入目标 buffer |
| 固体 | 半程反弹（halfway bounce-back），固体节点不参与碰撞，只在 pull 时被"反射"覆盖 |
| 边界 | 默认周期；`boundary="inlet_outlet"` 时左 Zou/He 速度入口、右 Zou/He 密度出口 |
| 单位 | 格子单位，`cs^2 = 1/3` |
| 精度 | 计算精度与输出精度分开指定，默认 `float32` / `float16` |
| 驱动 | 默认不加力；Guo forcing 可选，外力是体积力密度 |
| compile | 构造参数，只编译 `_advance`；失败直接抛异常 |
| 初始条件 | 外部传入密度场和速度场 |
| 时间步 | solver 内部维护，写进保存文件名 |
| 进度条 | `lbm.progress.progress` / `run_steps`，tqdm（必需依赖）分块刷新，不进计算图 |
| 输出 | npz 已实现（存 PDF）；h5（PDF）/ vtk（宏观量）只有签名 |
| 用途 | 算法验证 + 论文改动 |

## 需要提醒的两点

第一，`float16` 输出只适合看场、画图、省磁盘。做定量对比（比如 Poiseuille
误差分析）时把 `output_dtype` 显式指定成 `float32`，否则误差会被量化噪声淹没。

第二，v1 不做多卡、不做自由面、不做传热。这些不要提前设计接口，写进去的抽象
如果当下用不上，改造时反而更难删。
