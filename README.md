# LBM-torch

USED LLM EXCLUSIVELY.

格子 Boltzmann 求解器。v1 只实现 **D2Q9**，PyTorch 后端，可跑 CPU / MPS / CUDA。
架构上按"换模型不改代码"设计，下一步是滑移边界 + 可变粘度。

文档在 `docs/`，先读 `docs/README.md`。

## 安装

```bash
pip install -e .     # tqdm 是必需依赖，会一并装上
# 跑测试还需要
pip install pytest
```

## 最小用法

```python
import torch
from lbm import create_solver

nx, ny = 256, 64
solid = torch.zeros((nx, ny), dtype=torch.bool)
solid[:, 0] = True          # 下壁
solid[:, -1] = True         # 上壁

solver = create_solver(
    "D2Q9",
    (nx, ny),
    solid,
    nu=0.02,                # 运动粘度，tau = 3*nu + 0.5
    collision="trt",        # 默认；可选 "bgk" / "mrt"
    Lambda=3.0 / 16.0,      # TRT magic parameter，默认就是 3/16
    force=(1e-6, 0.0),      # 体积力密度，默认不加力
    device="mps",
    compute_dtype=torch.float32,
    output_dtype=torch.float16,
)

solver.initialize(rho=1.0, u=None)     # 平衡态，可传场
for _ in range(10):
    solver.step(1000)
    print(solver.diagnose())
```

保存原始 PDF（每 N 步一次）：

```python
solver.dump_setup("out/run", total_steps=100_000, save_interval=1000)  # metadata.json + solid.npz
for _ in range(100):
    solver.step(1000)
    solver.save_npz("out/run")       # pdf_00001000.npz, pdf_00002000.npz, ...
    solver.diagnose()
```

## 约定

- 全部用格子单位：`dx = dt = 1`，`cs^2 = 1/3`
- `pdf: (2, Q, Nx, Ny)`，`solid: (Nx, Ny) bool`，`True` 表示固体
- 速度定义 `u = (sum(f*c) + F/2)/rho`，全局唯一
- x、y 默认都是周期边界；壁面用域内固体行 + 半程反弹实现
- `boundary="inlet_outlet"` 时 x 改为 Zou/He 左速度入口 + 右密度出口（用法见 `examples/karman.py`）
- 保存的 `pdf` 是当前时刻的 buffer，形状 `(Q, Nx, Ny)`

## 测试

```bash
pytest              # 快速测试
pytest -m slow      # Poiseuille 剖面与网格收敛 + MRT 低粘度稳定性；本机 cpu+cuda 实测约 6 分钟
```

## 例子

```bash
python examples/poiseuille.py --device mps
python examples/karman.py --device mps --quick     # 圆柱绕流，Zou/He 入口出口
python examples/bench.py --shape 512 512 --steps 500 --device mps
```

## 尚未实现

- `D3Q19` / `D3Q27`：只有速度集元数据和类，`initialize` / `step` 抛 `NotImplementedError`
- HDF5 / VTK 输出（`lbm/io/h5.py`、`lbm/io/vtk.py` 是 stub）
- D3 MRT / 滑移边界 / 多相 / 传热 / 多卡
