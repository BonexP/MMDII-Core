# CPU 训练主机环境配置

本文记录在无 CUDA 的 Linux/JupyterLab 训练主机上，使用 `uv` 管理独立
Python 3.11 环境并安装 MMDII-Core 的标准流程。该流程不会替换或升级系统
Python，也不依赖 Conda。

## 适用范围

- 系统仅提供 Python 3.8，但不希望修改系统解释器；
- 训练主机没有 NVIDIA GPU，需要使用 CPU 版 PyTorch；
- 已检出 MMDII-Core 的 `codex/moderntcn-mil-baseline` 分支；
- 命令在 MMDII-Core 仓库根目录执行。

先确认分支和项目文件：

```bash
git branch --show-current
git log -1 --oneline
test -f pyproject.toml && echo "project root OK"
```

预期分支为 `codex/moderntcn-mil-baseline`。公开仓库的 `main` 当前不包含
Dataset v0.2 与模型训练实现，因此不能用它执行本流程。

## 为什么不能使用 Python 3.8

MMDII-Core 的 `pyproject.toml` 要求 Python 3.11 或更高版本，训练依赖包括
`scipy>=1.11`、`torch>=2.2` 和 `scikit-learn>=1.4`。在 Python 3.8 环境中
执行 `pip install -e ".[train]"` 会因找不到兼容的 SciPy 版本而终止，后续
PyTorch 也不会被安装。

不要通过降低 SciPy 版本规避这个约束。训练代码还使用 Python 3.11 标准库
功能，正确处理方式是为项目创建独立的 Python 3.11 环境。

## 使用 uv 创建 Python 3.11 环境

如果仓库中已有由 Python 3.8 创建的 `.venv`，先停用并移走。若无需保留，
也可以自行删除后再继续。

```bash
deactivate 2>/dev/null || true
mv .venv .venv-py38-old
```

安装 `uv`，并让当前 shell 立即识别命令：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"
uv --version
```

安装由 `uv` 管理的 Python 3.11：

```bash
uv python install 3.11
uv python list --only-installed
uv python find 3.11
```

创建并激活项目虚拟环境：

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
python --version
which python
cat .venv/pyvenv.cfg
```

`python --version` 必须显示 `3.11.x`，`which python` 必须指向当前仓库的
`.venv/bin/python`。系统解释器不会被修改，可单独检查：

```bash
/usr/bin/python3 --version
```

## 安装 CPU 训练依赖

明确选择 PyTorch CPU wheel，避免下载无用的 CUDA 运行时：

```bash
uv pip install --python .venv/bin/python --torch-backend=cpu -e ".[train]"
```

安装完成后验证解释器、科学计算依赖和 PyTorch：

```bash
python -c "import sys, scipy, sklearn, torch; print('python=', sys.version); print('scipy=', scipy.__version__); print('sklearn=', sklearn.__version__); print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('tensor=', torch.zeros(2, 3).sum())"
```

CPU 主机的正常结果包含：

```text
python= 3.11.x
cuda= False
tensor= tensor(0.)
```

`cuda=False` 是预期状态，不是错误。

## 检查代码与真实数据

先运行 Core 测试：

```bash
python -m unittest discover -s tests -v
```

然后检查训练环境和 Dataset v0.2 发布目录：

```bash
python scripts/check_training_environment.py \
  --config configs/moderntcn_mil_v0_1.toml \
  --release-dir /path/to/mmdii-v0-2/releases/<release-id> \
  --json
```

最后强制使用 CPU 完成一次真实数据优化步骤：

```bash
python scripts/smoke_train.py \
  --config configs/moderntcn_mil_v0_1.toml \
  --release-dir /path/to/mmdii-v0-2/releases/<release-id> \
  --fold 0 \
  --batch-size 1 \
  --device cpu
```

冒烟命令只验证真实数据读取、预处理、前向传播、损失、反向传播和一次参数
更新，不生成正式五折研究结论。

## 可选：注册 JupyterLab kernel

终端运行训练脚本不需要注册 kernel。仅在 Notebook 中使用该环境时执行：

```bash
uv pip install --python .venv/bin/python ipykernel
python -m ipykernel install \
  --user \
  --name mmdii-py311 \
  --display-name "MMDII Python 3.11 CPU"
```

随后在 JupyterLab 中选择 `MMDII Python 3.11 CPU`。

## 2026-08-12 训练主机实测记录

本流程已经在一台无 GPU 的 Linux/JupyterLab 主机上完成环境安装，系统
Python 保持为 `3.8.10`。实际验证版本如下：

| 组件 | 实测版本或结果 |
| --- | --- |
| `uv` | `0.12.3` |
| uv 托管 CPython | `3.11.15` |
| SciPy | `1.17.1` |
| scikit-learn | `1.9.0` |
| PyTorch | `2.13.0+cpu` |
| `torch.cuda.is_available()` | `False` |
| CPU tensor 自检 | `tensor(0.)` |

实际过程先在 Python 3.8 虚拟环境中失败，错误表现为无法满足
`scipy>=1.11`；切换到 uv 托管的 Python 3.11 后，17 个项目与训练依赖成功
解析和安装。该结果只证明环境依赖已连通，真实 Dataset v0.2 的环境检查和
冒烟训练仍需分别执行。
