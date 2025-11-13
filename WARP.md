# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## 基本信息
- 项目目标：使用 PyTorch 从零实现 Transformer，并提供数据下载、预处理、训练与推理脚本。
- 主要语言与运行环境：Python 3.8+，PyTorch（CUDA 可选）。
- 依赖安装：`pip install -r requirements.txt`

## 常用命令（Windows PowerShell）
- 设置 PYTHONPATH（在项目根目录下执行）
  - 当前会话：`$env:PYTHONPATH = "$env:PYTHONPATH;$(Get-Location)"`
- 安装依赖
  - `pip install -r requirements.txt`
- 下载数据（默认仅 1 个训练分片 + 验证集）
  - `python scripts/data_download.py --train_max 1 --train_dir data/train --val_dir data/val`
- 预处理数据（将 jsonl.zst 转为 HDF5 tokens）
  - `python scripts/data_preprocess.py --train_dir data/train --val_dir data/val --out_train_file data/train/pile_train.h5 --out_val_file data/val/pile_dev.h5 --tokenizer_name r50k_base --max_data 1000`
- 训练模型（配置见 `config/config.py`）
  - `python scripts/train_transformer.py`
- 文本生成（推理）
  - `python scripts/generate_text.py --model_path models/transformer_B.pt --input_text "hello" --max_new_tokens 100`

说明与注意：
- 以上命令均应在仓库根目录执行（与 `requirements.txt` 同级）。
- 训练默认设备在 `config/config.py` 中由 `DEVICE` 控制（默认为 `cuda`）。如需 CPU 推理，可在生成脚本中传入 `--model_path` 和 `--input_text` 后，修改脚本缺省或在代码中将设备改为 `cpu`（当前生成脚本函数默认参数为 `cuda`）。
- 本仓库未包含测试框架与测试用例，因此没有“运行单测”的命令。
- 本仓库未提供专门的构建、打包或 lint 配置（无 pyproject/ruff/flake8 配置），如需添加请在后续工作中补充。

## 高层架构与代码结构（大图景）
本项目围绕“数据 → 预处理（分词到 HDF5） → 批加载 → Transformer 训练 → Checkpoint 保存 → 推理”这条主线组织：

1) 配置中心（超参数与路径）
- 路径：`config/config.py`
- 内容：
  - 模型超参：`VOCAB_SIZE`, `CONTEXT_LENGTH`, `N_EMBED`, `N_HEAD`, `N_BLOCKS`
  - 训练超参：`T_BATCH_SIZE`, `T_CONTEXT_LENGTH`, `T_TRAIN_STEPS`, `T_EVAL_STEPS`, `T_EVAL_ITERS`, `T_LR`, `T_LR_DECAYED`, `T_LR_DECAY_STEP`
  - 数据路径：`TRAIN_PATH`, `DEV_PATH`
  - 设备与输出：`DEVICE`, `T_OUT_PATH`
- 作用：统一被训练与推理脚本读取，驱动模型规模与训练流程。

2) 数据阶段
- 下载脚本：`scripts/data_download.py`
  - 从 HuggingFace 链接下载验证集 `val.jsonl.zst` 与指定数量训练分片（`--train_max`）。
- 预处理脚本：`scripts/data_preprocess.py`
  - 读取 `*.jsonl.zst`，用 `tiktoken` 的 `r50k_base` 分词，将 token 序列（追加 `<|endoftext|>`）线性写入 HDF5 数据集 `tokens`。
  - 产物：`data/train/pile_train.h5`, `data/val/pile_dev.h5`
- 批数据迭代器：`data_loader/data_loader.py`
  - 函数：`get_batch_iterator(data_path, batch_size, context_length, device)`
  - 从 HDF5 的线性 token 序列切片出长度为 `context_length+1` 的样本，并随机打乱、按批产出 `(xb, yb)`（左移一位的 next-token 标签）。

3) 模型与子模块
- 顶层模型：`src/models/transformer.py`（类 `Transformer`）
  - 组件：
    - Token/Position Embedding
    - N 层 `Block` 堆叠（见下一小节）
    - 最终 `LayerNorm` 与 `lm_head`（线性输出 vocab logits）
  - 前向：
    - `forward(idx, targets=None)` 计算 logits 与可选的交叉熵损失
    - `generate(idx, max_new_tokens)` 自回归生成，逐步采样下一个 token
- 基础模块：`src/models/transformer_block.py`（类 `Block`）
  - 结构：`LN → MultiHeadAttention → Residual`，`LN → MLP → Residual`
- 注意力与 MLP：`src/models/attention.py`, `src/models/mlp.py`
  - 将多头注意力与前馈网络解耦，供 `Block` 复用。

模块关系（自顶向下）：
- `scripts/train_transformer.py` 读取 `config`，构建 `Transformer`，通过 `get_batch_iterator` 获取批数据 → 训练循环（forward/backward/step + 周期评估 + 学习率衰减）→ 保存 checkpoint（包含模型权重、优化器、loss 轨迹）。
- `scripts/generate_text.py` 加载 checkpoint 与同构的 `Transformer` → 编码输入 → `model.generate()` → 解码输出文本。

4) 训练与评估逻辑
- 训练脚本：`scripts/train_transformer.py`
  - 关键点：
    - 按 `config` 初始化模型与优化器（AdamW）
    - 训练中定期 `estimate_loss()` 在 train/dev 上评估平均损失（开关 eval 模式、重置为 train 模式）
    - 达到 `T_LR_DECAY_STEP` 后将学习率从 `T_LR` 降到 `T_LR_DECAYED`
    - 结束时保存 checkpoint 至 `models/`（如重名则追加后缀防覆盖）

5) 推理
- 推理脚本：`scripts/generate_text.py`
  - 与训练同构的超参数（从 `config` 读取）实例化 `Transformer`，加载权重，使用 `tiktoken` 编码与解码，调用 `generate` 进行自回归采样。

## 与 README 的要点对齐
- README 提供了从数据准备、模型组件解释、到训练与生成的详解与示例片段。
- 本 WARP.md 提炼了“如何运行”的关键命令与“模块如何协作”的高层视图，便于快速上手实际开发与调试。

## 后续可选增强（非现状，仅建议）
- 增加测试与最小用例（例如对 `get_batch_iterator` 与 `Transformer.generate` 的快速单测）。
- 引入 lint/格式化与 CI（ruff/black 或 pre-commit），并在此文件补充对应命令。
