# 导出与推理部门（zh-CN）

## 1. 部门概述

本部门包含两项训练后操作：

- `export/` 实现将 pre 和 post 模型导出为 ExecuTorch `.pte` 程序的 Hydra task。
- `demo.py` 从 PyTorch checkpoint 执行 greedy 或 beam-search 推理。

## 2. `export/` — ExecuTorch 导出任务

导出器从 checkpoint 目录加载 `pre_model/` 和 `post_model/`，生成包含多个
方法的 `pre_model.pte`、`post_model.pte`，以及记录两个程序所需算子和 dtype
变体的选择性构建清单。

pre 程序分别包含前缀解码、条件解码和交叉注意力 KV 投影方法。导出过程使用
动态序列维度，并可应用 XNNPACK 动态逐通道量化。

通过统一入口运行：

```bash
python main.py task=export
```

`config/task/export.yaml` 定义输入 checkpoint、输出目录和文件名、目标设备与
dtype、模型元数据、代表性输入尺寸、量化模式（`none`、`w8a8` 或 `w4a8`）、
计算图 strict/打印设置、内存规划行为及选择性构建清单配置。所有字段均可通过
Hydra 覆盖，例如：

```bash
python main.py task=export \
  task.checkpoint_dir=checkpoints/my-run/final_model \
  task.output_dir=export_output/my-run \
  task.quantization.mode=w8a8
```

`export/task.py` 校验运行配置、相对 Hydra 原始工作目录解析路径、捕获四个动态
计算图、执行量化和 XNNPACK lower、写出两个程序，并按配置生成逐模型及合并的
算子清单。

## 3. `demo.py` — 推理演示

Demo 通过命令行接收 checkpoint、词表配置、上下文、拼音、beam size、设备和
dtype：

```bash
python demo.py \
  --checkpoint checkpoints/<run>/final_model \
  --text "" \
  --pinyin ni hao \
  --beam-size 3
```

程序首先执行 greedy 解码；当 beam size 大于一时，继续输出 N-best beam-search
结果及耗时。
