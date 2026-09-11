# 导出与推理部门（zh-CN）

## 1. 部门概述

本部门包含两项训练后操作：

- `export.py` 将 pre 和 post 模型导出为 ExecuTorch `.pte` 程序。
- `demo.py` 从 PyTorch checkpoint 执行 greedy 或 beam-search 推理。

## 2. `export.py` — ExecuTorch 导出

导出器从 checkpoint 目录加载 `pre_model/` 和 `post_model/`，生成包含多个
方法的 `pre_model.pte`、`post_model.pte`，以及记录两个程序所需算子和 dtype
变体的选择性构建清单。

pre 程序分别包含前缀解码、条件解码和交叉注意力 KV 投影方法。导出过程使用
动态序列维度，并可应用 XNNPACK 动态逐通道量化。

当前通过 `python export.py` 运行；checkpoint、输出路径、模型元数据、beam
宽度、dtype 和量化模式暂时由文件顶部的模块常量配置。

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
