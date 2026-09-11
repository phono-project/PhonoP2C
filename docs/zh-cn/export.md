# 导出与推理部门（zh-CN）

## 1. 部门概述

本部门包含两项训练后操作：

- `export/` 实现将 pre 和 post 模型导出为 ExecuTorch `.pte` 程序的 Hydra task。
- `demo.py` 从 PyTorch checkpoint 执行 greedy 或 beam-search 推理。

## 2. `export/` — ExecuTorch 导出任务

导出器从 checkpoint 目录加载 `pre_model/` 和 `post_model/`，生成包含多个方法的 `pre_model.pte`、`post_model.pte`，以及记录两个程序所需算子和 dtype 变体的选择性构建清单。

pre 程序分别包含前缀解码、条件解码和交叉注意力 KV 投影方法。导出过程使用动态序列维度，并可应用 XNNPACK 动态逐通道量化。

通过统一入口运行：

```bash
python main.py task=export
```

`config/task/export.yaml` 定义输入 checkpoint、输出目录和文件名、目标设备与 dtype、模型元数据、代表性输入尺寸、量化模式（`none`、`w8a8` 或 `w4a8`）、计算图 strict/打印设置、内存规划行为及选择性构建清单配置。所有字段均可通过 Hydra 覆盖，例如：

```bash
python main.py task=export \
  task.checkpoint_dir=checkpoints/my-run/final_model \
  task.output_dir=export_output/my-run \
  task.quantization.mode=w8a8
```

`export/task.py` 校验运行配置、相对 Hydra 原始工作目录解析路径、捕获四个动态计算图、执行量化和 XNNPACK lower、写出两个程序，并按配置生成逐模型及合并的算子清单。

## 3. `demo.py` — 推理演示

Demo 通过命令行接收 checkpoint、词表配置、上下文、拼音、beam size、设备和 dtype：

```bash
python demo.py \
  --checkpoint checkpoints/<run>/final_model \
  --text "" \
  --pinyin ni hao \
  --beam-size 3
```

程序首先执行 greedy 解码；当 beam size 大于一时，继续输出 N-best beam-search 结果及耗时。

## 4. `tools/build_pack_v2_2.py` — phono-core 模型包构建器

版本化构建器会先校验导出的程序、Hugging Face 模型配置和 tokenizer 词表，再组装 phono-core 模型包。文件名和内部的 `MODEL_FORMAT_VERSION` 固定为 v2.2；将来出现新的模型包 schema 时应新增构建器，而不是修改本文件。

必需输入包括 pre/post PTE、各自的 `PhonoP2CPreModel` 与 `PhonoP2CPostModel` `config.json`、Chinese/context/pinyin 三个词表、输出目录，以及以 `v2_2-` 开头的模型版本。Segment PTE 与字符词表是可选输入，但必须同时提供；也可额外提供其 Hugging Face config，以启用更多校验。

```bash
python tools/build_pack_v2_2.py \
  --pre-model export_output/pre_model.pte \
  --post-model export_output/post_model.pte \
  --pre-config checkpoints/<run>/final_model/pre_model/config.json \
  --post-config checkpoints/<run>/final_model/post_model/config.json \
  --chinese-vocab vocabs/chinese_vocab.txt \
  --context-vocab vocabs/context_vocab.txt \
  --pinyin-vocab vocabs/pinyin_vocab.txt \
  --model-version v2_2-<name> \
  --output-dir <package-dir>
```

构建器使用当前 ExecuTorch runtime 解析每个 PTE，并核对精确的方法名、输入输出形状、batch 宽度、序列上限、KV Cache 维度、cross-KV 兼容性和候选宽度。它还会检查模型架构标识、attention 维度、词表大小、重复 token 及特殊 token 冲突。提供 segment 模型时，还会验证 `Tensor[1,L] -> Tensor[1,L-1]` 契约与配置的输入范围。输出先写入同级临时目录，全部复制成功后才原子改名；已有输出目录不会被覆盖。

最终目录包含 `config.json`、`bins/pre_model.pte`、`bins/post_model.pte`、`vocabs/` 下的三个词表，以及可选的 `bins/pinyin_segment.pte` 和 `vocabs/pinyin_char_vocab.txt`。`config.json` 将模型包绑定为字符串格式版本 `"2.2"`，并写入 phono-core 所要求的方法名与相对路径。
