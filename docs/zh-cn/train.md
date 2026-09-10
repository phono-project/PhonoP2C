# 训练部门（zh-CN）

## 1. 部门概述

训练部门根据 hydra 配置构建两个子模型，并在预处理好的数据集上联合训练。 覆盖入口（`main.py`）、全部配置（`config/`）、训练循环（`tasks/train.py`）、 模型架构（`model/`）、损失函数（`loss.py`）、验证指标（`metrics/`）以及 float8 转换过滤（`utils/float8.py`）。

训练期的数据表示（`dataset.py`）在预处理部门文档中描述；解码参数搜索 （`tasks/param_search.py`）在导出与推理部门文档中描述。

训练流水线的关键特征：

- **联合训练** — 因果 pre 模型编码中文上下文前缀，其输出投影为共享 K/V； 双向 post 模型以拼音为输入，通过对该 K/V 的交叉注意力解码汉字。post 模型掩码后 logits 上的单个损失同时反传到两个模型。
- **NJT 批处理** — 不同长度的样本 collate 成嵌套锯齿张量；训练器把 最小/最大序列长度作为 Python int 传入，使编译后的计算图不包含依赖数据的 控制流。
- **加速栈** — BF16 autocast、可选 torchao float8 线性层、可选 `torch.compile`、CUDA expandable segments、TF32。
- **优化** — AdamW / AdEMAMix（bitsandbytes，8 位或 32 位，可 paged）， 余弦调度与 warmup，weight decay 分组，梯度裁剪。
- **验证与日志** — 周期性验证计算 loss、ACC、Top-3/Top-5 ACC、 Sentence-ACC 与 adaptive ECE；wandb 日志；rich 进度条；按轮 `save_pretrained` checkpoint 及最终 `final_model` checkpoint。

## 2. `main.py` — 入口

**功能：** hydra 装饰的入口 （`@hydra.main(version_base=None, config_path="config", config_name="config")`）。 加载完整配置，应用全局运行时设置，并分发到指定的任务运行器。

**用法：** `python main.py`（训练，按 `config/task/train.yaml`），或 `python main.py task=param_search`（并附加任务级覆盖，见 `export.md`）。

### 单机多卡 DDP

训练使用原生 PyTorch DDP，由 `torchrun` 启动：

```bash
pixi run torchrun --standalone --nproc-per-node=4 main.py
```

`task.batchsize` 是每张卡的 batch size，因此全局 batch size 为
`task.batchsize * world_size`。Mosaic `StreamingDataset` 根据 torchrun 环境变量
自行切分训练数据，不应额外添加 `DistributedSampler`。验证集由不补齐、不重复的
sampler 交错分片，各 rank 同时执行前向与 beam search，最后汇总 token 加权 loss、
ACC/Top-k/S-ACC、beam sentence accuracy 和 adaptive ECE。W&B、进度条与 checkpoint
只由 rank 0 写入。

训练 loss 也按所有 rank 的全局有效 token 数加权，因此变长 NJT 批次的 DDP 梯度
与把各 rank 样本合并成一个全局 batch 的单进程梯度具有相同的归约语义。当前使用的
PyTorch 2.13 中，DDP 与 NJT activation checkpoint 重算组合会为同一 ragged 维度
生成不同的符号 ID；训练器因此仅在 DDP 模式关闭 gradient checkpointing。该兼容
处理不修改模型、attention、offsets 或 loss，只增加 activation 显存并省去重算开销；
单进程仍保持配置指定的 checkpointing 行为。

**行为。**

- 设置 `WANDB_CONSOLE=off`；`system.expandable_segments_enabled` 时启用 CUDA `expandable_segments` 分配器。
- `output.logging.log_level` 不是 `DEFAULT` 时应用全局 `logging.basicConfig`（并同步 `hydra`、`torch` 两个 logger 的级别）。
- `system.tf32_enabled` 时启用 matmul 与 cudnn 的 TF32。
- `system.set_seed` 时可选地固定 CPU/GPU 随机种子（默认 42）。
- 设置 `torch.set_float32_matmul_precision("high")` 并关闭 inductor FX graph cache（NJT 兼容性）。
- 分发：`task_type == "param_search"` -> `ParamSearchRunner`；其余 -> `Trainer.train()`。

## 3. `config/` — hydra 配置

**功能：** 全部运行时配置。`config.yaml` 组合以下配置组：`model: base`、 `dataset: pretrain_v1`、`system: default`、`task: train`、 `logging: train`、`output: default`。

### `config/model/base.yaml` — 架构
- `common`：`model_dim: 768`、`rope_theta: 1000.0`。
- `pre_model`：`max_seqlen: 128`、`mhsa_layers: 8`、`mhsa_heads: 4`、 `attn_dim: 256`、`ffn_common_dim: 4096`。
- `post_model`：`max_seqlen: 32`、`mhsa_layers: 12`、`mhsa_heads: 4`、 `attn_dim: 256`、`mhca_heads: 12`、`mhca_attn_dim: 768`、 `ffn_common_dim: 4096`。

### `config/dataset/pretrain_v1.yaml` — 数据
- 路径：`dataset_dir`、`vocabs_config`、`train_dir_mds` （`${dataset.dataset_dir}/train`）、`val_dir` （`${dataset.dataset_dir}/val`）。
- `augmentation`：`drop_vowels: 0.2`、`drop_last_vowel: 0.3`、 `vowels_droprate: [0.5, 1.0]`、`heteronym_confusion: 0.1`。
- `online_policy`：`strategy: "golden-sequence"`、`no_context_prob: 0.2`、 `forward_ratio: 0.0`（注意：变换读取的是 `forward_prob`，因此 forward/backward 混合比例实际回退到默认值 0.5）。
- `shuffle`：MDS 流式打乱（`algo: py1e`、`blocksize: 200000000000`、 `cache_limit: 140gb`）。

### `config/system/default.yaml` — 运行时
`seed: 'no'`、`set_seed: false`、`num_workers: 4`、`keep_in_memory: true`、 `tf32_enabled: true`、`mixed_precision: 'bf16'`、 `ao_acceleration: 'float8'`、`optim_8bit: true`、`optim_paged: false`、 `device: 'cuda'`、`expandable_segments_enabled: true`。

### `config/task/train.yaml` — 训练超参数
`epochs: 15`、`batchsize: 1280`、`optimizer: "adamw"`、 `learning_rate: 2e-4`、`weight_decay: 1e-2`、`warmup_steps: 10000`、 `gradient_clip_val: 1.0`、`compile_model: true`、`compile_mode: 'default'`、 `loss_type: "focal"`、`focal_loss_gamma: 1.0`、`focal_loss_alpha: 0.5`、 `ece_bins: 15`、`ece_top_k: 5`、`task_type: "train"`。

### `config/logging/train.yaml` — 日志与 checkpoint
`log_with_wandb: true`、`project_name: "PhonoP2C"`、`run_name: ...`、 `checkpoint_dir: ./checkpoints/<run>`、`histogram_interval: 10000`、 `val_interval: 5000`、`save_interval: 1`（轮）。

### `config/output/default.yaml` — 输出行为
- `progress_bar`：条宽、刷新率、已用/剩余时间、速度、比例列、 `transient: true`。
- `logging.log_level: "ERROR"`（或 `DEFAULT`）。

### `config/task/param_search.yaml` — 解码校准
见 `export.md`（由 `main.py task=param_search` 使用）。

## 4. `tasks/train.py` — Trainer

**功能：** 两个子模型的联合训练循环与验证流程，基于 NJT 批次。

### 进度条支持

#### `IterationSpeedColumn(ProgressColumn)`
- 功能：渲染 rich 任务的每秒迭代数。
- 行为：速度未知时显示 `? it/s`，否则显示 `X.XX it/s`。

#### `MofNCompleteColumn(ProgressColumn)`
- 功能：渲染任务的 `completed/total`。
- 行为：总量未知时显示 `?`。

#### `_build_progress(cfg)`
- 功能：按 `cfg.output.progress_bar` 构建 rich `Progress` 对象。
- 行为：组装列（描述、条、百分比，可选已用/剩余时间、比例、速度），刷新 率与 transient 模式可配置；始终追加任务 postfix 字段。

### 配置辅助

#### `build_model_configs(cfg, tokenizer)`
- 功能：按 `model.*` YAML 构建 `PreModelConfig` 与 `PostModelConfig`，并用 分词器的词表大小补齐。
- 行为：把 `cfg.model` 解析为字典；把 `context` / `pinyin` / `chinese` 词表大小传入 `build_configs_from_dict`。

### `Trainer`

#### `__init__(cfg)`
- 功能：构建全部训练组件。
- 行为（按序）：设置设备与日志；构建分词器；构建两个模型配置；在设备上 构建拼音->汉字可能性掩码并安装到 `post_model.logits_mask`；实例化两个模型 并打印参数量；可选地把 Linear 层转换为 float8 训练（torchao、 `pad_inner_dim=True`、`module_filter_fn`）；可选地 `torch.compile` 两个 模型（`dynamic=True`，dynamo 配置 `capture_scalar_outputs` / `capture_dynamic_output_shape_ops`）；创建 checkpoint 目录并（可选）初始化 wandb + `wandb.watch`（直方图日志）；构建在线策略/增强配置字典；构建 MDS 流式训练集与 `StreamingDataLoader`（打乱算法/块大小/cache limit、NJT collate）；计算 `epoch_steps` / `total_steps`；按 `warmup_steps` 或 `warmup_ratio` 计算 warmup 步数；构建验证集 （`transform_pinyin_predict_val`）与 `DataLoader`（pin_memory、NJT collate）；把参数分为 decay / no-decay 两组（decay：`nn.Linear` 权重； no-decay：bias、LayerNorm/RMSNorm/Embedding 权重；断言每个参数恰好分到 一组）；构建优化器（AdamW 或 AdEMAMix，8/32 位，可 paged）与余弦调度器； 经 `get_loss_fn` 构建损失；按 `system.mixed_precision == "bf16"` 设置 `use_amp`。

#### `validate(pre_model, post_model, loader, epoch, global_step, progress)`
- 功能：在验证集上评估两个模型。
- 行为：切换为 eval 模式；每批把 NJT 移到设备、取扁平 id/offsets、以 Python int 计算最小/最大序列长度、在 bf16 autocast 下跑 pre -> post， 在扁平掩码 logits 上计算损失；累加损失与指标（`MetricsAccumulator`）； 结束后计算平均验证损失与困惑度（loss < 20 时 `exp(loss)`，否则 `inf`）， 并向 wandb 记录 `val/loss`、`val/ACC`、`val/Top3-ACC`、`val/Top5-ACC`、 `val/S-ACC`、`val/ECE`（CE 损失时另加 `val/PPL`），步号为 global step； 恢复训练模式。

#### `train()`
- 功能：主训练循环。
- 行为：按轮迭代；每批：把 NJT 移到设备、取扁平 values/offsets、计算最小/ 最大序列长度、梯度清零、在 autocast 下前向（pre 模型输入扁平前缀 id + offsets -> pre_embed/pre_K/pre_V；post 模型输入扁平 postfix id + offsets， 附带 `pre_K/pre_V/pre_offsets` 与两组 min/max 边界 -> NJT logits）、在扁平 掩码 logits 上计算损失、反向传播、裁剪梯度范数（`gradient_clip_val > 0` 时）、优化器与调度器各走一步；每步向 wandb 记录 train/loss、LR、 grad_norm；每 `val_interval` 步以及最后一步验证；每 `save_interval` 轮保存 checkpoint（`epoch_<n>`/pre_model 与 post_model，经 `save_pretrained`， 编译模型需解包 `_orig_mod`）；全部轮次结束后保存 `final_model`；结束 wandb。

## 5. `model/` — 模型架构

### `config.py`

#### `PreModelConfig(PretrainedConfig)`
- 功能：pre 模型的配置类（`model_type = "phono_p2c_pre"`）。
- 字段：`model_dim`（768）、`attn_dim`、`rope_theta`、`max_seqlen`（128）、 `mhsa_layers`（8）、`mhsa_heads`（4）、`ffn_common_dim`（4096）、 `vocab_size`（context 词表大小）、`mhca_attn_dim`、 `cross_attn_heads`（共享 K/V 投影使用的 post 侧头数）。

#### `PostModelConfig(PretrainedConfig)`
- 功能：post 模型的配置类（`model_type = "phono_p2c_post"`）。
- 字段：`model_dim`、`attn_dim`、`rope_theta`、`pre_max_seqlen`（pre 的 max_seqlen，限定交叉注意力范围）、`max_seqlen`（32）、`mhsa_layers` （12）、`mhsa_heads`（4）、`use_moe_ffn`、`ffn_common_dim`（4096）、 `ffn_num_experts`、`ffn_choice`、`ffn_expert_dim`、`vocab_size` （pinyin 词表大小）、`proj_size`（chinese 词表大小）、`mhca_heads` （12）、`mhca_attn_dim`（768）。

#### `build_configs_from_dict(d, vocab_sizes)`
- 功能：由嵌套 YAML 字典加词表大小（`context`、`pinyin`、`chinese`）构建 两个配置。
- 行为：读取 `common` / `pre_model` / `post_model` 各节（带默认值）；接线 `pre.vocab_size = vocab_sizes["context"]`、 `post.vocab_size = vocab_sizes["pinyin"]`、 `post.proj_size = vocab_sizes["chinese"]`；pre 模型的 `mhca_attn_dim` 与 `cross_attn_heads` 取自 post 侧设置。

### `model.py`

#### `PhonoP2CPreModel(PreTrainedModel)`
- 功能：中文上下文前缀的因果编码器。
- 结构：嵌入（`vocab_size -> model_dim`）、`mhsa_layers` 层 `{mhsa, ffn(SwiGLU), norm1, norm2}`、末尾 RMSNorm，以及共享 `kv_proj`（`model_dim -> 2·mhca_attn_dim`），其输出按最后一维拆分为交叉 注意力的 K 和 V。
- `forward(input_ids, offsets=None, kv_cache_memory=None, current_seqlen=None, min_seqlen=None, max_seqlen=None, pre_cross_kv_cache=None, pre_cross_cache_pos=None)` — 三条路径：
- NJT 路径（给 offsets）：扁平嵌入，经 `make_local_position_ids` 计算逐 token 的 RoPE 位置；每层 MHSA 走 NJT 路径（因果）；返回 `(hidden, pre_K, pre_V)`，形状为 `[total_tokens, cross_attn_heads, head_dim]`。
- 带缓存批处理路径（给 `kv_cache_memory` + `current_seqlen`）：每层 MHSA 读写完整自注意力 KV Cache（`update_mhsa_kv`）；`pre_K/pre_V` 按 头重排；给出 `pre_cross_kv_cache` 与 `pre_cross_cache_pos` 时，K/V 经 `update_cross_kv` 就地写入共享交叉缓存并直接返回更新后的缓存（推理/ 导出使用）。
- 普通批处理路径：稠密 SDPA，返回 `(hidden, pre_K, pre_V)`。

#### `PhonoP2CPostModel(PreTrainedModel)`
- 功能：双向的拼音->汉字解码器。
- 结构：嵌入（`pinyin 词表 -> model_dim`）、`mhsa_layers` 层 `{mhsa（双向）, mhca（对 pre K/V 的交叉注意力）, norm1..3, ffn （SwiGLU 或 MoE_EC_FFN）}`、末尾 RMSNorm、`lm_head` （`model_dim -> proj_size`），以及注册的 `logits_mask` buffer（拼音->汉字 可能性掩码）。
- `forward(input_ids, input_offsets=None, pre_K=None, pre_V=None, pre_offsets=None, pre_cross_kv_cache=None, current_seqlen=None, min_seqlen=None, max_seqlen=None, min_seqlen_pre=None, max_seqlen_pre=None, return_last_hidden=False)` — 两条路径：
- NJT 路径（给 `input_offsets`）：逐 token 自注意力位置；交叉注意力位置 由 `MHCALayer.compute_position_ids` 计算；每层依次为双向 MHSA、对 `pre_K/pre_V` 的 MHCA（查询与 KV 各自独立的 min/max 边界）、FFN（MoE 需要 offsets）；末尾归一化；扁平隐藏态过 `lm_head`；按输入 token id 索引 `logits_mask` 应用逐位置掩码（非法类别 -> `-inf`）；扁平 logits 再 包回 NJT（`return_last_hidden` 时连同 hidden 一起返回）。
- 批处理路径（无 offsets）：层布局相同；使用缓存时 MHCA 以 `current_seqlen` 作为 pre 上下文总长度读取共享交叉缓存；返回稠密 `[B, S, proj_size]` logits（已掩码）或 `(logits, hidden)`。

### `attn.py`

#### `MHSALayer(nn.Module)`
- 功能：带 RoPE 的多头自注意力；支持 NJT、带缓存批处理、普通批处理三条 路径。
- 结构：`qkv_proj`（`model_dim -> 3·attn_dim`，无 bias）、`out_proj`、 旋转嵌入。
- `forward(...)` 行为：
- NJT 路径：投影扁平隐藏态、按头重排、用局部位置 id 应用 RoPE、构建 q/k/v NJT（给定了 min/max 序列长度则直接使用，否则从 offsets 推导）、 运行 `scaled_dot_product_attention`（因果或非因果）、投影回。
- 带缓存批处理路径：计算新块的 q/k/v，把 k/v 写入按层完整的缓存 （`update_mhsa_kv`），切出有效 KV 窗口，对 q（新位置）与整个有效 K 窗口应用 RoPE，构建因果掩码（新位置可注意自身及之前的全部缓存位置）， 用该掩码运行 SDPA。
- 普通批处理路径：对 `arange(S)` 应用 RoPE，标准因果 SDPA。

#### `MHCALayer(nn.Module)`
- 功能：多头交叉注意力：查询来自 post 模型，K/V 来自 pre 模型的共享投影。
- 结构：`q_proj`（`model_dim -> mhca_attn_dim`）、`out_proj`、旋转嵌入。
- `compute_position_ids(offsets, pre_offsets)`（静态）：计算查询位置 id （post 局部位置 + 每行 pre 前缀长度）与 KV 位置 id（pre 局部位置），供 NJT 路径使用。
- `forward(...)` 行为：
- NJT 路径：投影查询、用计算的交叉位置应用 RoPE、对 pre K/V 应用 RoPE、 以查询与 KV 各自独立的 min/max 边界构建 q/k/v NJT、运行非因果 SDPA。
- 批处理路径：读取共享 `pre_kv_cache` 元组中 `total_kv = cache_pos[0]` 之前的部分（K 以未加 RoPE 的形式存储），对整个有效 K 窗口与新查询位置 应用 RoPE，运行非因果 SDPA。

### `ffn.py`

#### `SwiGLU(nn.Module)`
- 功能：SwiGLU 前馈块。
- 结构：`up_proj`、`gate_proj`（`in -> hidden`，无 bias）、`down_proj` （`hidden -> out`，无 bias）。
- 行为：`down(silu(gate(x)) * up(x))`。

### `moe.py`

#### `MoE_EC_FFN(nn.Module)`
- 功能：expert-choice MoE 前馈，含一个常开的共享专家与 N 个被路由专家 （全部专家合并为一次批矩阵乘）。
- 结构：`affinity` 门控（`dim -> num_experts`）、共享 `expert_common` SwiGLU、堆叠专家权重 `experts_up_proj` / `experts_gate_proj` （`[num_experts, dim, expert_dim]`）与 `experts_down_proj` （`[num_experts, expert_dim, dim]`），xavier 初始化。
- `forward(hidden, offsets=None)` 行为：把输入压平为 `[total_tokens, dim]`；专家容量 `max(1, total·choice / num_experts)`； 门控分数 -> 每个专家对 token 做 `topk`（expert choice）； `_batched_swiglu` 用一次 `bmm` 计算全部专家输出；合并权重为 `sigmoid(topk_vals)`；路由结果经 `index_add`（就地不可，使用拷贝）按 token 位置加回，并与共享专家输出相加；返回扁平（NJT）或 `[B, S, dim]` 输出。

### `utils.py`

#### `RotaryEmbedding(nn.Module)`
- 功能：RoPE 频率表（`inv_freq = theta^(-2i/dim)`）。
- 行为：`forward(position_ids)` 返回形状为 `[T, dim]` 的 `(cos, sin)` （频率沿最后一维复制）。

#### `rotate_half(x)`
- 功能：最后一维的半数旋转（RoPE 辅助）。

#### `apply_rotary_pos_emb(v, cos, sin)`
- 功能：对 `[..., seq, heads, head_dim]` 张量应用 RoPE。
- 行为：`v·cos + rotate_half(v)·sin`，cos/sin 沿批次与头广播。

#### `make_local_position_ids(offsets)`
- 功能：把 NJT offsets 转换为逐 token 的局部位置。
- 行为：全局 id 减去每行起始位置（`repeat_interleave` offsets[:-1]）。

### `custom_ops.py`
- 功能：在 `phono` 命名空间注册自定义 `torch.library` 算子，以函数式语义 （clone + `index_copy_`）实现 KV Cache 就地更新，使 torch.export / ExecuTorch 可以无别名问题地捕获缓存更新。
- 算子：`update_kv_cache(cache, value, start_pos)`（通用，沿第 1 维 `index_copy_`）、`update_cross_kv(cache, pre_K, pre_V, start_pos)` （K 与 V 写入 `[2, B, S, ...]` 缓存）、`update_mhsa_kv(cache, k, v, start_pos, layer_idx)`（按层的自注意力缓存）。每个算子都有函数式与 `.out` 两个变体，含 CPU 实现与 `fake`（meta）实现。
- 行为：包装函数（`kv_cache_write`、`update_cross_kv`、`update_mhsa_kv`） 把张量形式的 `start_pos` 经 `.item()` 解包。`kv_cache_write` 在 `model/__init__.py` 中导出；当前模型前向路径直接使用 `update_cross_kv` / `update_mhsa_kv`。

### `__init__.py`
- 功能：重新导出配置、模型、层、RoPE 辅助与 `kv_cache_write` 作为包的公开 API。

## 6. `loss.py` — 损失函数

### `FocalLoss(nn.Module)`
- 功能：面向类别不均衡的 focal loss： `FL = -alpha · (1 - p_t)^gamma · log(p_t)`。
- 用法：`loss_type="focal"` 时选用；由 `focal_loss_alpha` / `focal_loss_gamma` 配置。
- 行为：计算逐 token 交叉熵（忽略 `ignore_index` 目标），由 `p_t = exp(-ce)` 得到聚焦权重，掩掉被忽略位置，按均值（对有效 token） 或求和归约。

### `LabelSmoothingCrossEntropy(nn.Module)`
- 功能：感知模型 `logits_mask` 的 label smoothing。
- 用法：`loss_type="ce"` 且设置了 `label_smoothing_epsilon` 时选用。
- 行为：只在 logits 有限（即掩码允许）的类别上平滑——被掩掉的类别为 `-inf`，永远不会分到概率质量；目标类占 `1 - epsilon`，其余 `epsilon` 均匀分摊到其他有效类别；目标类是该行唯一有效类别的行退化为普通交叉熵 （epsilon 视为 0，避免除零）；掩掉被忽略行；按均值或求和归约。

### `get_loss_fn(loss_type="ce", ignore_index=-100, focal_loss_alpha=0.25, focal_loss_gamma=2.0, label_smoothing_epsilon=None)`
- 功能：损失工厂。
- 行为：`"ce"` -> `nn.CrossEntropyLoss`（设置了 `label_smoothing_epsilon` 时返回 `LabelSmoothingCrossEntropy`）；`"focal"` -> `FocalLoss`；其他 取值抛出 `ValueError`。

## 7. `metrics/accumulator.py` — MetricsAccumulator

**功能：** 验证期运行指标累加器：逐 token top-1 ACC、top-3 / top-5 ACC、 句子级 ACC，以及 Top-K adaptive ECE（经 `netcal.metrics.confidence.ACE` 的分位数分箱）。

**用法：** `acc = MetricsAccumulator(ece_bins=15, ece_top_k=5)`；每批 `acc.update(logits, targets, target_offsets)`；`acc.compute()` 返回指标 字典；`acc.reset()`。

#### `__init__(ece_bins=15, ece_top_k=1)`
- 功能：配置分箱数与 ECE 的 confidence/correct 定义所用 k。

#### `update(logits, targets, target_offsets)`
- 功能：摄入一批扁平 logits、扁平 targets 与句子 offsets。
- 行为：计算 softmax 概率；累加 top-1/top-3/top-5 的 token 数与正确数； 累加句子级正确性（一句内所有 token 都正确）；为 ECE 累加逐 token 的 top-k confidence 与 correct 数组（float64 CPU）。空批次直接忽略。

#### `compute()`
- 功能：汇总全部指标。
- 行为：返回 `{acc, top3_acc, top5_acc, s_acc, ece}`；ECE 用 ACE detector 指标在累加的 confidence / correct 数组上计算（无数据时为 0.0）。

#### `reset()`
- 功能：清空全部运行统计。

#### `__len__()`
- 功能：已累加的 token 数。

## 8. `utils/float8.py` — Float8 转换过滤

#### `module_filter_fn(mod, fqn)`
- 功能：决定 torchao `convert_to_float8_training` 转换哪些模块。
- 行为：只对全限定名包含 `expert`、`qkv_proj`、`out_proj`、`up_proj`、 `gate_proj`、`down_proj`、`q_proj` 或 `kv_proj` 的 `nn.Linear` 返回 True；其余模块（norm、embedding、门控等）保持原精度。
