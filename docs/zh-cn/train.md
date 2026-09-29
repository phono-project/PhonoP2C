# 训练与模型语义

本文说明当前源码与默认配置，不代表某个已发布 checkpoint 的历史训练参数。复现实验需要训练时解析后的完整配置、数据清单和源码提交。

## 运行入口与环境

在本项目目录使用项目自己的 pixi 环境：

```sh
pixi run python main.py task=train dataset=pretrain_v2
pixi run python main.py task=train --cfg job --resolve
pixi run test
```

当前 `pixi.toml` 仅声明 Linux，并依赖 CUDA 13；不要将其他子项目的 Windows 环境视为等价训练环境。Hydra 组合入口见 [config.yaml](../../config/config.yaml)。多进程训练由 torchrun 和 [分布式辅助代码](../../utils/distributed.py)管理；`task.batchsize` 的解释以 Trainer 的分布式配置检查为准。

`main.py` 在 `system.set_seed=true` 时设置 Python、NumPy 与 Torch 种子；默认是 `false`。直接调用预处理模块的 CLI 不经过该初始化。多进程、设备和算子仍可能影响可复现性，应记录实际入口与解析后的配置。

## 当前默认值

配置文件是权威来源，以下是本次文档修订时的概览：

| 配置 | 当前设置 |
| --- | --- |
| [base.yaml](../../config/model/base.yaml) | 维度 768；pre 为 8 层因果解码器，post 为 4 层双向拼音编码器；自注意力维度 256、4 heads；FFN 维度 2048；最大中文／拼音长度 128／32 |
| [train.yaml](../../config/task/train.yaml) | 7 epochs；batchsize 4096；AdamW；最大学习率 4e-4，最小 4e-5；betas [0.9, 0.95]；weight decay 1e-3；warmup 7500，decay 150000 steps；梯度裁剪 1.0 |
| 损失 | CE；`unconditional_loss_lambda=1.0`；focal 和 label smoothing 是可选配置，并非当前默认 |
| 自由解码验证 | beam width 3；chunk size 64；stride 32，按验证数据顺序每隔若干样本取一条，不等于按来源随机抽样 |
| [system/default.yaml](../../config/system/default.yaml) | CUDA、bf16、float8 线性层训练加速、8-bit 优化器、compile；gradient checkpointing 默认开启，当前 DDP 路径会禁用并提示 |

训练时 float8 加速与部署 W8A8 量化是不同阶段，不能互相替代说明。导出与校准参见 [export.md](export.md)。

## 两遍前向与缓存契约

[pre 模型](../../model/model.py)是中文因果解码器：自注意力后可交叉关注拼音编码器，再经过 FFN；最终 `lm_proj` 输出汉字 logits。[post 模型](../../model/model.py)是双向拼音编码器，返回隐藏状态与逐位置合法字掩码，不输出汉字 logits，也不交叉关注 pre。

第一遍不使用拼音条件，建立中文历史的 self-attention KV，并按配置计算无条件损失。第二遍以历史最后一个 token 作为条件分支的首个查询，随后使用右移的目标字符；decoder 交叉关注整段拼音，逐位置约束输出。两项损失由 [训练包装器](../../model/wrapper.py)组合：

`loss = conditional_loss + unconditional_loss_lambda * unconditional_loss`

将 lambda 设为零会跳过无条件损失的计算与日志，但第二遍所需的无条件历史前向仍然存在。

[Python beam search](../../model/beam_search.py)先预填充 `prefix[:-1]`，再对末 token 进行 pass 2。该末位置产生的条件 self-attention KV 必须保留到本次生成结束；它不能作为以后已提交文本的无条件历史。交叉注意力 KV 来自拼音隐藏状态，与这份条件 self-attention KV 不是同一个张量。运行时提交文本应通过无条件路径重新填充。

## 训练、验证与产物

[Trainer](../../tasks/train.py)加载 tokenizer、训练流与冻结验证集，构建 pre/post 及损失包装器，执行前向、反向、裁剪和调度。分布式路径处理 token 数加权；保存时由主进程分别写出 pre/post 的 `save_pretrained` 产物。

验证中的逐 token ACC、Top-K ACC、S-ACC 和 ECE 使用 teacher-forced 条件 logits；它们不等于自由解码的候选整串命中率。Trainer 另对按 stride 选出的样本运行 beam 解码。发布比较时应明确使用哪一类指标、分母、样本选择、beam 宽度及展示候选数量。

训练产物应附带：解析配置与随机种子、源码提交、数据与划分标识、词表及读音频率哈希、checkpoint 哈希；部署产物另附导出／校准配置、工具版本与二进制构建信息。当前目录中的 YAML 不能替代历史记录。

研究报告与跨项目实验已独立放在集合根目录 `research/`；此处仅维护 PhonoP2C 训练实现的说明。
