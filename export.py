"""
Export PhonoP2CPreModel and PhonoP2CPostModel to ExecuTorch .pte files.
"""

import os
import torch
from torch.export import Dim
from model.model import PhonoP2CPreModel, PhonoP2CPostModel

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass

CHECKPOINT_DIR = "./checkpoints/v1_0-base/final_model"
MODEL_TYPE = torch.float32
SAVE_DIR = "./export_output"


def load_model_from_checkpoint(checkpoint_dir: str, device: torch.device):
    """Load pre and post models from a checkpoint directory.

    The directory should contain:
      pre_model/  — save_pretrained output (config.json + model.safetensors)
      post_model/ — save_pretrained output (config.json + model.safetensors)
    """
    pre_path = os.path.join(checkpoint_dir, "pre_model")
    post_path = os.path.join(checkpoint_dir, "post_model")

    pre_model = PhonoP2CPreModel.from_pretrained(pre_path).to(device)
    post_model = PhonoP2CPostModel.from_pretrained(post_path).to(device)

    pre_model.eval()
    post_model.eval()
    return pre_model, post_model


device = torch.device("cpu")

pre_model, post_model = load_model_from_checkpoint(
    checkpoint_dir=CHECKPOINT_DIR,
    device=device,
)
pre_model.to(MODEL_TYPE)
post_model.to(MODEL_TYPE)

for param in pre_model.parameters():
    param.requires_grad = False
    
for param in post_model.parameters():
    param.requires_grad = False

BATCH_SIZE = 1

pre_cfg = pre_model.config
post_cfg = post_model.config

pre_nheads = pre_cfg.cross_attn_heads  # heads used for the cross-KV projection
pre_head_dim = pre_cfg.mhca_attn_dim // pre_cfg.cross_attn_heads
pre_max = pre_cfg.max_seqlen

post_nheads = post_cfg.mhca_heads
post_head_dim = post_cfg.mhca_attn_dim // post_cfg.mhca_heads

# Self-attention KV cache for pre_model: (num_layers, 2, B, pre_max, nheads, head_dim)
pre_self_nheads = pre_cfg.mhsa_heads
pre_self_head_dim = pre_cfg.attn_dim // pre_cfg.mhsa_heads
dummy_pre_kv_cache = torch.zeros(
    (pre_cfg.mhsa_layers, 2, BATCH_SIZE, pre_max, pre_self_nheads, pre_self_head_dim),
    device=device,
    dtype=MODEL_TYPE,
)

# Shared cross-attention KV cache, written by pre_model and read by post_model:
# (2, B, pre_max, post_nheads, post_head_dim)
dummy_pre_cross_kv_cache = torch.zeros(
    (2, BATCH_SIZE, pre_max, post_nheads, post_head_dim),
    device=device,
    dtype=MODEL_TYPE,
)

dummy_new_prefix_len = 4
dummy_current_seqlen = 8  # tokens already resident in the caches

dummy_post_len = 4
dummy_post_input_ids = torch.randint(
    low=0, high=post_cfg.vocab_size, size=(BATCH_SIZE, dummy_post_len), dtype=torch.long
)
# Total number of pre-encoded prefix tokens visible to cross-attention.
dummy_total_pre = torch.tensor(
    [dummy_current_seqlen + dummy_new_prefix_len], dtype=torch.long, device=device
)


dummy_pre_input_ids = torch.randint(
    low=0, high=pre_cfg.vocab_size, size=(BATCH_SIZE, dummy_new_prefix_len), dtype=torch.long
)
dummy_pre_cache_pos = torch.tensor([dummy_current_seqlen], dtype=torch.long, device=device)

pre_example_kwargs = {
    "input_ids": dummy_pre_input_ids,
    "offsets": None,
    "kv_cache_memory": dummy_pre_kv_cache,
    "current_seqlen": dummy_pre_cache_pos,
    "min_seqlen": None,
    "max_seqlen": None,
    "pre_cross_kv_cache": dummy_pre_cross_kv_cache,
    "pre_cross_cache_pos": dummy_pre_cache_pos,
}

new_len_dim = Dim("new_prefix_len", min=1, max=pre_max)
pre_dynamic_shapes_kwargs = {
    "input_ids": {1: new_len_dim},
    "offsets": None,
    "kv_cache_memory": None,
    "current_seqlen": None,
    "min_seqlen": None,
    "max_seqlen": None,
    "pre_cross_kv_cache": None,
    "pre_cross_cache_pos": None,
}


post_example_kwargs = {
    "input_ids": dummy_post_input_ids,
    "input_offsets": None,
    "pre_K": None,
    "pre_V": None,
    "pre_offsets": None,
    "pre_cross_kv_cache": dummy_pre_cross_kv_cache,
    "current_seqlen": dummy_total_pre,
    "min_seqlen": None,
    "max_seqlen": None,
    "min_seqlen_pre": None,
    "max_seqlen_pre": None,
    "return_last_hidden": False,
}

post_len_dim = Dim("post_len", min=1, max=post_cfg.pre_max_seqlen)
post_dynamic_shapes_kwargs = {
    "input_ids": {1: post_len_dim},
    "input_offsets": None,
    "pre_K": None,
    "pre_V": None,
    "pre_offsets": None,
    "pre_cross_kv_cache": None,
    "current_seqlen": None,
    "min_seqlen": None,
    "max_seqlen": None,
    "min_seqlen_pre": None,
    "max_seqlen_pre": None,
    "return_last_hidden": None,
}

exported_pre = torch.export.export(
    pre_model,
    args=(),
    kwargs=pre_example_kwargs,
    dynamic_shapes=pre_dynamic_shapes_kwargs,
).module()

exported_post = torch.export.export(
    post_model,
    args=(),
    kwargs=post_example_kwargs,
    dynamic_shapes=post_dynamic_shapes_kwargs,
).module()

exported_pre.print_readable()
exported_post.print_readable()

from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

# ExecuTorch XNNPACK backend
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

quantizer = XNNPACKQuantizer()
quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))

prepared_pre_model = prepare_pt2e(exported_pre, quantizer)
prepared_post_model = prepare_pt2e(exported_post, quantizer)

# Dummy forward...
with torch.no_grad():
    prepared_pre_model(**pre_example_kwargs)
    prepared_post_model(**post_example_kwargs)

quantized_pre_model = convert_pt2e(prepared_pre_model)
quantized_post_model = convert_pt2e(prepared_post_model)

quantized_exported_pre_model = torch.export.export(
    quantized_pre_model,
    args=(),
    kwargs=pre_example_kwargs,
    dynamic_shapes=pre_dynamic_shapes_kwargs,
)

quantized_exported_post_model = torch.export.export(
    quantized_post_model,
    args=(),
    kwargs=post_example_kwargs,
    dynamic_shapes=post_dynamic_shapes_kwargs,
)

et_pre_program = to_edge_transform_and_lower(
    quantized_exported_pre_model,
    partitioner=[XnnpackPartitioner()],
).to_executorch(
    ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
    )
)

et_post_program = to_edge_transform_and_lower(
    quantized_exported_post_model,
    partitioner=[XnnpackPartitioner()],
).to_executorch(
    ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
    )
)

os.makedirs(SAVE_DIR, exist_ok=True)
pre_save_path = os.path.join(SAVE_DIR, "pre_model.pte")
with open(pre_save_path, "wb") as f:
    et_pre_program.write_to_file(f)
print(f"Saved pre_model to {pre_save_path}")

post_save_path = os.path.join(SAVE_DIR, "post_model.pte")
with open(post_save_path, "wb") as f:
    et_post_program.write_to_file(f)
print(f"Saved post_model to {post_save_path}")