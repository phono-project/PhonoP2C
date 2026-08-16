"""
Export PhonoP2CPreModel and PhonoP2CPostModel to ExecuTorch .pte files.

The new-standard architecture exports three graphs:

  * pre_model_pass1.pte — the decoder's unconditional pass (no cross
    attention, no logits mask): updates the self-attn KV cache in place.
  * pre_model_pass2.pte — the decoder's conditional pass: self-attention over
    the cache (updated in place), cross-attention over the pinyin encoder
    hidden states, masked logits.
  * post_model.pte      — the pinyin encoder: hidden states + logits mask.
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


def quantize_and_lower(exported_module, example_kwargs, dynamic_shapes):
    """XNNPACK dynamic per-channel quantization + edge lowering."""
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))

    prepared = prepare_pt2e(exported_module, quantizer)
    with torch.no_grad():
        prepared(**example_kwargs)
    quantized = convert_pt2e(prepared)

    quantized_exported = torch.export.export(
        quantized,
        args=(),
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )

    return to_edge_transform_and_lower(
        quantized_exported,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )


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

# Self-attention KV cache for pre_model:
# (num_layers, 2, B, pre_max, nheads, head_dim)
pre_self_nheads = pre_cfg.mhsa_heads
pre_self_head_dim = pre_cfg.attn_dim // pre_cfg.mhsa_heads
dummy_pre_kv_cache = torch.zeros(
    (pre_cfg.mhsa_layers, 2, BATCH_SIZE, pre_cfg.max_seqlen, pre_self_nheads, pre_self_head_dim),
    device=device,
    dtype=MODEL_TYPE,
)

dummy_new_prefix_len = 4
dummy_current_seqlen = 8  # tokens already resident in the caches
dummy_post_len = 4

dummy_pre_input_ids = torch.randint(
    low=0, high=pre_cfg.vocab_size, size=(BATCH_SIZE, dummy_new_prefix_len), dtype=torch.long
)
dummy_pre_cache_pos = torch.tensor([dummy_current_seqlen], dtype=torch.long, device=device)

# ---------------------------------------------------------------- pre pass 1
pre1_example_kwargs = {
    "input_ids": dummy_pre_input_ids,
    "kv_cache_memory": dummy_pre_kv_cache,
    "current_seqlen": dummy_pre_cache_pos,
    "use_custom_ops": True,
}
new_len_dim = Dim("new_prefix_len", min=1, max=pre_cfg.max_seqlen)
pre1_dynamic_shapes = {
    "input_ids": {1: new_len_dim},
    "kv_cache_memory": None,
    "current_seqlen": None,
    "use_custom_ops": None,
}
exported_pre1 = torch.export.export(
    pre_model,
    args=(),
    kwargs=pre1_example_kwargs,
    dynamic_shapes=pre1_dynamic_shapes,
).module()

# ---------------------------------------------------------------- post model
dummy_post_input_ids = torch.randint(
    low=0, high=post_cfg.vocab_size, size=(BATCH_SIZE, dummy_post_len), dtype=torch.long
)
post_example_kwargs = {"input_ids": dummy_post_input_ids}
post_len_dim = Dim("post_len", min=1, max=post_cfg.max_seqlen)
post_dynamic_shapes = {"input_ids": {1: post_len_dim}}
exported_post = torch.export.export(
    post_model,
    args=(),
    kwargs=post_example_kwargs,
    dynamic_shapes=post_dynamic_shapes,
).module()

# ---------------------------------------------------------------- pre pass 2
dummy_post_hidden = torch.zeros(
    (BATCH_SIZE, dummy_post_len, post_cfg.model_dim), device=device, dtype=MODEL_TYPE
)
dummy_logits_mask = torch.ones(
    (BATCH_SIZE, dummy_post_len, pre_cfg.proj_size), dtype=torch.bool
)
# Full decoder length: prefix tokens already cached + this chunk + pinyin len
dummy_post_position_offset = dummy_current_seqlen + dummy_new_prefix_len + dummy_post_len

pre2_example_kwargs = {
    "input_ids": dummy_pre_input_ids,
    "kv_cache_memory": dummy_pre_kv_cache,
    "current_seqlen": dummy_pre_cache_pos,
    "post_hidden": dummy_post_hidden,
    "post_position_offset": dummy_post_position_offset,
    "logits_mask": dummy_logits_mask,
    "use_custom_ops": True,
}
pre2_dynamic_shapes = {
    "input_ids": {1: Dim("chunk_len", min=1, max=pre_cfg.max_seqlen)},
    "kv_cache_memory": None,
    "current_seqlen": None,
    "post_hidden": {1: Dim("post_len2", min=1, max=post_cfg.max_seqlen)},
    "post_position_offset": None,
    "logits_mask": None,
    "use_custom_ops": None,
}
exported_pre2 = torch.export.export(
    pre_model,
    args=(),
    kwargs=pre2_example_kwargs,
    dynamic_shapes=pre2_dynamic_shapes,
).module()

exported_pre1.print_readable()
exported_pre2.print_readable()
exported_post.print_readable()

os.makedirs(SAVE_DIR, exist_ok=True)

programs = {
    "pre_model_pass1.pte": quantize_and_lower(exported_pre1, pre1_example_kwargs, pre1_dynamic_shapes),
    "pre_model_pass2.pte": quantize_and_lower(exported_pre2, pre2_example_kwargs, pre2_dynamic_shapes),
    "post_model.pte": quantize_and_lower(exported_post, post_example_kwargs, post_dynamic_shapes),
}

for name, program in programs.items():
    save_path = os.path.join(SAVE_DIR, name)
    with open(save_path, "wb") as f:
        program.write_to_file(f)
    print(f"Saved {name} to {save_path}")
