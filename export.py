"""
Export PhonoP2CPreModel and PhonoP2CPostModel to ExecuTorch .pte files.

The new-standard architecture exports two programs:

  * pre_model.pte  — a multi-method program containing both decoder passes.
    Keeping the methods together lets ExecuTorch deduplicate their shared
    pre-model parameters.
  * post_model.pte — the pinyin encoder: hidden states + logits mask.
"""

import os
import torch
from torch.export import Dim
from model.model import PhonoP2CPreModel, PhonoP2CPostModel

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.utils.fuser_utils import validate_partition

CHECKPOINT_DIR = "./checkpoints/v2_0-base-alpha05/final_model"
MODEL_TYPE = torch.float32
SAVE_DIR = "./export_output"
MODEL_VERSION = "v2_0-base-alpha05"
MODEL_FORMAT_VERSION = 2
BEAM_SIZE = 3


class _ContiguousXnnpackPartitioner(XnnpackPartitioner):
    """Merge adjacent config matches without GroupBasedPartitioner's cross-layer fusion."""

    def generate_partitions(self, ep):
        graph_nodes = list(ep.graph_module.graph.nodes)
        order = {node: index for index, node in enumerate(graph_nodes)}
        atomic = [list(part.nodes) for part in self.generate_per_op_partitions(ep)]
        atomic.sort(
            key=lambda nodes: min(
                order[node] for node in nodes if node.op == "call_function"
            )
        )

        runs = []
        for nodes in atomic:
            if not runs:
                runs.append(nodes)
                continue
            current_compute = [node for node in runs[-1] if node.op == "call_function"]
            next_compute = [node for node in nodes if node.op == "call_function"]
            boundary = graph_nodes[
                max(order[node] for node in current_compute)
                + 1 : min(order[node] for node in next_compute)
            ]
            candidate = runs[-1] + nodes
            if not any(
                node.op == "call_function" for node in boundary
            ) and validate_partition(candidate):
                runs[-1] = candidate
            else:
                runs.append(nodes)
        return [Partition(id=index, nodes=nodes) for index, nodes in enumerate(runs)]


# Where the per-model + merged ExecuTorch selective-build manifests are written
# after export. ExecuTorch's own gen_oplist can only derive an operator list
# from a single .pte file, so the manifests are generated here, spanning all
# exported programs. Copy them into phono-core/ops_config/ before building
# phono-core to prune its kernel library to exactly the operators and dtypes
# the models use (see phono-core/CMakeLists.txt).
MANIFEST_DIR = os.path.join(SAVE_DIR, "manifests")
MANIFEST_TAG = MODEL_VERSION


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


def quantize_exported(exported_module, example_kwargs, dynamic_shapes):
    """Quantize one exported graph and return it as an ExportedProgram."""
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True)
    )

    prepared = prepare_pt2e(exported_module, quantizer)
    # torchao dynamic quantization still needs a representative forward to
    # initialize the prepared graph before convert_pt2e is called.
    with torch.no_grad():
        prepared(**example_kwargs)
    quantized = convert_pt2e(prepared)

    quantized_exported = torch.export.export(
        quantized,
        args=(),
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
    )

    return quantized_exported


def lower_and_save(programs, save_path):
    """Lower one or more named graphs into a single ExecuTorch program."""
    # Lower independently before EdgeProgramManager composes the multi-method
    # PTE, preserving shared-constant deduplication.
    lowered_methods = {}
    for name, program in programs.items():
        partitioner = (
            _ContiguousXnnpackPartitioner()
            if name.startswith("pre_model_")
            else XnnpackPartitioner()
        )
        single = to_edge_transform_and_lower(
            program,
            partitioner=[partitioner],
        )
        lowered_methods[name] = next(iter(single._edge_programs.values()))
    edge_program = EdgeProgramManager(lowered_methods)
    executorch_program = edge_program.to_executorch(
        ExecutorchBackendConfig(
            memory_planning_pass=MemoryPlanningPass(alloc_graph_input=False),
        )
    )

    with open(save_path, "wb") as f:
        executorch_program.write_to_file(f)
    print(f"Saved {os.path.basename(save_path)} to {save_path}")


def _collect_ops_metadata(pte_path):
    """Return (sorted unique op names, {op: [kernel keys]}) for one .pte file.

    The kernel keys carry the exact dtype/dim-order variants the model invokes,
    which is what ExecuTorch's dtype-selective build consumes.
    """
    from executorch.codegen.tools.gen_oplist import (
        _get_kernel_metadata_for_model,
        _get_operators,
    )

    ops = sorted(set(_get_operators(pte_path)))
    metadata = _get_kernel_metadata_for_model(pte_path)
    return ops, metadata


def _dump_ops_manifest(ops, metadata, model_name, path):
    """Write one manifest in ExecuTorch's selected_operators.yaml format."""
    from torchgen.selective_build.operator import SelectiveBuildOperator

    import yaml

    operators = {}
    for op_name in ops:
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": True,
                "is_used_for_training": True,
                "include_all_overloads": False,
                "debug_info": [model_name],
            },
        )
        operators[op_name] = op.to_dict()

    output = {
        "operators": operators,
        "custom_classes": [],
        "build_features": [],
        "include_all_non_op_selectives": False,
        "include_all_operators": False,
        "kernel_metadata": {},
        "et_kernel_metadata": metadata,
    }
    with open(path, "wb") as f:
        f.write(yaml.safe_dump(output, default_flow_style=False).encode("utf-8"))


def generate_ops_manifests():
    """Generate per-model + merged ExecuTorch selective-build manifests.

    phono-core deploys two programs (pre_model.pte + post_model.pte) on one
    runtime, but ExecuTorch's upstream tooling can only build an operator list
    from a single .pte model. This reads the serialized programs and writes:

      * <MANIFEST_TAG>_pre_ops.yaml / <MANIFEST_TAG>_post_ops.yaml — per-model
        manifests (operators + dtype/dim-order kernel metadata), and
      * <MANIFEST_TAG>_ops.yaml — their union, for a combined build.

    Copy whichever you need into phono-core/ops_config/ before building
    phono-core.
    """
    os.makedirs(MANIFEST_DIR, exist_ok=True)

    per_model = {}
    for tag, pte_name in (("pre", "pre_model.pte"), ("post", "post_model.pte")):
        pte_path = os.path.join(SAVE_DIR, pte_name)
        ops, metadata = _collect_ops_metadata(pte_path)
        per_model[tag] = (ops, metadata)
        out = os.path.join(MANIFEST_DIR, f"{MANIFEST_TAG}_{tag}_ops.yaml")
        _dump_ops_manifest(ops, metadata, f"{MANIFEST_TAG}-{tag}", out)
        print(f"Manifest {tag}: {len(ops)} operators -> {out}")

    all_ops = sorted(set().union(*(ops for ops, _ in per_model.values())))
    merged_metadata = {}
    for _, metadata in per_model.values():
        for op, keys in metadata.items():
            seen = merged_metadata.setdefault(op, [])
            for key in keys:
                if key not in seen:
                    seen.append(key)
    merged_out = os.path.join(MANIFEST_DIR, f"{MANIFEST_TAG}_ops.yaml")
    _dump_ops_manifest(all_ops, merged_metadata, MANIFEST_TAG, merged_out)
    print(f"Manifest merged: {len(all_ops)} operators -> {merged_out}")


if __name__ == "__main__":
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

    pre_cfg = pre_model.config
    post_cfg = post_model.config

    # Self-attention KV cache for pre_model:
    # (num_layers, 2, B, pre_max, nheads, head_dim)
    pre_self_nheads = pre_cfg.mhsa_heads
    pre_self_head_dim = pre_cfg.attn_dim // pre_cfg.mhsa_heads
    dummy_pre1_kv_cache = torch.zeros(
        (
            pre_cfg.mhsa_layers,
            2,
            1,
            pre_cfg.max_seqlen,
            pre_self_nheads,
            pre_self_head_dim,
        ),
        device=device,
        dtype=MODEL_TYPE,
    )
    dummy_pre2_kv_cache = torch.zeros(
        (
            pre_cfg.mhsa_layers,
            2,
            BEAM_SIZE,
            pre_cfg.max_seqlen,
            pre_self_nheads,
            pre_self_head_dim,
        ),
        device=device,
        dtype=MODEL_TYPE,
    )

    dummy_new_prefix_len = 4
    dummy_current_seqlen = 8  # tokens already resident in the caches
    dummy_post_len = 4

    dummy_pre1_input_ids = torch.randint(
        low=0, high=pre_cfg.vocab_size, size=(1, dummy_new_prefix_len), dtype=torch.long
    )
    dummy_pre2_input_ids = torch.randint(
        low=0, high=pre_cfg.vocab_size, size=(BEAM_SIZE, 1), dtype=torch.long
    )
    dummy_pre_cache_pos = torch.tensor(
        [dummy_current_seqlen], dtype=torch.long, device=device
    )

    # ---------------------------------------------------------------- pre pass 1
    pre1_example_kwargs = {
        "input_ids": dummy_pre1_input_ids,
        "kv_cache_memory": dummy_pre1_kv_cache,
        "current_seqlen": dummy_pre_cache_pos,
        "use_custom_ops": True,
        "return_logits": False,
    }
    new_len_dim = Dim("new_prefix_len", min=1, max=pre_cfg.max_seqlen)
    pre1_dynamic_shapes = {
        "input_ids": {1: new_len_dim},
        "kv_cache_memory": None,
        "current_seqlen": None,
        "use_custom_ops": None,
        "return_logits": None,
    }
    exported_pre1 = torch.export.export(
        pre_model,
        args=(),
        kwargs=pre1_example_kwargs,
        dynamic_shapes=pre1_dynamic_shapes,
        strict=True,
    ).module()

    # ---------------------------------------------------------------- post model
    dummy_post_input_ids = torch.randint(
        low=0, high=post_cfg.vocab_size, size=(1, dummy_post_len), dtype=torch.long
    )
    post_example_kwargs = {"input_ids": dummy_post_input_ids}
    post_len_dim = Dim("post_len", min=1, max=post_cfg.max_seqlen)
    post_dynamic_shapes = {"input_ids": {1: post_len_dim}}
    exported_post = torch.export.export(
        post_model,
        args=(),
        kwargs=post_example_kwargs,
        dynamic_shapes=post_dynamic_shapes,
        strict=True,
    ).module()

    # ---------------------------------------------------------------- pre pass 2
    dummy_post_hidden = torch.zeros(
        (BEAM_SIZE, dummy_post_len, post_cfg.model_dim), device=device, dtype=MODEL_TYPE
    )
    # The logits mask is aligned 1:1 with the current decode chunk.
    dummy_logits_mask = torch.ones((BEAM_SIZE, 1, pre_cfg.proj_size), dtype=torch.bool)
    # Local cross-attn positions: pinyin key offset 1, query step = chunk start.
    dummy_post_position_offset = torch.tensor(1, dtype=torch.long, device=device)
    dummy_cross_q_pos_start = torch.tensor(0, dtype=torch.long, device=device)

    pre2_example_kwargs = {
        "input_ids": dummy_pre2_input_ids,
        "kv_cache_memory": dummy_pre2_kv_cache,
        "current_seqlen": dummy_pre_cache_pos,
        "post_hidden": dummy_post_hidden,
        "post_position_offset": dummy_post_position_offset,
        "cross_q_pos_start": dummy_cross_q_pos_start,
        "logits_mask": dummy_logits_mask,
        "use_custom_ops": True,
    }
    pre2_dynamic_shapes = {
        "input_ids": None,
        "kv_cache_memory": None,
        "current_seqlen": None,
        "post_hidden": {
            1: Dim("post_len2", min=1, max=post_cfg.max_seqlen),
        },
        "post_position_offset": None,
        "cross_q_pos_start": None,
        "logits_mask": None,
        "use_custom_ops": None,
    }
    exported_pre2 = torch.export.export(
        pre_model,
        args=(),
        kwargs=pre2_example_kwargs,
        dynamic_shapes=pre2_dynamic_shapes,
        strict=True,
    ).module()

    if os.environ.get("PHONOP2C_EXPORT_PRINT_GRAPHS"):
        exported_pre1.print_readable()
        exported_pre2.print_readable()
        exported_post.print_readable()

    os.makedirs(SAVE_DIR, exist_ok=True)

    quantized_pre_programs = {
        "pre_model_pass1": quantize_exported(
            exported_pre1, pre1_example_kwargs, pre1_dynamic_shapes
        ),
        "pre_model_pass2": quantize_exported(
            exported_pre2, pre2_example_kwargs, pre2_dynamic_shapes
        ),
    }
    lower_and_save(quantized_pre_programs, os.path.join(SAVE_DIR, "pre_model.pte"))

    quantized_post = quantize_exported(
        exported_post, post_example_kwargs, post_dynamic_shapes
    )
    lower_and_save(
        {"post_model": quantized_post}, os.path.join(SAVE_DIR, "post_model.pte")
    )

    generate_ops_manifests()
