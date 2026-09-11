"""
Export PhonoP2CPreModel and PhonoP2CPostModel to ExecuTorch .pte files.

The new-standard architecture exports two programs:

  * pre_model.pte  — a multi-method program containing both decoder passes.
    Keeping the methods together lets ExecuTorch deduplicate their shared
    pre-model parameters.
  * post_model.pte — the pinyin encoder: hidden states + logits mask.
"""

from pathlib import Path

import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.export import Dim
from model.model import PhonoP2CPreModel, PhonoP2CPostModel

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import EdgeProgramManager, to_edge_transform_and_lower
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.passes import MemoryPlanningPass
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.utils.fuser_utils import validate_partition

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


class _CrossKvProjector(torch.nn.Module):
    """Expose every pre-layer MHCALayer.project_kv as one export method."""

    def __init__(self, pre_model):
        super().__init__()
        self.layers = pre_model.layers

    def forward(self, post_hidden):
        return torch.stack(
            [torch.stack(layer["mhca"].project_kv(post_hidden)) for layer in self.layers]
        )


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


def _absolute_path(path: str | Path) -> Path:
    """Resolve a configured path against Hydra's original working directory."""
    return Path(to_absolute_path(str(Path(path).expanduser())))


def load_model_from_checkpoint(checkpoint_dir: str | Path, device: torch.device):
    """Load pre and post models from a checkpoint directory.

    The directory should contain:
      pre_model/  — save_pretrained output (config.json + model.safetensors)
      post_model/ — save_pretrained output (config.json + model.safetensors)
    """
    checkpoint_dir = Path(checkpoint_dir)
    pre_path = checkpoint_dir / "pre_model"
    post_path = checkpoint_dir / "post_model"
    if not pre_path.is_dir() or not post_path.is_dir():
        raise FileNotFoundError(
            f"{checkpoint_dir} must contain pre_model/ and post_model/ directories"
        )

    pre_model = PhonoP2CPreModel.from_pretrained(pre_path).to(device)
    post_model = PhonoP2CPostModel.from_pretrained(post_path).to(device)

    pre_model.eval()
    post_model.eval()
    return pre_model, post_model


def prepare_exported(
    exported_module,
    example_kwargs,
    dynamic_shapes,
    quantization_cfg: DictConfig,
    strict: bool,
):
    """Apply the configured quantization and return an ExportedProgram."""
    mode = str(quantization_cfg.mode).lower()
    if mode == "none":
        return torch.export.export(
            exported_module,
            args=(),
            kwargs=example_kwargs,
            dynamic_shapes=dynamic_shapes,
            strict=strict,
        )
    if mode not in {"w8a8", "w4a8"}:
        raise ValueError(f"Unsupported quantization mode: {mode}")

    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e

    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    quantizer = XNNPACKQuantizer()
    weight_range = {} if mode == "w8a8" else {
        "weight_qmin": -8,
        "weight_qmax": 7,
    }
    quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=bool(quantization_cfg.per_channel),
            is_dynamic=bool(quantization_cfg.dynamic),
            **weight_range,
        )
    )
    quantizer.set_filter_function(
        lambda node: not (
            node.target == torch.ops.aten.linear.default
            and getattr(node.args[1], "op", None) not in {"placeholder", "get_attr"}
        )
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
        strict=strict,
    )

    return quantized_exported


def lower_and_save(programs, save_path: Path, alloc_graph_input: bool):
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
            memory_planning_pass=MemoryPlanningPass(
                alloc_graph_input=alloc_graph_input
            ),
        )
    )

    with save_path.open("wb") as f:
        executorch_program.write_to_file(f)
    print(f"Saved {save_path.name} to {save_path}")


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


def generate_ops_manifests(
    output_dir: Path,
    manifest_dir: Path,
    manifest_tag: str,
    pre_filename: str,
    post_filename: str,
):
    """Generate per-model + merged ExecuTorch selective-build manifests.

    phono-core deploys two programs (pre_model.pte + post_model.pte) on one
    runtime, but ExecuTorch's upstream tooling can only build an operator list
    from a single .pte model. This reads the serialized programs and writes:

      * <tag>_pre_ops.yaml / <tag>_post_ops.yaml — per-model
        manifests (operators + dtype/dim-order kernel metadata), and
      * <tag>_ops.yaml — their union, for a combined build.

    Copy whichever you need into phono-core/ops_config/ before building
    phono-core.
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)

    per_model = {}
    for tag, pte_name in (("pre", pre_filename), ("post", post_filename)):
        pte_path = output_dir / pte_name
        ops, metadata = _collect_ops_metadata(pte_path)
        per_model[tag] = (ops, metadata)
        out = manifest_dir / f"{manifest_tag}_{tag}_ops.yaml"
        _dump_ops_manifest(ops, metadata, f"{manifest_tag}-{tag}", out)
        print(f"Manifest {tag}: {len(ops)} operators -> {out}")

    all_ops = sorted(set().union(*(ops for ops, _ in per_model.values())))
    merged_metadata = {}
    for _, metadata in per_model.values():
        for op, keys in metadata.items():
            seen = merged_metadata.setdefault(op, [])
            for key in keys:
                if key not in seen:
                    seen.append(key)
    merged_out = manifest_dir / f"{manifest_tag}_ops.yaml"
    _dump_ops_manifest(all_ops, merged_metadata, manifest_tag, merged_out)
    print(f"Manifest merged: {len(all_ops)} operators -> {merged_out}")


def run_export(task_cfg: DictConfig) -> None:
    """Run the ExecuTorch export task from its Hydra configuration."""
    device = torch.device(str(task_cfg.device))
    if device.type != "cpu":
        raise ValueError("ExecuTorch XNNPACK export currently requires task.device=cpu")
    try:
        dtype = DTYPES[str(task_cfg.dtype).lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported export dtype {task_cfg.dtype!r}; choose one of {sorted(DTYPES)}"
        ) from exc

    checkpoint_dir = _absolute_path(task_cfg.checkpoint_dir)
    output_dir = _absolute_path(task_cfg.output_dir)
    pre_filename = str(task_cfg.outputs.pre_model)
    post_filename = str(task_cfg.outputs.post_model)
    manifest_dir = output_dir / str(task_cfg.outputs.manifests_dir)
    example_cfg = task_cfg.example_inputs
    beam_size = int(example_cfg.beam_size)
    if beam_size <= 0:
        raise ValueError("example_inputs.beam_size must be positive")
    dummy_new_prefix_len = int(example_cfg.prefix_length)
    dummy_current_seqlen = int(example_cfg.current_sequence_length)
    dummy_post_len = int(example_cfg.post_length)
    max_candidate_width = int(example_cfg.max_candidate_width)
    if dummy_new_prefix_len <= 0 or dummy_post_len <= 0:
        raise ValueError("example input sequence lengths must be positive")
    if dummy_current_seqlen < 0:
        raise ValueError("example_inputs.current_sequence_length cannot be negative")
    if max_candidate_width <= 0:
        raise ValueError("example_inputs.max_candidate_width must be positive")

    print(
        f"Exporting {task_cfg.model_metadata.version} "
        f"(format {task_cfg.model_metadata.format_version}, "
        f"quantization={task_cfg.quantization.mode})"
    )

    pre_model, post_model = load_model_from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        device=device,
    )
    pre_model.to(dtype)
    post_model.to(dtype)

    for param in pre_model.parameters():
        param.requires_grad = False

    for param in post_model.parameters():
        param.requires_grad = False

    pre_cfg = pre_model.config
    post_cfg = post_model.config
    cross_kv_projector = _CrossKvProjector(pre_model)
    post_model.enable_sparse_logits()
    candidate_width = post_model.logits_candidate_ids.shape[1]

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
        dtype=dtype,
    )
    dummy_pre2_kv_cache = torch.zeros(
        (
            pre_cfg.mhsa_layers,
            2,
            beam_size,
            pre_cfg.max_seqlen,
            pre_self_nheads,
            pre_self_head_dim,
        ),
        device=device,
        dtype=dtype,
    )

    if dummy_current_seqlen + dummy_new_prefix_len > pre_cfg.max_seqlen:
        raise ValueError("example prefix and cache lengths exceed pre_model.max_seqlen")
    if dummy_post_len > post_cfg.max_seqlen:
        raise ValueError("example_inputs.post_length exceeds post_model.max_seqlen")

    dummy_pre1_input_ids = torch.randint(
        low=0, high=pre_cfg.vocab_size, size=(1, dummy_new_prefix_len), dtype=torch.long
    )
    dummy_pre2_input_ids = torch.randint(
        low=0, high=pre_cfg.vocab_size, size=(beam_size, 1), dtype=torch.long
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
        strict=bool(task_cfg.export.strict),
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
        strict=bool(task_cfg.export.strict),
    ).module()

    # ---------------------------------------------------------------- pre pass 2
    dummy_post_hidden = torch.zeros(
        (1, dummy_post_len, post_cfg.model_dim), device=device, dtype=dtype
    )
    dummy_cross_kv = cross_kv_projector(dummy_post_hidden)
    dummy_candidate_ids = torch.zeros(
        min(max_candidate_width, candidate_width), dtype=torch.long
    )
    # Local cross-attn positions: pinyin key offset 1, query step = chunk start.
    dummy_post_position_offset = torch.tensor(1, dtype=torch.long, device=device)
    dummy_cross_q_pos_start = torch.tensor(0, dtype=torch.long, device=device)

    pre2_example_kwargs = {
        "input_ids": dummy_pre2_input_ids,
        "kv_cache_memory": dummy_pre2_kv_cache,
        "current_seqlen": dummy_pre_cache_pos,
        "cross_kv": dummy_cross_kv,
        "post_position_offset": dummy_post_position_offset,
        "cross_q_pos_start": dummy_cross_q_pos_start,
        "logits_candidate_ids": dummy_candidate_ids,
        "use_custom_ops": True,
    }
    candidate_dim = Dim("candidate_width", min=1, max=candidate_width)
    pre2_dynamic_shapes = {
        "input_ids": None,
        "kv_cache_memory": None,
        "current_seqlen": None,
        "cross_kv": {
            3: Dim("cross_len", min=1, max=post_cfg.max_seqlen),
        },
        "post_position_offset": None,
        "cross_q_pos_start": None,
        "logits_candidate_ids": {0: candidate_dim},
        "use_custom_ops": None,
    }
    exported_pre2 = torch.export.export(
        pre_model,
        args=(),
        kwargs=pre2_example_kwargs,
        dynamic_shapes=pre2_dynamic_shapes,
        strict=bool(task_cfg.export.strict),
    ).module()

    cross_example_kwargs = {"post_hidden": dummy_post_hidden}
    cross_dynamic_shapes = {
        "post_hidden": {1: Dim("cross_source_len", min=1, max=post_cfg.max_seqlen)}
    }
    exported_cross_kv = torch.export.export(
        cross_kv_projector,
        args=(),
        kwargs=cross_example_kwargs,
        dynamic_shapes=cross_dynamic_shapes,
        strict=bool(task_cfg.export.strict),
    ).module()

    if task_cfg.export.print_graphs:
        exported_pre1.print_readable()
        exported_pre2.print_readable()
        exported_post.print_readable()

    output_dir.mkdir(parents=True, exist_ok=True)

    quantized_pre_programs = {
        "pre_model_pass1": prepare_exported(
            exported_pre1,
            pre1_example_kwargs,
            pre1_dynamic_shapes,
            task_cfg.quantization,
            bool(task_cfg.export.strict),
        ),
        "pre_model_pass2": prepare_exported(
            exported_pre2,
            pre2_example_kwargs,
            pre2_dynamic_shapes,
            task_cfg.quantization,
            bool(task_cfg.export.strict),
        ),
        "pre_model_cross_kv": prepare_exported(
            exported_cross_kv,
            cross_example_kwargs,
            cross_dynamic_shapes,
            task_cfg.quantization,
            bool(task_cfg.export.strict),
        ),
    }
    lower_and_save(
        quantized_pre_programs,
        output_dir / pre_filename,
        bool(task_cfg.export.alloc_graph_input),
    )

    quantized_post = prepare_exported(
        exported_post,
        post_example_kwargs,
        post_dynamic_shapes,
        task_cfg.quantization,
        bool(task_cfg.export.strict),
    )
    lower_and_save(
        {"post_model": quantized_post},
        output_dir / post_filename,
        bool(task_cfg.export.alloc_graph_input),
    )

    if task_cfg.manifests.enabled:
        generate_ops_manifests(
            output_dir=output_dir,
            manifest_dir=manifest_dir,
            manifest_tag=str(task_cfg.manifests.tag),
            pre_filename=pre_filename,
            post_filename=post_filename,
        )
