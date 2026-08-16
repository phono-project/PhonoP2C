"""In-place KV cache update as a custom op (export-time side).

Usage in a model
----------------
    from model.custom_ops.kv_cache_ops import update_kv_cache
    ...
    update_kv_cache(k_cache, k, pos)   # mutates k_cache in place

Import this module once (anywhere) before calling torch.export.export() /
to_edge_transform_and_lower().
"""
import torch

_NAMESPACE = "phono"
_OP_NAME = "update_kv_cache"

_lib = torch.library.Library(_NAMESPACE, "DEF")

_lib.define(f"{_OP_NAME}(Tensor cache, Tensor value, SymInt start_pos) -> Tensor")
_lib.define(f"{_OP_NAME}.out(Tensor cache, Tensor value, SymInt start_pos, *, Tensor(a!) out) -> Tensor(a!)")

@torch.library.impl(_lib, _OP_NAME, "CPU")
def _update_kv_cache_impl(cache: torch.Tensor, value: torch.Tensor, start_pos: int) -> torch.Tensor:
    seq_len = value.shape[1]
    indices = torch.arange(seq_len, device=cache.device) + start_pos
    res = cache.clone()
    res.index_copy_(1, indices, value)
    return res

@torch.library.impl(_lib, f"{_OP_NAME}.out", "CPU")
def _update_kv_cache_out_impl(cache: torch.Tensor, value: torch.Tensor, start_pos: int, *, out: torch.Tensor) -> torch.Tensor:
    seq_len = value.shape[1]
    indices = torch.arange(seq_len, device=cache.device) + start_pos
    if out.data_ptr() != cache.data_ptr():
        out.copy_(cache)
    out.index_copy_(1, indices, value)
    return out

@torch.library.register_fake(f"{_NAMESPACE}::{_OP_NAME}")
def _update_kv_cache_fake(cache, value, start_pos):
    return torch.empty_like(cache)

@torch.library.register_fake(f"{_NAMESPACE}::{_OP_NAME}.out")
def _update_kv_cache_out_fake(cache, value, start_pos, *, out):
    return out

def kv_cache_write(cache: torch.Tensor, value: torch.Tensor, start_pos) -> torch.Tensor:
    if isinstance(start_pos, torch.Tensor):
        start_pos = start_pos.item()
    return torch.ops.phono.update_kv_cache(cache, value, start_pos)


_CROSS_OP = "update_cross_kv"
_MHSA_OP = "update_mhsa_kv"

_lib.define(f"{_CROSS_OP}(Tensor cache, Tensor pre_K, Tensor pre_V, SymInt start_pos) -> Tensor")
_lib.define(f"{_CROSS_OP}.out(Tensor cache, Tensor pre_K, Tensor pre_V, SymInt start_pos, *, Tensor(a!) out) -> Tensor(a!)")
_lib.define(f"{_MHSA_OP}(Tensor cache, Tensor k_val, Tensor v_val, SymInt start_pos, SymInt layer_idx) -> Tensor")
_lib.define(f"{_MHSA_OP}.out(Tensor cache, Tensor k_val, Tensor v_val, SymInt start_pos, SymInt layer_idx, *, Tensor(a!) out) -> Tensor(a!)")


@torch.library.impl(_lib, _CROSS_OP, "CPU")
def _update_cross_kv_impl(cache, pre_K, pre_V, start_pos):
    res = cache.clone()
    for ch, val in enumerate([pre_K, pre_V]):
        seq_len = val.shape[1]
        indices = torch.arange(seq_len, device=cache.device) + start_pos
        res[ch].index_copy_(1, indices, val)
    return res


@torch.library.impl(_lib, f"{_CROSS_OP}.out", "CPU")
def _update_cross_kv_out_impl(cache, pre_K, pre_V, start_pos, *, out):
    if out.data_ptr() != cache.data_ptr():
        out.copy_(cache)
    for ch, val in enumerate([pre_K, pre_V]):
        seq_len = val.shape[1]
        indices = torch.arange(seq_len, device=cache.device) + start_pos
        out[ch].index_copy_(1, indices, val)
    return out


@torch.library.impl(_lib, _MHSA_OP, "CPU")
def _update_mhsa_kv_impl(cache, k_val, v_val, start_pos, layer_idx):
    res = cache.clone()
    for ch, val in enumerate([k_val, v_val]):
        seq_len = val.shape[1]
        indices = torch.arange(seq_len, device=cache.device) + start_pos
        res[layer_idx, ch].index_copy_(1, indices, val)
    return res


@torch.library.impl(_lib, f"{_MHSA_OP}.out", "CPU")
def _update_mhsa_kv_out_impl(cache, k_val, v_val, start_pos, layer_idx, *, out):
    if out.data_ptr() != cache.data_ptr():
        out.copy_(cache)
    for ch, val in enumerate([k_val, v_val]):
        seq_len = val.shape[1]
        indices = torch.arange(seq_len, device=cache.device) + start_pos
        out[layer_idx, ch].index_copy_(1, indices, val)
    return out


@torch.library.register_fake(f"{_NAMESPACE}::{_CROSS_OP}")
def _update_cross_kv_fake(cache, pre_K, pre_V, start_pos):
    return torch.empty_like(cache)


@torch.library.register_fake(f"{_NAMESPACE}::{_CROSS_OP}.out")
def _update_cross_kv_out_fake(cache, pre_K, pre_V, start_pos, *, out):
    return out


@torch.library.register_fake(f"{_NAMESPACE}::{_MHSA_OP}")
def _update_mhsa_kv_fake(cache, k_val, v_val, start_pos, layer_idx):
    return torch.empty_like(cache)


@torch.library.register_fake(f"{_NAMESPACE}::{_MHSA_OP}.out")
def _update_mhsa_kv_out_fake(cache, k_val, v_val, start_pos, layer_idx, *, out):
    return out


def update_cross_kv(cache, pre_K, pre_V, start_pos):
    if isinstance(start_pos, torch.Tensor):
        start_pos = start_pos.item()
    return torch.ops.phono.update_cross_kv(cache, pre_K, pre_V, start_pos)


def update_mhsa_kv(cache, k_val, v_val, start_pos, layer_idx):
    if isinstance(start_pos, torch.Tensor):
        start_pos = start_pos.item()
    return torch.ops.phono.update_mhsa_kv(cache, k_val, v_val, start_pos, layer_idx)


def update_mhsa_kv_standard(cache, k_val, v_val, start_pos, layer_idx):
    """Pure-PyTorch equivalent of ``update_mhsa_kv`` (no custom op).

    The ``phono::update_mhsa_kv`` torch.library op only has a CPU backend and
    is intended for ExecuTorch export.  During eager eval/inference (e.g. beam
    search on CUDA) we route to this standard implementation instead, which
    works on any device and stays differentiable.
    """
    if isinstance(start_pos, torch.Tensor):
        start_pos = start_pos.item()
    seq_len = k_val.shape[1]
    indices = torch.arange(seq_len, device=cache.device) + start_pos
    res = cache.clone()
    res[layer_idx, 0].index_copy_(1, indices, k_val)
    res[layer_idx, 1].index_copy_(1, indices, v_val)
    return res