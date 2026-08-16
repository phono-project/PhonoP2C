"""Compiled-wrapper test (CUDA only; the CPU build lacks a jagged causal SDPA)."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA jagged SDPA")
def test_compiled_wrapper_forward_backward(tiny_models):
    from model.wrapper import PhonoP2CTrainWrapper
    from loss import get_loss_fn

    pre, post, _, _ = tiny_models
    wrapper = PhonoP2CTrainWrapper(pre, post, get_loss_fn("ce"))

    import torch._dynamo.config as dynamo_config
    dynamo_config.capture_scalar_outputs = True
    dynamo_config.capture_dynamic_output_shape_ops = True
    compiled = torch.compile(wrapper, mode="default", dynamic=True)

    def _njt(id_lists):
        ts = [torch.tensor(ids, dtype=torch.long) for ids in id_lists]
        return torch.nested.nested_tensor(ts, layout=torch.jagged)

    prefix_njt = _njt([[1, 5, 7], [1, 3]])
    suffix_njt = _njt([[4, 10], [4, 30, 40]])
    uncond_njt = _njt([[7, 8], [9]])
    postfix_njt = _njt([[3, 5], [2, 4, 6]])
    target_njt = _njt([[10, 20], [30, 40, 50]])
    prefix_lens = torch.tensor([3, 2])

    flat_prefix = prefix_njt.values()
    prefix_offsets = prefix_njt.offsets()
    flat_suffix = suffix_njt.values()
    suffix_offsets = suffix_njt.offsets()
    flat_uncond = uncond_njt.values()
    flat_postfix = postfix_njt.values()
    flat_target = target_njt.values()

    lens_prefix = prefix_offsets[1:] - prefix_offsets[:-1]
    lens_suffix = suffix_offsets[1:] - suffix_offsets[:-1]
    lens_full = prefix_lens + lens_suffix

    out = compiled(
        flat_prefix, prefix_offsets, flat_suffix, suffix_offsets,
        flat_postfix, flat_uncond, flat_target, prefix_lens,
        int(lens_prefix.min()), int(lens_prefix.max()),
        int(lens_suffix.min()), int(lens_suffix.max()),
        int(lens_full.min()), int(lens_full.max()),
    )
    out.loss.backward()
    assert torch.isfinite(out.loss)
