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

    pre_njt = _njt([[1, 5, 7, 9, 11], [1, 3, 4, 12]])
    postfix_njt = _njt([[3, 5], [2, 4, 6]])
    target_njt = _njt([[10, 20], [30, 40, 50]])

    flat_pre = pre_njt.values()
    pre_offsets = pre_njt.offsets()
    flat_postfix = postfix_njt.values()
    postfix_offsets = postfix_njt.offsets()
    flat_target = target_njt.values()

    lens_pre = pre_offsets[1:] - pre_offsets[:-1]
    lens_post = postfix_offsets[1:] - postfix_offsets[:-1]

    out = compiled(
        flat_pre, pre_offsets, flat_postfix, postfix_offsets, flat_target,
        int(lens_pre.min()), int(lens_pre.max()),
        int(lens_post.min()), int(lens_post.max()),
    )
    out.loss.backward()
    assert torch.isfinite(out.loss)
