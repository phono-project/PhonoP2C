"""CPU tests for exact validation partitioning and metric reduction state."""

import pytest
import torch


def test_ddp_local_mean_scaling_matches_global_token_mean():
    from utils.distributed import ddp_local_mean_scale

    # Two ranks have different NJT token counts and local mean losses.
    counts = torch.tensor([3.0, 7.0])
    local_means = torch.tensor([2.0, 5.0])
    global_count = counts.sum()
    scales = torch.stack([
        ddp_local_mean_scale(count, global_count, world_size=2)
        for count in counts
    ])

    ddp_averaged_loss = (local_means * scales).mean()
    global_token_mean = (local_means * counts).sum() / global_count
    assert ddp_averaged_loss == pytest.approx(global_token_mean)


def test_ddp_local_mean_scaling_handles_no_valid_tokens():
    from utils.distributed import ddp_local_mean_scale

    scale = ddp_local_mean_scale(
        torch.tensor(0.0), torch.tensor(0.0), world_size=2
    )
    assert scale.item() == 0.0


def test_distributed_eval_sampler_partitions_without_duplicates():
    from utils.distributed import DistributedEvalSampler

    dataset = list(range(11))
    partitions = [
        list(DistributedEvalSampler(dataset, rank=rank, world_size=3))
        for rank in range(3)
    ]

    assert partitions == [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8]]
    assert sorted(index for part in partitions for index in part) == dataset


def test_metrics_state_merge_matches_single_process():
    from metrics import MetricsAccumulator

    torch.manual_seed(7)
    logits = torch.randn(6, 8)
    targets = torch.tensor([0, 3, 2, 6, 1, 5])

    reference = MetricsAccumulator(ece_bins=3, ece_top_k=2)
    reference.update(logits, targets, torch.tensor([0, 2, 5, 6]))

    rank0 = MetricsAccumulator(ece_bins=3, ece_top_k=2)
    rank0.update(logits[:5], targets[:5], torch.tensor([0, 2, 5]))
    rank1 = MetricsAccumulator(ece_bins=3, ece_top_k=2)
    rank1.update(logits[5:], targets[5:], torch.tensor([0, 1]))

    merged = MetricsAccumulator(ece_bins=3, ece_top_k=2)
    merged.merge_state_dict(rank0.state_dict())
    merged.merge_state_dict(rank1.state_dict())

    expected = reference.compute()
    actual = merged.compute()
    assert actual.keys() == expected.keys()
    for name in expected:
        assert actual[name] == pytest.approx(expected[name])
