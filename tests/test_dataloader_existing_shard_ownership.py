"""Pre-sharded data ownership, exact coverage and non-default sampler preservation.

Tests deliberately execute the real Accelerate preparation path in process-index
configuration, but do NOT claim actual multi-node collective synchronization.
"""
import itertools
import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset, Subset

from accelerate import Accelerator
from accelerate.data_loader import BatchSamplerShard, IterableDatasetShard, prepare_data_loader
from accelerate.utils import DataLoaderConfiguration


class LocalIterable(IterableDataset):
    def __init__(self, items):
        self.items = tuple(items)

    def __iter__(self):
        yield from self.items


def numbers(loader):
    return list(itertools.chain.from_iterable(
        batch.cpu().reshape(-1).tolist() for batch in loader
    ))


def test_four_rank_map_style_ownership_has_no_missing_or_duplicate_ids():
    source = torch.arange(48)
    observed = []
    for rank in range(4):
        local = list(range(rank, 48, 4))
        loader = DataLoader(Subset(source, local), batch_size=3, shuffle=False)
        result = prepare_data_loader(
            loader, num_processes=4, process_index=rank,
            already_sharded=True, dispatch_batches=False,
        )
        actual = numbers(result)
        assert actual == local
        assert not isinstance(result.batch_sampler, BatchSamplerShard)
        observed.extend(actual)
    assert sorted(observed) == list(range(48))
    assert len(set(observed)) == 48


def test_four_rank_iterable_ownership_not_divided_again():
    observed = []
    for rank in range(4):
        local = list(range(rank, 44, 4))
        loader = DataLoader(LocalIterable(local), batch_size=2)
        result = prepare_data_loader(
            loader, num_processes=4, process_index=rank,
            already_sharded=True, dispatch_batches=False,
        )
        assert not isinstance(result.dataset, IterableDatasetShard)
        assert numbers(result) == local
        observed.extend(numbers(result))
    assert sorted(observed) == list(range(44))


def test_default_global_map_loader_retains_regular_distributed_sharding():
    source = torch.arange(48)
    result = prepare_data_loader(
        DataLoader(source, batch_size=3), num_processes=4, process_index=1,
        dispatch_batches=False,
    )
    assert isinstance(result.batch_sampler, BatchSamplerShard)
    assert len(numbers(result)) == 12


def test_already_sharded_iterable_disable_auto_dispatch_when_device_move_on():
    # Iterable + put_on_device=True normally implies dispatch_batches=True.
    # Pre-sharded local data must retain local ownership and not broadcast.
    loader = DataLoader(LocalIterable([9, 10, 11, 12]), batch_size=2)
    result = prepare_data_loader(
        loader, device=torch.device("cpu"), put_on_device=True,
        num_processes=2, process_index=1,
        already_sharded=True, dispatch_batches=None,
    )
    assert not isinstance(result.dataset, IterableDatasetShard)
    assert numbers(result) == [9, 10, 11, 12]


@pytest.mark.parametrize("bad", [
    {"dispatch_batches": True, "put_on_device": True},
    {"split_batches": True},
    {"use_seedable_sampler": True},
])
def test_reject_conflicting_repartitioning_controls_before_iteration(bad):
    loader = DataLoader(torch.arange(8), batch_size=2)
    with pytest.raises(ValueError, match="already_sharded=True"):
        prepare_data_loader(loader, num_processes=2, process_index=0,
                            already_sharded=True, **bad)


def test_single_process_accelerator_config_plumbs_data_ownership():
    cfg = DataLoaderConfiguration(already_sharded=True)
    accelerator = Accelerator(dataloader_config=cfg, cpu=True)
    assert accelerator.dataloader_config.already_sharded
    prepared = accelerator.prepare_data_loader(DataLoader(torch.arange(8), batch_size=2))
    assert numbers(prepared) == list(range(8))


def test_rank_local_shuffle_or_sampler_identity_is_not_silently_rewritten():
    # The local sampler can implement its own deterministic domain-specific order.
    class ReverseSampler(torch.utils.data.Sampler):
        def __init__(self, n):
            self.n = n
        def __iter__(self):
            yield from range(self.n - 1, -1, -1)
        def __len__(self):
            return self.n

    source = torch.arange(8)
    base = DataLoader(source, sampler=ReverseSampler(8), batch_size=2)
    actual = prepare_data_loader(base, num_processes=4, process_index=3,
                                 already_sharded=True, dispatch_batches=False)
    assert numbers(actual) == list(range(7, -1, -1))
