"""Checkpoint RNG restore failures must be visible and CPU-state recoverable."""
import random

import numpy as np
import pytest
import torch

from accelerate.checkpointing import load_accelerator_state, save_accelerator_state
from accelerate.utils import RNG_STATE_NAME


def fixture_checkpoint(tmp_path):
    # No Accelerator launcher or GPU is needed for a fully initialized,
    # saved vanilla model/optimizer plus all CPU RNG streams.
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    saved_before = cpu_rng_snapshot()
    save_accelerator_state(
        output_dir=str(tmp_path), model_states=[model.state_dict()],
        optimizers=[optimizer], schedulers=[], dataloaders=[],
        process_index=0, step=7,
    )
    return model, optimizer, saved_before, tmp_path / f"{RNG_STATE_NAME}_0.pkl"


def cpu_rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def same_rng(a, b):
    assert a[0] == b[0]
    assert a[1][0] == b[1][0]
    np.testing.assert_array_equal(a[1][1], b[1][1])
    assert a[1][2:] == b[1][2:]
    assert torch.equal(a[2], b[2])


def load_checkpoint(tmp_path, model, opt, **kw):
    return load_accelerator_state(
        input_dir=str(tmp_path), models=[model], optimizers=[opt],
        schedulers=[], dataloaders=[], process_index=0, **kw
    )


def test_valid_cpu_rng_restore_preserves_full_random_continuation(tmp_path):
    model, opt, before, _ = fixture_checkpoint(tmp_path)
    random.seed(777)
    np.random.seed(777)
    torch.manual_seed(777)
    attrs = load_checkpoint(tmp_path, model, opt, strict_rng_state=True)
    assert attrs.get("step") == 7
    same_rng(cpu_rng_snapshot(), before)


def test_truncated_rng_file_warns_and_does_not_alter_cpu_streams(tmp_path, monkeypatch):
    model, opt, _, rng_path = fixture_checkpoint(tmp_path)
    rng_path.write_bytes(b"CORRUPTED INCOMPLETE CHECKPOINT")
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    before = cpu_rng_snapshot()
    warnings = []
    from accelerate import checkpointing
    monkeypatch.setattr(checkpointing.logger, "warning", lambda message: warnings.append(message))
    result = load_checkpoint(tmp_path, model, opt)
    assert "step" not in result
    assert len(warnings) == 1
    assert "NOT guaranteed to reproduce" in warnings[0]
    assert str(rng_path) in warnings[0]
    same_rng(cpu_rng_snapshot(), before)


def test_strict_corruption_fails_closed_without_mutating_cpu_rng(tmp_path):
    model, opt, _, rng_path = fixture_checkpoint(tmp_path)
    rng_path.write_bytes(b"CORRUPTED INCOMPLETE CHECKPOINT")
    random.seed(5)
    np.random.seed(6)
    torch.manual_seed(7)
    before = cpu_rng_snapshot()
    with pytest.raises(RuntimeError, match="Could not restore RNG state"):
        load_checkpoint(tmp_path, model, opt, strict_rng_state=True)
    same_rng(cpu_rng_snapshot(), before)


def test_missing_rng_field_does_not_partially_restore_python_rng(tmp_path, monkeypatch):
    model, opt, _, rng_path = fixture_checkpoint(tmp_path)
    # The file is readable and the training step is present, but a required
    # CPU RNG field is missing; do not accept a partial restore as success.
    torch.save(
        {"step": 7, "random_state": random.getstate(),
         "numpy_random_seed": np.random.get_state()},
        rng_path,
    )
    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    prior = cpu_rng_snapshot()
    warnings = []
    from accelerate import checkpointing
    monkeypatch.setattr(checkpointing.logger, "warning", lambda message: warnings.append(message))
    result = load_checkpoint(tmp_path, model, opt)
    assert "step" not in result
    assert any("torch_manual_seed" in msg for msg in warnings)
    same_rng(cpu_rng_snapshot(), prior)
