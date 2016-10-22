# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import math
import numpy as np
from .. import Function
from ..ops import times
from ..utils import one_hot, cntk_device
from ..trainer import *
from ..learner import *
from .. import cross_entropy_with_softmax, classification_error, parameter, \
        input_variable, times, plus, reduce_sum
import pytest

def test_trainer(tmpdir):
    in1 = input_variable(shape=(1,))
    labels = input_variable(shape=(1,))
    p = parameter(shape=(2,), init=10)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, ce, errs, \
            [momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant)])
    in1_value = [[1],[2]]
    label_value = [[0], [1]]
    arguments = {in1: in1_value, labels: label_value}
    z_output = z.output
    updated, var_map = trainer.train_minibatch(arguments, [z_output])

    p = str(tmpdir / 'checkpoint.dat')
    trainer.save_checkpoint(p)
    trainer.restore_from_checkpoint(p)

    assert trainer.model.name == 'z'

    # Ensure that Swig is not leaking raw types
    assert isinstance(trainer.model, Function)
    assert trainer.model.__doc__
    assert isinstance(trainer.parameter_learners[0], Learner)

def test_output_to_retain():
    in1 = input_variable(shape=(1,))
    labels = input_variable(shape=(1,))
    p = parameter(shape=(2,), init=10)
    z = plus(in1, reduce_sum(p), name='z')
    ce = cross_entropy_with_softmax(z, labels)
    errs = classification_error(z, labels)

    momentum_time_constant = momentum_as_time_constant_schedule(1100)
    lr_per_sample = learning_rate_schedule(0.007, UnitType.sample)
    trainer = Trainer(z, ce, errs, \
            [momentum_sgd(z.parameters, lr_per_sample, momentum_time_constant)])
    in1_value = [[1],[2]]
    label_value = [[0], [1]]
    arguments = {in1: in1_value, labels: label_value}
    z_output = z.output
    updated, var_map = trainer.train_minibatch(arguments, [z_output])

    assert np.allclose(var_map[z_output], np.asarray(in1_value)+20)


def test_eval_sparse(sparse_type):
    from scipy.sparse import csr_matrix
    dim = 10
    in1 = input_variable(shape=(dim,), is_sparse=True)
    z = times(1, in1 * 2)
    value = np.eye(dim)
    expected = value * 2
    sparse_val = [csr_matrix(value)]
    assert np.allclose(z.eval({in1: sparse_val}), expected)

@pytest.mark.parametrize("one_hot_batch", [
     ([[2,5],
      [0,1,6]]),
     ([[1],
      [1],[2],[3]]),
    ])
def test_eval_one_hot(one_hot_batch, device_id):
    dim = 10
    # Just so that we can detect some impact on the one-hot vectors
    multiplier = 2
    in1 = input_variable(shape=dim)
    # Convert CNTK node value to dense so that we can compare it later
    z = times(in1, 1) 
    z = z * multiplier
    # Convert expectation to dense
    eye = np.eye(dim)
    expected = [eye[seq]*multiplier for seq in one_hot_batch]
    batch = one_hot(one_hot_batch, num_classes=dim, device=cntk_device(device_id))
    assert np.all(np.allclose(a,b) \
            for a,b in zip(z.eval({in1: batch}), expected))

@pytest.mark.parametrize("one_hot_batch, dim", [
    ([[11]], 10),
    ([[0, 1]], 1), 
    ])
# FIXME
def _test_eval_one_hot_bad(one_hot_batch, dim, device_id):
    in1 = input_variable(shape=dim)
    # Convert CNTK node value to dense so that we can compare it later
    z = times(in1, 1) 
    # Convert expectation to dense
    batch = one_hot(one_hot_batch, num_classes=dim, device=cntk_device(device_id))
    with pytest.raises(ValueError):
        z.eval({in1: batch})

