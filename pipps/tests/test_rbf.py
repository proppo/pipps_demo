import torch
import pytest
import proppo.tests.proppo_test_utils as utils

from pipps.modules.rbf import RBF


@pytest.mark.parametrize("input_size", [16])
@pytest.mark.parametrize("output_size", [8])
@pytest.mark.parametrize("batch_size", [256])
@pytest.mark.parametrize("num_centers", [50])
def test_rbf(input_size, output_size, batch_size, num_centers):
    model = RBF(input_size, output_size, batch_size, num_centers)

    x = torch.randn(batch_size, input_size)

    y = model(x)

    # check shape
    assert y.shape == (batch_size, output_size)

    # check gradient
    loss = ((y - 1.0)**2).mean()
    loss.backward()
    utils.check_not_zero(model.centers.grad)
    utils.check_not_zero(model.lam.grad)
    utils.check_not_zero(model.fc.weight.grad)
