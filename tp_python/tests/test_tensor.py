import numpy as np
from tptoolbox.tensor import tprod


def test_tprod_matches_manual():
    tensor = np.arange(24).reshape(2, 3, 4)
    U1 = np.array([[1.0, 0.0]])
    U3 = np.eye(4)
    result = tprod(tensor, [U1, None, U3])

    expected = np.tensordot(U1, tensor, axes=(1, 0))
    expected = np.moveaxis(expected, 0, 0)
    expected = np.tensordot(U3, expected, axes=(1, 2))
    expected = np.moveaxis(expected, 0, 2)

    assert np.allclose(result, expected)
