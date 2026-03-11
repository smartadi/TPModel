import numpy as np
from tptoolbox.model import TPmodel


def test_tpmodel_basic():
    S = np.array([[1.0, 2.0], [3.0, 4.0]])
    U = [np.eye(2)]
    G = [np.array([0.0, 1.0])]
    y = TPmodel(np.array([0.0]), S, U, G)
    assert np.allclose(y, np.array([1.0, 2.0]))
