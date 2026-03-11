import numpy as np


def layout(tensor: np.ndarray, dim: int) -> np.ndarray:
    """Unfold a tensor along a dimension into a matrix."""
    ndim = tensor.ndim
    if ndim == 2:
        return tensor.T if dim == 1 else tensor
    order = list(range(ndim))
    order = order[dim:] + order[:dim]
    rotated = np.transpose(tensor, order)
    return rotated.reshape(rotated.shape[0], -1)


def fold(matrix: np.ndarray, dim: int, shape) -> np.ndarray:
    """Fold a matrix back into a tensor with the given shape."""
    ndim = len(shape)
    if ndim == 2:
        return matrix.T if dim == 1 else matrix
    new_shape = list(shape)
    new_shape = new_shape[dim:] + new_shape[:dim]
    new_shape[0] = matrix.shape[0]
    tensor = matrix.reshape(new_shape)
    order = list(range(ndim))
    order = order[-dim:] + order[:-dim]
    return np.transpose(tensor, order)


def tprod_n(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    """Mode-n tensor-matrix product."""
    result = np.tensordot(matrix, tensor, axes=(1, mode))
    return np.moveaxis(result, 0, mode)


def tprod(tensor: np.ndarray, matrices) -> np.ndarray:
    """Tensor product over all provided dimensions."""
    result = tensor
    for idx, mat in enumerate(matrices):
        if mat is None or len(mat) == 0:
            continue
        mat = np.asarray(mat)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        result = tprod_n(result, mat, idx)
    return result
