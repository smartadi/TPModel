import numpy as np


def gridvectors(omega: np.ndarray, density: np.ndarray):
    """Create grid vectors for each dimension.

    Args:
        omega: (N, 2) array of [min, max] ranges per dimension.
        density: (N,) array with number of grid points per dimension.

    Returns:
        List of numpy arrays, one per dimension.
    """
    grids = []
    for idx, count in enumerate(density):
        grids.append(np.linspace(omega[idx, 0], omega[idx, 1], int(count)))
    return grids


def weighting_function(p: np.ndarray, U, G):
    """Compute weighting function values at point p.

    Args:
        p: 1D array of parameter values.
        U: list of weighting matrices (rows aligned with grid points).
        G: list of grid vectors.

    Returns:
        List of 1xR vectors (numpy arrays) for each dimension.
    """
    w = []
    for n, Un in enumerate(U):
        if Un is None or len(Un) == 0:
            w.append(None)
            continue
        grid = G[n]
        if p[n] <= grid[0]:
            w.append(Un[0, :])
        elif p[n] >= grid[-1]:
            w.append(Un[-1, :])
        else:
            # Interpolate each column independently
            values = np.array(
                [np.interp(p[n], grid, Un[:, col]) for col in range(Un.shape[1])]
            )
            w.append(values)
    return w
