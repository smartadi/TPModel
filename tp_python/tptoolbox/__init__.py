"""Python port of the TP Matlab Toolbox 2023."""

from .utils import gridvectors, weighting_function
from .tensor import tprod, tprod_n, layout, fold
from .hosvd import hosvd, svdtrunc
from .hull import genhull, genhullall
from .model import ConvexTPStructure, TPmodel
from .ssmodels import build_grid, build_grid_tensor, linearise, extract_lpv, extract_qlpv

__all__ = [
    "gridvectors",
    "weighting_function",
    "tprod",
    "tprod_n",
    "layout",
    "fold",
    "hosvd",
    "svdtrunc",
    "genhull",
    "genhullall",
    "ConvexTPStructure",
    "TPmodel",
    "build_grid",
    "build_grid_tensor",
    "linearise",
    "extract_lpv",
    "extract_qlpv",
]
