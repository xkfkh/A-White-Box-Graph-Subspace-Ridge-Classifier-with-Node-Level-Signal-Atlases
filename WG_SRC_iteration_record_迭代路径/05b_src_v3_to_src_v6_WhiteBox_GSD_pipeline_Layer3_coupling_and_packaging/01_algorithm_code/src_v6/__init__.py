"""HA-SDC: Homophily-Heterophily Adaptive Signed Dual-Subspace Crosslinking.

A white-box, non-neural node classifier for graphs with mixed homophily/heterophily.
"""

from .model import HASDC, HASDCModelState, Subspace

__all__ = ["HASDC", "HASDCModelState", "Subspace"]
__version__ = "0.1.0"
