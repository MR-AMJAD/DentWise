"""ioustitch - tracking-free stitching of intraoral ultrasound sweeps (probe-fixed layer separation, masked
registration with Laplace covariances, lift-off-aware multi-lag pairs, robust factor graph with GNC and VCE,
uncertainty-weighted two-band compositing)."""
from .config import StitchConfig
from .pipeline import stitch_sweep
__version__ = "1.0.0"
