"""styletransfer-lite: fast neural style transfer without TensorFlow or PyTorch."""
from .core import NUM_STYLES, StyleTransfer, StyleTransferError, load_image, parse_style

__version__ = "0.1.0"
__all__ = ["StyleTransfer", "StyleTransferError", "NUM_STYLES", "load_image", "parse_style", "__version__"]
