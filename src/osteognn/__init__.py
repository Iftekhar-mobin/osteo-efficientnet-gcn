"""EfficientNet-GCN: hybrid convolutional-graph osteoporosis severity classification."""
from .config import ABLATIONS, BASELINES, Config, load_config, variant_config

__version__ = "1.0.0"
__all__ = ["Config", "load_config", "variant_config", "ABLATIONS", "BASELINES"]
