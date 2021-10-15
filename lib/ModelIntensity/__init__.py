from .model_intensity import ModelIntensity
from .model_latency import ModelLatency

ModelIntensity.registry = {
  'Latency': ModelLatency
}

def create_model_intensity(config):
  return ModelIntensity.registry[config.type](config)

__all__ = ['create_model_intensity']
