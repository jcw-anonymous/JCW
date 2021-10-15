from .model_evaluator import Evaluator

from .ps_evaluator import PSEvaluator

Evaluator.registry = {
  'PSEvaluator': PSEvaluator
}

def create_model_evaluation(config):
  return Evaluator.registry[config.type](config)

__all__ = ['create_model_evaluation']
