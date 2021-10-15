from .trainer import Trainer
from .regular_trainer import RegularTrainer
from .admm_trainer import ADMMTrainer
from .ft_trainer import FTTrainer
from .sparse_trainer import SparseTrainer

Trainer.registry = {
  'RegularTrainer' : RegularTrainer,
  'ADMMTrainer' : ADMMTrainer,
  'FTTrainer' : FTTrainer,
  'SparseTrainer' : SparseTrainer
}

def create_trainer(config):
  return Trainer.registry[config.type](config)

__all__ = ['create_trainer']
