import os
import time

import torch

from .trainer import Trainer
from .regular_trainer import RegularTrainer
from .admm_trainer import ADMMTrainer
from .ft_trainer import FTTrainer

name2trainer = {
  'REGULAR': RegularTrainer,
  'ADMM': ADMMTrainer,
  'FT': FTTrainer
}

class SparseTrainer(Trainer):
  def __init__(self, config):
    super(SparseTrainer, self).__init__()

    self._init_config_info(config)

  def _init_config_info(self, config):
    self._init_model_config(config.MODEL)

    def _init_config(name):
      cfg_attr_name = '_{}_config'.format(name.lower())
      trainer_attr_name = '_{}_trainer'.format(name.lower())
      if hasattr(config, name):
        print('Initializing {} trainer'.format(name.lower()))
        cfg = getattr(config, name)
        cfg.save = config.save
        trainer = name2trainer[name](cfg)
      else:
        cfg = None
        trainer = None
      setattr(self, cfg_attr_name, cfg)
      setattr(self, trainer_attr_name, trainer)
      

    _init_config('REGULAR')
    _init_config('ADMM')
    _init_config('FT')


  def run_trainer(self, trainer, config):
    if trainer is None or config is None:
      return

    trainer._model = self._model
    trainer.model = self.model
    trainer._init_config_info(config)
    trainer.run()

  def run(self):
    self.run_trainer(self._regular_trainer, self._regular_config)
    self.run_trainer(self._admm_trainer, self._admm_config)
    self.run_trainer(self._ft_trainer, self._ft_config)
