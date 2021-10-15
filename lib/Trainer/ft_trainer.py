import os
import time

import torch

from .regular_trainer import RegularTrainer

class FTTrainer(RegularTrainer):
  # Trainer for fine-tuning
  def __init__(self, config):
    super(FTTrainer, self).__init__(config)

    self._model_ck = os.path.join(config.save, 'model.ft.pth')
    self._best_model_ck = os.path.join(config.save, 'model.ft.best.pth')

  def _init_config_info(self, config):
    super(FTTrainer, self)._init_config_info(config)
    self._init_model_info()

  def _init_model_info(self):
    print('init sparse model info')
    import models
    self._prunable_convs = [m for m in self._model.modules() if isinstance(m, models.ops.PrunableConv) and not m.is_dw]
    self.M = []

    for conv in self._prunable_convs:
      weight = conv.weight.data[:conv.out_channels, :conv.in_channels, :, :]
      weight_flat = weight.reshape(conv.out_channels // conv.group_size, conv.group_size, -1)
      weight_norm = weight_flat.norm(p = 2, dim = 1).flatten()
      mask = torch.zeros_like(weight_norm)
      nremained = int(round(conv.wp_density * weight_norm.numel()))
      sorted_idx = weight_norm.argsort(descending = True)
      mask[sorted_idx[:nremained]] = 1
      self.M.append(mask)

    self._mask_params()
    self.test()
    self._best_acc = self._acc1

    print('Accuracy before fine-tuning: {:.2f}'.format(self._acc1 * 100))


  def _mask_params(self):
    for conv, mask in zip(self._prunable_convs, self.M):
      weight = conv.weight.data[:conv.out_channels, :conv.in_channels, :, :]
      weight = weight.reshape(conv.out_channels // conv.group_size, conv.group_size, -1)
      mask = mask.view(conv.out_channels // conv.group_size, 1, -1)
      weight.mul_(mask)
