from __future__ import division
import math

class LRScheduler:
  def __init__(self, optimizer, epochs, lr, warmup_epochs = 0, base_lr = 0.):
    self.optimizer = optimizer
    self.epochs = epochs
    self.lr = lr
    self.warmup_epochs = warmup_epochs
    self.base_lr = base_lr
    self._cur_epoch = 0

  def _get_lr(self):
    raise NotImplementedError

  def get_lr(self):
    if self._cur_epoch < self.warmup_epochs:
      return self.base_lr + self._cur_epoch / self.warmup_epochs * (self.lr - self.base_lr)
    else:
      return self._get_lr()

  def step(self):
    lr = self.get_lr()
    for pg in self.optimizer.param_groups:
      pg['lr'] = lr
    self._cur_epoch += 1

class CosineLRScheduler(LRScheduler):
  def __init__(self, optimizer, epochs, lr, warmup_epochs = 0, base_lr = 0.):
    super(CosineLRScheduler, self).__init__(optimizer, epochs, lr, warmup_epochs, base_lr)

  def _get_lr(self):
    theta = math.pi * (self._cur_epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
    return self.lr * 1 / 2. * (1. + math.cos(theta))

class StepLRScheduler(LRScheduler):
  def __init__(self, optimizer, epochs, lr, decay_epochs, gamma = 0.1, warmup_epochs = 0, base_lr = 0.):
    super(StepLRScheduler, self).__init__(optimizer, epochs, lr, warmup_epochs, base_lr)
    self.decay_epochs = sorted(decay_epochs)
    self.gamma = gamma

  def _get_lr(self):
    lr = self.lr
    for e in self.decay_epochs:
      if self._cur_epoch >= e:
        lr *= self.gamma
    return lr

class ConstantLRScheduler(LRScheduler):
  def __init__(self, optimizer, epochs, lr, warmup_epochs = 0, base_lr = None):
    if base_lr is None:
      base_lr = lr
    super(ConstantLRScheduler, self).__init__(optimizer, epochs, lr, warmup_epochs, base_lr)

  def get_lr(self):
    return self.lr

def get_lrscheduler(optimizer, config):
  if config.lr_scheduler == 'cosine':
    return CosineLRScheduler(optimizer, config.epochs, config.lr, config.warmup_epochs, config.base_lr)
  elif config.lr_scheduler == 'step':
    return StepLRScheduler(optimizer, config.epochs, config.lr, config.decay_epochs, config.gamma, config.warmup_epochs, config.base_lr)
  elif config.lr_scheduler == 'constant':
    return ConstantLRScheduler(optimizer, config.epochs, config.lr, config.warmup_epochs, config.base_lr)
  else:
    raise NotImplementedError
