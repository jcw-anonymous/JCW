import os
import time

import numpy as np
import torch

import utils
from .trainer import Trainer

def pre_forward_hook(trainer):
  if hasattr(trainer, 'M'):
    trainer._mask_params()

def pre_epoch_hook(trainer):
  trainer.start = time.time()
  print('EPOCH {} | {}...'.format(trainer._epoch + 1, trainer._epochs))
  trainer._scheduler.step()
  trainer._model.disable_conv_wp()
  
def epoch_hook(trainer, train_loss, train_acc1, train_acc5): 
  trainer._model.disable_conv_wp()
  if hasattr(trainer, 'M'):
    trainer._mask_params()
  trainer.test()
  state = trainer.state
  torch.save(state, trainer._model_ck)
  if trainer._acc1 > trainer._best_acc:
    trainer._best_acc = trainer._acc1
    torch.save(state, trainer._best_model_ck)
  trainer.end = time.time()

  print('Runtime: {} min. ' \
       'Loss: {}[{}]. ' \
       'Acc@1: {:.2f}[{:.2f}]. ' \
       'Acc@5: {:.2f}[{:.2f}]. ' \
       'Best acc: {:.2f}.'.format(
       (trainer.end - trainer.start) / 60.,
       train_loss,trainer._loss,
       train_acc1 * 100, trainer._acc1 * 100,
       train_acc5 * 100, trainer._acc5 * 100,
       trainer._best_acc * 100
      ))

def stop_hook(trainer):
  ck = torch.load(trainer._best_model_ck)
  trainer._model.load_state_dict(ck['model'])

  # delete unnecessary tensors to save memory
  del ck
  del trainer._optimizer
  del trainer._scheduler  
  if hasattr(trainer, 'M'):
    del trainer.M
 

class RegularTrainer(Trainer):
  def __init__(self, config):
    super(RegularTrainer, self).__init__()
    if hasattr(config, 'MODEL'):
      self._init_config_info(config)

    self._model_ck = os.path.join(config.save, 'model.pth')
    self._best_model_ck = os.path.join(config.save, 'model.best.pth')

    # variables
    self._epoch = 0
    self._best_acc = 0
    self._loss = 0
    self._acc1 = 0
    self._acc5 = 0 

    self._register_hook(pre_forward_hook, '_pre_forward')
    self._register_hook(pre_epoch_hook, '_pre_epoch')
    self._register_hook(epoch_hook, '_epoch')   
    self._register_hook(stop_hook, '_stop')



  @property
  def state(self):
    return {
      'model' : self._model.state_dict(),
      'optimizer' : self._optimizer.state_dict(),
      'epoch' : self._epoch,
      'validation' : [self._loss, self._acc1, self._acc5]
    }
