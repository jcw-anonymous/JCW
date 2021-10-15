import os
import time

import torch

from .trainer import Trainer

def pre_epoch_hook(trainer):
  print('ADMM EPOCH {} | {}...'.format(trainer._epoch + 1, trainer._epochs))
  trainer.start = time.time() 
  trainer._scheduler.step()
  trainer._model.disable_conv_wp()

def epoch_hook(trainer, train_loss, train_acc1, train_acc5): 
  trainer._model.disable_conv_wp()
  trainer._update_uz()
  trainer.test()
  state = trainer.state
  torch.save(state, trainer._model_ck)
  trainer.end = time.time()
 
  # print ADMM residual
  admm_residuals = []
  for m, u in zip(trainer._prunable_convs, trainer.U):
    weight = m.weight.data[:m.out_channels, :m.in_channels, :, :]
    admm_residuals.append((weight - u).norm().item())

  print('ADMM Residuals: {}'.format(' '.join(['{:.2f}'.format(res) for res in admm_residuals])))

  print('Runtime: {} min. ' \
       'Loss: {}[{}]. ' \
       'Acc@1: {:.2f}[{:.2f}]. ' \
       'Acc@5: {:.2f}[{:.2f}]. '.format(
       (trainer.end - trainer.start) / 60.,
       train_loss,trainer._loss,
       train_acc1 * 100, trainer._acc1 * 100,
       train_acc5 * 100, trainer._acc5 * 100
      ))

def pre_step_hook(trainer):
  trainer._adjust_grad() 

def stop_hook(trainer):
  # delete unnecessary tensors to save memory
  del trainer._optimizer
  del trainer._scheduler
  del trainer.U
  del trainer.Z

class ADMMTrainer(Trainer):
  def __init__(self, config):
    super(ADMMTrainer, self).__init__()
    
    self._model_ck = os.path.join(config.save, 'model.admm.pth')

    self._epoch = 0

    if hasattr(config, 'MODEL'):
      self._init_config_info(config)

    self._register_hook(pre_epoch_hook, '_pre_epoch')
    self._register_hook(epoch_hook, '_epoch')
    self._register_hook(pre_step_hook, '_pre_step')
    self._register_hook(stop_hook, '_stop')

  def _init_optim_config(self, optim_config):
    super(ADMMTrainer, self)._init_optim_config(optim_config)
    self._admm_rho = optim_config.rho

  def _init_config_info(self, config):
    super(ADMMTrainer, self)._init_config_info(config)
    # very tricky to resume training, may be improved
    config.setdefault('skip', False) 
    self.skip = config.skip
    self._init_model_info()

  def _init_model_info(self):
    import models
    self._prunable_convs = [m for m in self._model.modules() if isinstance(m, models.ops.PrunableConv) and not m.is_dw]
    self._init_uz()

  def _init_uz(self):
    self.Z = []
    self.U = []
    for m in self._prunable_convs:
      weight = m.weight.data[:m.out_channels, :m.in_channels, :, :]
      self.Z.append(torch.zeros_like(weight))
      self.U.append(weight.clone())

    self._update_uz()

  def _update_uz(self):
    self._update_u()
    self._update_z()

  def _update_u(self):
    for m, u, z in zip(self._prunable_convs, self.U, self.Z):
      weight = m.weight.data[:m.out_channels, :m.in_channels, :, :]
      u.copy_(weight)
      u.add_(z)
      
      u_flat = u.view(m.out_channels // m.group_size, m.group_size, -1)
      u_norm = u_flat.norm(p = 2, dim = 1).flatten()
      mask_flat = torch.zeros_like(u_norm)
      sorted_indices = u_norm.argsort(descending = True)
      nremained = int(round(m.wp_density * mask_flat.numel()))
      mask_flat[sorted_indices[:nremained]] = 1
      mask = mask_flat.view(m.out_channels // m.group_size, 1, -1)
      u_flat.mul_(mask)

  def _update_z(self):
    for m, u, z in zip(self._prunable_convs, self.U, self.Z):
      weight = m.weight.data[:m.out_channels, :m.in_channels, :, :]
      z.add_(weight - u)

  def _adjust_grad(self):
    for m, u, z in zip(self._prunable_convs, self.U, self.Z):
      grad = m.weight.grad[:m.out_channels, :m.in_channels, :, :]
      weight = m.weight.data[:m.out_channels, :m.in_channels, :, :]
      grad.add_(self._admm_rho, weight - u + z)

  def run(self):
    if self.skip:
      print('Skip admm training...')
      print('Load admm state from {}'.format(self._model_ck))
      ck = torch.load(self._model_ck, map_location = torch.device('cpu'))
      self._model.load_state_dict(ck['model'])
      del ck 
      # run stop hook and stop training
      for hook in self._stop_hooks.values():
        hook(self)
      return 
    super(ADMMTrainer, self).run()
      

  @property
  def state(self):
    return {
      'model': self._model.state_dict(),
      'optimizer': self._optimizer.state_dict(),
      'U': self.U,
      'Z': self.Z
    }
