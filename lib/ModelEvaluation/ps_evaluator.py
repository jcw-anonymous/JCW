import time

import numpy as np
import torch

from .model_evaluator import Evaluator
import utils

class PSEvaluator(Evaluator):
  def __init__(self, config):
    self._init_config_info(config)

  def _init_config_info(self, config):
    # model config
    self._init_model_config(config.MODEL)
    # dataset config
    self._init_dset_config(config.DATASET)
    # optimization config
    self._init_optim_config(config.OPTIM)
    # bn update
    config.setdefault('BNUPDATE', {
      'iters': 10,
      'recompute': True,
      'mode': 'exact'
    })
    self._init_bnupdate_config(config.BNUPDATE)

  def _init_bnupdate_config(self, bnupdate_config):
    self._bnupdate_config = bnupdate_config
    self._bnupdate_iters = bnupdate_config.iters
    self._bnupdate_recompute = bnupdate_config.recompute
    self._bnupdate_mode = bnupdate_config.mode

    # init bn update loader
    bnupdate_config.setdefault('reinit_dataloader', True)
    if not bnupdate_config.reinit_dataloader:
      self._bnupdate_loader = self._train_loader
    else:
      self._bnupdate_loader = []
      iters = 0
      while iters < self._bnupdate_iters:
        for data, target in self._train_loader:
          if iters >= self._bnupdate_iters:
            break
          self._bnupdate_loader.append((data, target))
          iters += 1

  def _init_dset_config(self, dset_config):
    self._dset_config = dset_config
    self._train_loader, self._val_loader = utils.get_dataloader(self._dset_config)

    # init validation loader
    dset_config.setdefault('reinit_val_loader', True)
    if dset_config.reinit_val_loader:
      val_loader = []
      for data, target in self._val_loader:
        val_loader.append((data, target))
  
      self._val_loader = val_loader


  def _init_model_config(self, model_config):
    self._model_config = model_config
    self._model = utils.get_model(self._model_config)
    if hasattr(self._model_config, 'devices'):
      self.model = torch.nn.DataParallel(self._model, self._model_config.devices)
      self.model.to(self._model_config.devices[0])
      self.devices = model_config.devices
    else:
      self.model = self._model
    self._loss = torch.nn.CrossEntropyLoss()

  def _init_optim_config(self, optim_config, prefix = ''):
    if hasattr(optim_config, 'CNS'):
      self._init_optim_config(optim_config.CNS, '_cns')
    if hasattr(optim_config, 'WP'):
      self._init_optim_config(optim_config.WP, '_wp')
    if (not hasattr(optim_config, 'CNS')) and (not hasattr(optim_config, 'WP')):
      if prefix == '':
        self._init_optim_config(optim_config, '_cns') 
      else:
        setattr(self, prefix + '_epochs', optim_config.epochs)
        setattr(self, prefix + '_nmodels_per_iter', optim_config.nmodels_per_iter)
        
        optim_config.setdefault('no_bias_decay', True)
        optim_config.setdefault('momentum', 0.9)
        optim_config.setdefault('mask', False)
        setattr(self, prefix + '_optim_config', optim_config)
    
        optim_config.LR.epochs = optim_config.epochs
        setattr(self, prefix + '_lr_config', optim_config.LR)

  def _create_model(self):
    if hasattr(self._model_config, 'load_model_from'):
      state_dict = torch.load(self._model_config.load_model_from)
      utils.load_state_dict(self._model, state_dict)   
    else:
      self._model.init_weights()

  def _create_optimizer(self, prefix = '_cns'):
    optim_config = getattr(self, prefix + '_optim_config')
    param_groups = utils.get_model_params(self._model, optim_config)
    optimizer = torch.optim.SGD(param_groups, lr = optim_config.LR.lr,
      weight_decay = optim_config.weight_decay, momentum = optim_config.momentum)
    setattr(self, prefix + '_optimizer', optimizer)
    

  def _create_lr_scheduler(self, prefix = '_cns'):
    lr_config = getattr(self, prefix + '_lr_config')
    optimizer = getattr(self, prefix + '_optimizer')
    scheduler = utils.get_lrscheduler(optimizer, lr_config) 
    setattr(self, prefix + '_scheduler', scheduler)


  def train(self, models, prefix = ''):
    if prefix == '':
      self._model.disable_conv_wp()
      self.train(models, '_cns')
      if hasattr(self, '_wp_epochs'):
        self._model.update_importance()
        self._model.enable_conv_wp()
        self.train(models, '_wp')
    else:
      if prefix == '_cns':
        self._create_model()
      self._create_optimizer(prefix)
      self._create_lr_scheduler(prefix)
      scheduler = getattr(self, prefix + '_scheduler') 
      epochs = getattr(self, prefix + '_epochs')
      for e in range(epochs):
        print('EPOCH {} | {}...'.format(e + 1, epochs))
        if scheduler is not None:
          scheduler.step()
        self._train_one_epoch(models, prefix)

  def evaluate(self, models):
    self.train(models)
    self._model.save_bn_running_states()
    return self._test_all_models(models)

  def _train_one_epoch(self, models, prefix = '_cns'):
    nmodels = len(models)
    data_meter = utils.AvgMeter()
    fb_meter = utils.AvgMeter()
    loss_meter = [utils.AvgMeter() for _ in range(nmodels)]
    acc_meter = [utils.AccMeter() for _ in range(nmodels)]
    self.model.train()
    end = time.time()
    optimizer = getattr(self, prefix + '_optimizer')
    for data, target in self._train_loader:
      start = time.time()
      data_meter.update((start - end) * 1000.)
      nmodels_per_iter = getattr(self, prefix + '_nmodels_per_iter')
      ms = np.random.choice(nmodels, nmodels_per_iter, replace = False)
      optimizer.zero_grad()
      self._model.update_importance()
      for m in ms:
        self._model.extract_model(models[m], m)
        out = self.model(data)
        target = target.to(out.device)
        l = self._loss(out, target)
        l.backward()
        loss_meter[m].update(l.item())
        acc_meter[m].update(out, target)
      optimizer.step()
      end = time.time()
      fb_meter.update((end - start) * 1000.)
    print('Data time: {} ms, FB time: {} ms, Total time: {} min.'.format(
      data_meter.avg, fb_meter.avg, (data_meter.sum + fb_meter.sum) / 1000. / 60.
    ))
    print('Loss@Train: {}'.format(' '.join(['{}'.format(l.avg) for l in loss_meter])))
    print('Acc@1@Train: {}'.format(' '.join(['{:.2f}'.format(a.acc1 * 100.) for a in acc_meter])))

  def _test_all_models(self, models):
    ret = []
    for m in range(len(models)):
      ret.append(self._test_model(models[m], m))
    return ret

  #@profile
  def _test_model(self, model, m):
    with torch.no_grad():
      self._model.extract_model(model, m)
      self._model.enable_conv_wp()
      self._update_bn()
      self.model.eval()
      loss_meter = utils.AvgMeter()
      acc_meter = utils.AccMeter()
      loss = torch.nn.CrossEntropyLoss()
      for data, target in self._val_loader:
        out = self.model(data)
        target = target.to(out.device)
        l = loss(out, target)
        loss_meter.update(l.item()) 
        acc_meter.update(out, target)
  
      log = ['model:']
      log.extend(['{:.2f}'.format(m) for m in model])
      log.append('Loss:')
      log.append('{}'.format(loss_meter.avg))
      log.append('AccA1:')
      log.append('{:.2f}'.format(acc_meter.acc1 * 100.))
      print(' '.join(log))
  
      return (loss_meter.avg, acc_meter.acc1 * 100.)

  #@profile
  def _update_bn(self):
    import models

    #
    if self._bnupdate_iters == 0:
      return

    need_bn_sync = False # (hasattr(self, 'devices') and (len(self.devices) > 1))
    bn_modules = [m for m in self._model.modules() if isinstance(m, models.ops.PrunableBN)] 
    running_states = None
    handles = []

    if need_bn_sync:
      for i, bn in enumerate(bn_modules):
        bn.bn_id = i
      device_to_idx = {self.devices[i] : i for i in range(len(self.devices))}
      running_states = [[None] * 2 * len(bn_modules) for _ in self.devices]
      def bn_hook(m, inputs, output):
        rm = m.get_parameter('running_mean')
        rv = m.get_parameter('running_var')
        device_idx = device_to_idx[rm.device.index]
        bn_idx = m.bn_id
        running_states[device_idx][2 * m.bn_id] = rm
        running_states[device_idx][2 * m.bn_id + 1] = rv
        
      for m in bn_modules:
        handle = m.register_forward_hook(bn_hook)
        handles.append(handle)
    
    self._model.load_bn_running_states()
    if self._bnupdate_recompute:
      self._model.reset_bn_running_states()

    if self._bnupdate_mode == 'exact':
      for m in self._model.modules():
        if isinstance(m, models.ops.PrunableBN):
          m.momentum = None
          m.reset_running_states()
    else:
      assert(self._bnupdate_mode == 'running')

    self.model.train()
    
    with torch.no_grad():
      it = 0
      while True:
        if it >= self._bnupdate_iters:
          break
        for data, target in self._bnupdate_loader:
          if it >= self._bnupdate_iters:
            break
          it += 1
          out = self.model(data)
          if need_bn_sync:
            bn_states = torch.cuda.comm.reduce_add_coalesced(running_states, self.devices[0])
            for state in bn_states:
              state /= len(self.devices)

            for m in bn_modules:
              rm = m.get_parameter('running_mean')
              rv = m.get_parameter('running_var')
              rm.copy_(bn_states[2 * m.bn_id])
              rv.copy_(bn_states[2 * m.bn_id + 1])

    for handle in handles:
      handle.remove()
