from collections import OrderedDict
import time

import torch
import torch.utils.hooks as hooks

import utils

class Trainer(object):

  registry = {}

  def __init__(self):
    self._pre_epoch_hooks = OrderedDict()
    self._pre_forward_hooks = OrderedDict()
    self._pre_loss_hooks = OrderedDict()
    self._pre_backward_hooks = OrderedDict()
    self._pre_step_hooks = OrderedDict()
    self._step_hooks = OrderedDict()
    self._epoch_hooks = OrderedDict()
    self._stop_hooks = OrderedDict()

  def _init_config_info(self, config):
    if hasattr(config, 'MODEL'):
      self._init_model_config(config.MODEL)
    else:
      assert(hasattr(self, '_model'))

    self._init_optim_config(config.OPTIM)
    self._init_dset_config(config.DATASET)
     
  def _init_model_config(self, config):
    self._model = utils.get_model(config)
    # create forward model
    self.model = self._model
    if hasattr(config, 'devices'):
      self.model = torch.nn.DataParallel(self._model, config.devices)
      self._model.to(config.devices[0])

    # extract submodel, if given
    if hasattr(config, 'EXTRACT'):
      print('Extracting model: {}'.format(config.EXTRACT.feature))
      self._extract_model(config.EXTRACT)

  def _extract_model(self, config):
    def submodel(feature):
      self._feature = feature
      if all([0 < f and f <= 1.0 for f in feature]):
        self._model.extract_model(feature)
      elif all([isinstance(f, int) for f in feature]):
        self._model.set_channels(feature)

    if isinstance(config.feature, (list, tuple)):
      submodel(config.feature)
    else:
      ck = torch.load(config.feature.load_from)
      feature = ck['models'][config.feature.index]
      submodel(feature)

  def _init_optim_config(self, config):
    self._epochs = config.epochs
    config.setdefault('momentum', 0.9)
    params = utils.get_model_params(self._model, config)
    self._optimizer = torch.optim.SGD(params, lr = config.LR.lr, weight_decay = config.weight_decay, momentum = config.momentum)
    config.LR.epochs = self._epochs
    self._scheduler = utils.get_lrscheduler(self._optimizer, config.LR)

    # loss
    # Label smooth loss is equivelent to crossenstropy loss
    # when label smooth is set to 0.0
    config.setdefault('label_smooth', 0.0)
    self._label_smooth = config.label_smooth
    self._loss_func = utils.LabelSmoothKLDivLoss

    if self._label_smooth > 0:
      print('Using label smooth')

  def _init_dset_config(self, config):
    self._train_loader, self._val_loader = utils.get_dataloader(config)


  def _register_hook(self, hook, prefix):
    hook_dict = getattr(self, prefix + '_hooks')
    handle = hooks.RemovableHandle(hook_dict)
    hook_dict[handle.id] = hook
    return handle


  def _train_one_epoch(self):
    self._model.train()
    data_meter = utils.AvgMeter()
    fb_meter = utils.AvgMeter()
    loss_meter = utils.AvgMeter()
    acc_meter = utils.AccMeter()

    end = time.time()
    for i, (data, target) in enumerate(self._train_loader):
      start = time.time()
      data_meter.update((start - end) * 1000.)
      for hook in self._pre_forward_hooks.values():
        hook(self)
      self._optimizer.zero_grad()
      out = self.model(data)
      target = target.to(out.device)
      for hook in self._pre_loss_hooks.values():
        hook(self)
      l = self._loss_func(out, target, self._label_smooth)
      for hook in self._pre_backward_hooks.values():
        hook(self)
      l.backward()
      for hook in self._pre_step_hooks.values():
        hook(self)
      self._optimizer.step()
      for hook in self._step_hooks.values():
        hook(self)
      acc_meter.update(out, target)
      loss_meter.update(l.item())
      end = time.time()
      fb_meter.update((end - start) * 1000)


    print('Data time: {:.2f} ms.' \
          'FB time: {:.2f} ms. Loss: {}'.format(
          data_meter.avg, fb_meter.avg, loss_meter.avg
          ))

    return loss_meter.avg, acc_meter.acc1, acc_meter.acc5

  def test(self):
    with torch.no_grad():
      self._model.eval()
      loss = torch.nn.CrossEntropyLoss()
      self._loss, self._acc1, self._acc5 = utils.test(self.model, loss, self._val_loader)

  def run(self):
    for e in range(self._epoch, self._epochs):
      self._epoch = e
      for hook in self._pre_epoch_hooks.values():
        hook(self)

      train_loss, train_acc1, train_acc5 = self._train_one_epoch()
      
      for hook in self._epoch_hooks.values():
        hook(self, train_loss, train_acc1, train_acc5)

    for hook in self._stop_hooks.values():
      hook(self)
