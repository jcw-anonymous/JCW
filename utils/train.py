from __future__ import division

import os
import math
import time

import torch
import numpy as np

from .eval import test
from .meter import AccMeter, AvgMeter

def train(model, loss, train_loader, val_loader, optimizer, schedular, epochs, steps, checkpoint):
  if epochs > 0 and steps > 0:
    raise ValueError('Only one of epochs and steps should be given.')

  if steps > 0:
    epochs = int(math.ceil(float(steps) / len(train_loader)))

  if checkpoint is not None and not os.path.exists(checkpoint):
    os.makedirs(checkpoint)

  if checkpoint is not None:
    model_pt = os.path.join(checkpoint, 'model.pt')
    model_best_pt = os.path.join(checkpoint, 'model.best.pt')

  best_loss = 0
  best_acc = 0

  for e in range(epochs):
    start = time.time()
    print('EPOCH {} | {}.'.format(e + 1, epochs)) 

    train_loss, train_acc = train_epoch(model, loss, train_loader, optimizer, steps)
    steps -= len(train_loader)
    val_loss, val_acc, _ = test(model, loss, val_loader)

    if schedular is not None:
      schedular.step()

    state = {
      'model': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
      'optimizer': optimizer.state_dict(),
      'acc': val_acc,
      'epoch': e + 1  
    }
    if checkpoint is not None:
      torch.save(state, model_pt)
    if val_acc > best_acc:
      if checkpoint is not None:
        torch.save(state, model_best_pt)
      best_acc = val_acc
      best_loss = val_loss

    end = time.time()
    print('Runtime: {:.2f} min. '
          'Loss: {}[{}]. '
          'Acc: {}[{}].'.format(
            (end - start) / 60.,
            val_loss, train_loss,
            val_acc, train_acc
          ))

  return best_loss, best_acc

def train_epoch(model, loss, train_loader, optimizer, steps):
  model.train()
  data_time = AvgMeter()
  fb_time = AvgMeter()
  loss_meter = AvgMeter()
  acc_meter = AccMeter()

  end = time.time()
  for i, (data, target) in enumerate(train_loader):
    start = time.time()
    data_time.update((start - end) * 1000)
    output = model(data)
    target = target.to(output.device)
    l = loss(output, target)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()
    loss_meter.update(l.item())
    acc_meter.update(output, target)
    end = time.time()
    fb_time.update((end - start) * 1000)
    
    if steps > 0 and i + 1 >= steps:
      break

  print('Data time: {:.2f} ms. '
        'FB time: {:.2f} ms. '.format(
          data_time.avg, fb_time.avg
        ))

  return loss_meter.avg, acc_meter.acc1
