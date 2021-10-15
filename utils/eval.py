import time

import torch
import numpy as np

from .meter import AvgMeter, AccMeter

def test(model, loss, val_set):
  model.eval()
  loss_meter = AvgMeter()
  acc_meter = AccMeter()

  for data, label in val_set:
    y = model(data)
    label = label.to(y.device)
    l = loss(y, label)
    loss_meter.update(l.item())
    acc_meter.update(y, label)

  return loss_meter.avg, acc_meter.acc1, acc_meter.acc5
