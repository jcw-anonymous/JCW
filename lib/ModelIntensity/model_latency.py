from scipy.interpolate import interpn

from .model_intensity import ModelIntensity

import numpy as np

import models

class ModelLatency(ModelIntensity):
  def __init__(self, config):
    self._runtime = np.load(config.runtime)

  def evaluate(self, model):
    convs = [m for m in model.modules() if isinstance(m, models.ops.PrunableConv) and not m.is_dw]
    assert(len(convs) == len(self._runtime))

    def _make_list(c, N):
      ret = set()
      for i in range(1, N):
        ret.add(c // N * i)
      ret.add(c)
      ret = list(ret)
      ret.sort()
      if len(ret) < N:
        ret += [c + i for i in range(1, N - len(ret) + 1)]
      return ret
      

    self._val = 0
    for i in range(len(convs)):
      full_oc, full_ic, _, _ = convs[i].weight.shape 
      _, noc, nic, nratios = self._runtime.shape
      _ocs = [0] + _make_list(full_oc, noc - 1)
      _ics = [0] + _make_list(full_ic, nic - 1)
      if _ics[1] == 0:
        _ics[0] = -1
      if _ocs[1] == 0:
        _ocs[0] = -1
      _ratios = np.linspace(0, 1, nratios)

      oc = convs[i].out_channels
      ic = convs[i].in_channels
      ratio = convs[i].wp_density

      val = interpn((_ocs, _ics, _ratios), self._runtime[i], [oc, ic, ratio])[0]

      self._val += val 

    return self._val
