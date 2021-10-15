from . import ops

from .mbv1 import MobileNetV1
from .mbv2 import MobileNetV2
from .resnet_imagenet import resnet18

def save_bn_running_states(model):
  for m in model.modules():
    if isinstance(m, ops.PrunableBN):
      m.save_running_states()

def load_bn_running_states(model):
  for m in model.modules():
    if isinstance(m, ops.PrunableBN):
      m.load_running_states()

def reset_bn_running_states(model):
  for m in model.modules():
    if isinstance(m, ops.PrunableBN):
      m.reset_running_states()
