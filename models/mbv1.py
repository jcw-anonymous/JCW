import math

import torch
import numpy as np

from . import ops

class MobileNetV1(ops.PrunableModel):
  def __init__(self, num_classes = 1000, configs = None, bn_share_scale = True, bn_share_mv = True, nmodels = 1, group_size = 4):
    super(MobileNetV1, self).__init__()

    self.bn_share_scale = bn_share_scale
    self.bn_share_mv = bn_share_mv
    self.nmodels = nmodels
    self.group_size = group_size

    def conv_bn(inc, outc, stride):
      return torch.nn.Sequential(
        ops.PrunableConv(inc, outc, 3, stride, 1, bias = False, group_size = group_size),
        ops.PrunableBN(outc, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels),
        torch.nn.ReLU(inplace = True)
      ) 


    def conv_dw(inc, outc, stride):
      return torch.nn.Sequential(
        ops.PrunableConv(inc, inc, 3, stride, 1, groups = inc, bias = False),
        ops.PrunableBN(inc, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels),
        torch.nn.ReLU(),

        ops.PrunableConv(inc, outc, 1, 1, 0, bias = False, group_size = group_size),
        ops.PrunableBN(outc, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels),
        torch.nn.ReLU()
      )

    self.full_channels = [32, 64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2
    self.strides = [2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1] 

    layers = []
    inc = 3
    for channels, stride in zip(self.full_channels, self.strides):
      if len(layers) == 0:
        layers.append(conv_bn(inc, channels, stride)) 
      else:
        layers.append(conv_dw(inc, channels, stride))

      inc = channels

    self.feature = torch.nn.Sequential(*layers)
    # self.pool = torch.nn.AvgPool2d(7)
    # modify ordinary pooling to adaptive pooling to support more input image size
    self.pool = torch.nn.AdaptiveAvgPool2d(1)
    self.fc = ops.PrunableLinear(1024, num_classes)

  def forward(self, x):
    x = self.feature(x)
    x = self.pool(x)
    x = x.flatten(1)
    x = self.fc(x)
    return x

  def cns_feature_dims(self):
    return len(self.feature) 

  def _extract_model(self, feature):
    channels = self.feature_to_channels(feature)
    self.set_channels(channels)

  def feature_to_channels(self, feature):
    assert(len(feature) == self.cns_feature_dims())

    def _make_divisible(full_channels, ratio, divisor, min_value = None):
      if min_value is None:
        min_value = divisor
      
      channels = int(math.ceil(full_channels // divisor * ratio) * divisor)
      channels = min(full_channels, max(min_value, channels))

      return channels

    channels = []
    for f, outc in zip(feature, self.full_channels):
      channels.append(_make_divisible(outc, f, self.group_size))
    return channels

  def set_channels(self, channels):
    assert(len(channels) == len(self.feature))

    inc = 3
    for outc, stage in zip(channels, self.feature):
      if len(stage) == 3:
        # conv bn relu
        stage[0].set_input_output_channels(inc, outc)
        stage[1].set_num_features(outc)
      else:
        assert(len(stage) == 6)
        # dw bn relu conv bn relu
        stage[0].set_input_output_channels(inc, inc)
        stage[1].set_num_features(inc)
        stage[3].set_input_output_channels(inc, outc)
        stage[4].set_num_features(outc)
      inc = outc
    self.fc._set_input_features(inc)

  def _parse_bound(self, bound):
    assert(hasattr(bound, 'CNS') and hasattr(bound, 'WP'))

    cns_config = bound.CNS
    cns_feature_dims = self.cns_feature_dims()
    cns_bound = [cns_config.bound] * cns_feature_dims
    # The first layer
    if hasattr(cns_config, 'fy'):
      cns_bound[0] = max(cns_bound[0], cns_config.fy)
    # The last layer
    if hasattr(cns_config, 'ly'):
      cns_bound[-1] = max(cns_bound[-1], cns_config.ly)
    # Strided convolutions
    if hasattr(cns_config, 'strided'):
      # The first convolution layer is stried conv
      cns_bound[0] = max(cns_bound[0], cns_config.strided)
      for i, stride in enumerate(self.strides):
        if stride != 1 and i > 0:
          cns_bound[i - 1] = max(cns_bound[i - 1], cns_config.strided)

    wp_config = bound.WP
    wp_feature_dims = self.wp_feature_dims()
    wp_bound = [wp_config.bound] * wp_feature_dims
    # The first layer
    if hasattr(wp_config, 'fy'):
      wp_bound[0] = max(wp_bound[0], wp_config.fy)
    # The last layer
    if hasattr(wp_config, 'ly'):
      wp_bound[-1] = max(wp_bound[-1], wp_config.ly)

    return np.array(cns_bound + wp_bound)

  def conv_info(self, input_shape):
    info = super(MobileNetV1, self).conv_info(input_shape)
    assert(len(info) == len(self.strides)) 

    for i in range(len(self.strides)):
      info[i][4] = self.strides[i]
      if i > 0:
        info[i][5] = True

    return info
