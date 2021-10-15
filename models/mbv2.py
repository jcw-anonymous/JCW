import math

import numpy as np
import torch

from . import ops

class InvertedResidual(torch.nn.Module):
  def __init__(self, inc, outc, stride, expansion_ratio, group_size = 4, bn_share_scale = True, bn_share_mv = True, nmodels = 1):
    super(InvertedResidual, self).__init__()
    
    ext_c = inc * expansion_ratio

    if expansion_ratio == 1:
      self.conv = torch.nn.Sequential(
        # dw conv
        ops.PrunableConv(inc, inc, 3, stride = stride, padding = 1, groups = inc, bias = False),
        ops.PrunableBN(inc, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels),
        torch.nn.ReLU6(),

        # pw conv
        ops.PrunableConv(inc, outc, 1, bias = False, group_size = group_size),
        ops.PrunableBN(outc)
      )
      self.expansion = False
    else:
      self.conv = torch.nn.Sequential(
        # pw expansion
        ops.PrunableConv(inc, ext_c, 1, bias = False, group_size = group_size),
        ops.PrunableBN(ext_c, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels),
        torch.nn.ReLU6(),

        # dw conv
        ops.PrunableConv(ext_c, ext_c, 3, stride = stride, padding = 1, groups = ext_c, bias = False),
        ops.PrunableBN(ext_c, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels),
        torch.nn.ReLU6(),

        # pw conv
        ops.PrunableConv(ext_c, outc, 1, bias = False, group_size = group_size),
        ops.PrunableBN(outc, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels)
      )
      self.expansion = True

    self.residual = (inc == outc and stride == 1)

  def forward(self, x):
    res = x
    x = self.conv(x)

    if self.residual:
      x = res + x

    return x

  def set_channels(self, inc, midc, outc):
    if not self.expansion:
      # dwconv | bn | relu6 | conv | bn
      self.conv[0].set_input_output_channels(inc, midc)
      self.conv[1].set_num_features(midc)
      self.conv[3].set_input_output_channels(midc, outc)
      self.conv[4].set_num_features(outc)
    else:
      # conv | bn | relu6 | dwconv | bn | relu6 | conv | bn
      self.conv[0].set_input_output_channels(inc, midc)
      self.conv[1].set_num_features(midc)
      self.conv[3].set_input_output_channels(midc, midc)
      self.conv[4].set_num_features(midc)
      self.conv[6].set_input_output_channels(midc, outc)
      self.conv[7].set_num_features(outc)

class MobileNetV2(ops.PrunableModel):
  def __init__(self, num_classes = 1000, bn_share_scale = True, bn_share_mv = True, nmodels = 1, group_size = 4):
    super(MobileNetV2, self).__init__()

    self.group_size = group_size
    self.bn_share_scale = bn_share_scale
    self.bn_share_mv = bn_share_mv    

    self.residual_params = [
      #t  c   n  s
      [1, 16, 1, 1],
      [6, 24, 2, 2],
      [6, 32, 3, 2],
      [6, 64, 4, 2],
      [6, 96, 3, 1],
      [6, 160, 3, 2],
      [6, 320, 1, 1]
    ]

    feature = []
    self.full_channels = []
    
    # The first convolution layer
    feature.append(torch.nn.Sequential(
      ops.PrunableConv(3, 32, 3, stride = 2, padding = 1, bias = False, group_size = group_size),
      ops.PrunableBN(32, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels),
      torch.nn.ReLU6()
    ))
    self.inc = 32

    def _make_residual(t, c, n, s):
      blocks = []
      for i in range(n):
        blocks.append(InvertedResidual(self.inc, c, s if i == 0 else 1, t, bn_share_scale = bn_share_scale, bn_share_mv = bn_share_mv, nmodels = nmodels, group_size = group_size))
        self.inc = c  
      blocks = torch.nn.Sequential(*blocks)
      feature.append(blocks)

    for t, c, n, s in self.residual_params:
      _make_residual(t, c, n, s)

    feature.append(torch.nn.Sequential(
      ops.PrunableConv(self.inc, 1280, 1, bias = False, group_size = group_size),
      ops.PrunableBN(1280, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels),
      torch.nn.ReLU6()
    ))

    self.feature = torch.nn.Sequential(*feature)

    self.full_channels = [m.out_channels for m in self.modules() if isinstance(m, ops.PrunableConv) and not m.is_dw]

    # self.pool = torch.nn.AvgPool2d(7)
    # modify ordinary pooling to adaptive pooling to support more input image sizes.
    self.pool = torch.nn.AdaptiveAvgPool2d(1)
    self.classifier = ops.PrunableLinear(1280, num_classes)

    self.init_weights()

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, ops.PrunableConv):
        n = np.prod(m.kernel_size) * m.weight.data.shape[0]
        m.weight.data.normal_(0, np.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, ops.PrunableBN):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        m.reset_running_states()
      elif isinstance(m, ops.PrunableLinear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_() 

  def forward(self, x):
    x = self.feature(x)
    x = self.pool(x)
    x = x.flatten(1)
    x = self.classifier(x)

    return x

  def cns_feature_dims(self):
    # feature dims for CNS(channel number search)
    
    ## The first convolution
    dims = 1
    
    ## Feature dims of inverted residual blocks
    def block_dims(block):
      nonlocal dims
      nres_blocks = len(block)
      if not block[0].residual:
        dims += 1
      
      if block[0].expansion:
        dims += nres_blocks

    for block in self.feature[1:-1]:
      block_dims(block)

    ## Last convolution
    dims += 1

    return dims

  def set_channels(self, channels):
    # The first convolution
    self.feature[0][0]._set_output_channels(channels[0])
    self.feature[0][1].set_num_features(channels[0])

    channel_idx = 1
    def set_block_channels(block):
      nonlocal channel_idx
      nres_blocks = len(block)
      for res in block:
        inc = channels[channel_idx - 1] 
        if not res.expansion:
          midc = inc
          outc = channels[channel_idx]
          channel_idx += 1
        else:
          midc = channels[channel_idx]
          outc = channels[channel_idx + 1]
          channel_idx += 2
        res.set_channels(inc, midc, outc)

    # inverted residual blocks
    for block in self.feature[1:-1]:
      set_block_channels(block)

    # the last convolution
    inc = channels[channel_idx - 1]
    self.feature[-1][0].set_input_output_channels(inc, channels[-1])
    self.feature[-1][1].set_num_features(channels[-1])

    # the classifier 
    self.classifier._set_input_features(channels[-1])


  def feature_to_channels(self, feature):
    # from feature to number of channels

    def _make_divisible(full_channels, ratio, divisor, min_value = None):
      if min_value is None:
        min_value = divisor

      channels = int(math.ceil(full_channels // divisor * ratio) * divisor)
      channels = min(full_channels, max(min_value, channels))
      return channels

    channels = self.full_channels.copy()
    ## The first convolution
    channels[0] = _make_divisible(self.full_channels[0], feature[0], self.group_size)
    
    feature_idx = 1
    channel_idx = 1
    def block_feature_to_channels(block):
      nonlocal feature_idx, channel_idx
      nres_blocks = len(block)
      if not block[0].expansion:
        full_residual_channels = self.full_channels[channel_idx]
      else:
        full_residual_channels = self.full_channels[channel_idx + 1]
      # residual channel numbers
      if not block[0].residual:
        residual_channels = _make_divisible(full_residual_channels, feature[feature_idx], self.group_size)
        feature_idx += 1
      else:
        residual_channels = channels[channel_idx - 1]

      # set residual channels1
      if not block[0].expansion:
        residual_interval = 1
        residual_id_start = channel_idx
      else:
        residual_interval = 2
        residual_id_start = channel_idx + 1
      for i in range(nres_blocks):
        channels[residual_id_start + i * residual_interval] = residual_channels

      # set middle channels
      if block[0].expansion:
        for i in range(nres_blocks):
          cid = channel_idx + i * 2
          full_channels = self.full_channels[cid]
          channels[cid] = _make_divisible(full_channels, feature[feature_idx], self.group_size)
          feature_idx += 1

      # update channel idx
      if block[0].expansion:
        channel_idx += (2 * nres_blocks)
      else:
        channel_idx += nres_blocks

    for block in self.feature[1:-1]:
      block_feature_to_channels(block)

    # The last convolution layers
    channels[-1] = _make_divisible(self.full_channels[-1], feature[-1], self.group_size)

    return channels

  def _extract_model(self, feature):
    assert(len(feature) == self.cns_feature_dims()) 
    channels = self.feature_to_channels(feature)
    self.set_channels(channels)    

  def conv_info(self, input_shape):

    if hasattr(self, 'info'):
      info = self.info.copy()
      out_channels = [m.out_channels for m in self.modules() if isinstance(m, ops.PrunableConv) and not m.is_dw]
      for i in range(len(info)):
        # ic ih oc k s with_dw
        info[i][2] = out_channels[i]
        if i > 0:
          info[i][0] = out_channels[i-1]
      return info

    info = super(MobileNetV2, self).conv_info(input_shape)

    # ic ih oc k s relu

    infoid = 1
    for t, c, n, s in self.residual_params: 
      # update stride
      if t == 1:
        # without expansion
        info[infoid][4] = s
      else:
        info[infoid + 1][4] = s

      # update with dw
      if t == 1:
        infoid_start = infoid
        infoid_interval = 1
      else:
        infoid_start = infoid + 1
        infoid_interval = 2
      for i in range(n):
        info[infoid_start + i * infoid_interval][5] = True

      # update infoid
      if t == 1:
        infoid += n
      else:
        infoid += 2 * n

    self.info = info

    return info


  def _parse_bound(self, bound):
    assert(hasattr(bound, 'CNS') and hasattr(bound, 'WP'))

    def _update(data, idx, value):  
      data[idx] = max(data[idx], value) 
 
    cns_config = bound.CNS
    cns_feature_dims = self.cns_feature_dims()
    cns_bound = [cns_config.bound] * cns_feature_dims
    # The first layer
    if hasattr(cns_config, 'fy'):
      _update(cns_bound, 0, cns_config.fy)
    # The last layer
    if hasattr(cns_config, 'ly'):
      _update(cns_bound, -1, cns_config.ly)
    # Strided layers
    if hasattr(cns_bound, 'strided'):
      # The first convolution layer is strided convolution
      _update(cns_bound, 0, cns_config.strided)
      boundid = 1
      for t, c, n, s in self.residual_params:
        if s != 1 and t != 1:
          _update(cns_bound, boundid, cns_config.strided)
       
        boundid += 1
        if t != 1:
          boundid += n 
    # Residuals
    if hasattr(cns_config, 'residual'):
      boundid = 1
      for t, c, n, s in self.residual_params:
        _update(cns_bound, boundid, cns_config.residual)
        boundid += 1
        if t != 1:
          boundid += n

    wp_config = bound.WP
    wp_feature_dims = self.wp_feature_dims()
    wp_bound = [wp_config.bound] * wp_feature_dims
    # The first layer
    if hasattr(wp_config, 'fy'):
      _update(wp_bound, 0, wp_config.fy)
    # The last layer
    if hasattr(wp_config, 'ly'):
      _update(wp_bound, -1, wp_config.ly)
    # Strided convolution
    if hasattr(wp_config, 'strided'):
      info = self.conv_info([3, 224, 224])
      for i in range(len(info)):
        if info[i][4] != 1:
          _update(wp_bound, i, wp_config.strided)

    return np.array(cns_bound + wp_bound) 
