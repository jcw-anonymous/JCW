import math

import torch
import numpy as np

if __name__ == '__main__':
  import ops
else:
  from . import ops

def conv3x3(inc, outc, stride = 1, group_size = 1):
  return ops.PrunableConv(inc, outc, 3, stride = stride, padding = 1, bias = False, group_size = group_size)

class BasicBlock(torch.nn.Module):
  expansion = 1
  nconvs = 2
  def __init__(self, inplanes, planes, stride = 1, downsample = None, bn_share_scale = True, bn_share_mv = True, nmodels = 1, group_size = 4):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride, group_size = group_size)
    self.bn1 = ops.PrunableBN(planes, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels)
    self.relu = torch.nn.ReLU(inplace = True)
    self.conv2 = conv3x3(planes, planes, group_size = group_size)
    self.bn2 = ops.PrunableBN(planes, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)
    out = self.conv2(out)
    out = self.bn2(out) 

    if self.downsample is not None:
      residual = self.downsample(x)

    out = out + residual
    out = self.relu(out)

    return out


  def set_channels(self, inc, midc, outc):
    self.conv1.set_input_output_channels(inc, midc)
    self.bn1.set_num_features(midc)
    self.conv2.set_input_output_channels(midc, outc)
    self.bn2.set_num_features(outc)
    if self.downsample is not None:
      self.downsample[0].set_input_output_channels(inc, outc)
      self.downsample[1].set_num_features(outc)


# Currently support resnet18 only
class ResNet(ops.PrunableModel):
  def __init__(self, block, layers, num_classes = 1000, bn_share_scale = True, bn_share_mv = True, nmodels = 1, group_size = 4):
    super(ResNet, self).__init__()

    self.bn_share_scale = bn_share_scale
    self.bn_share_mv = bn_share_mv
    self.nmodels = nmodels
    self.group_size = group_size
    
    self.conv1 = ops.PrunableConv(3, 64, kernel_size = 7, stride = 2, padding = 3,
                                  bias = False, group_size = group_size)
    self.bn1 = ops.PrunableBN(64, share_scale = bn_share_scale, share_mv = bn_share_mv, nmodels = nmodels)
    self.relu = torch.nn.ReLU(inplace = True)
    self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
    self.inplanes = 64
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)
    # self.avgpool = torch.nn.AvgPool2d(7)
    # modify oedinary pooling to adaptive pooling to support more input image sizes.
    self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    self.fc = ops.PrunableLinear(512 * block.expansion, num_classes)

    self.full_channels = []
    for name, m in self.named_modules():
      if isinstance(m, ops.PrunableConv) and 'downsample' not in name:
        self.full_channels.append(m.weight.shape[0])

    self.init_weights()

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, ops.PrunableConv):
        out_channels = m.weight.shape[0]
        n = m.kernel_size[0] * m.kernel_size[1] * out_channels
        m.weight.data.normal_(0, np.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, ops.PrunableLinear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
      elif isinstance(m, ops.PrunableBN):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
        m.reset_running_states()

  def _make_layer(self, block, planes, blocks, stride = 1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = torch.nn.Sequential(
        ops.PrunableConv(self.inplanes, planes * block.expansion, 1,
                         stride = stride, bias = False, group_size = self.group_size),
        ops.PrunableBN(planes * block.expansion, share_scale = self.bn_share_scale, share_mv = self.bn_share_mv, nmodels = self.nmodels)
      )
    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, bn_share_scale = self.bn_share_scale, bn_share_mv = self.bn_share_mv, nmodels = self.nmodels, group_size = self.group_size))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes, bn_share_scale = self.bn_share_scale, bn_share_mv = self.bn_share_mv, nmodels = self.nmodels, group_size = self.group_size))

    return torch.nn.Sequential(*layers)


  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    x = self.avgpool(x)
    x = x.flatten(1)
    x = self.fc(x)

    return x

  def cns_feature_dims(self):
    # The first convolution layer
    dims = 1

    def stage_cns_feature_dims(stage):
      nonlocal dims

      if stage[0].downsample is not None:
        # an extra dim for residual  
        dims += 1
      dims += len(stage)

    stage_cns_feature_dims(self.layer1)
    stage_cns_feature_dims(self.layer2)
    stage_cns_feature_dims(self.layer3)
    stage_cns_feature_dims(self.layer4)

    return dims

  def set_channels(self, channels):
    # The first convolution layer
    self.conv1.set_input_output_channels(3, channels[0])
    self.bn1.set_num_features(channels[0])

    channel_idx = 1
    def stage_set_channels(stage):
      nonlocal channel_idx
      nblocks = len(stage)
      for i in range(nblocks):
        inc = channels[channel_idx - 1]
        midc = channels[channel_idx]
        outc = channels[channel_idx + 1]
        stage[i].set_channels(inc, midc, outc)
        channel_idx += 2

    stage_set_channels(self.layer1) 
    stage_set_channels(self.layer2) 
    stage_set_channels(self.layer3) 
    stage_set_channels(self.layer4)

    self.fc._set_input_features(channels[-1])

  def feature_to_channels(self, feature):
    assert(len(feature) == self.cns_feature_dims())

    def _make_divisible(full_channels, ratio, divisor, min_value = None):
      if min_value is None:
        min_value = divisor

      channels = int(math.ceil(full_channels // divisor * ratio) * divisor)
      channels = min(full_channels, max(min_value, channels))
      return channels

    channels = self.full_channels.copy()
    # The first convolution layer
    channels[0] = _make_divisible(self.full_channels[0], feature[0], self.group_size)
    
    channel_idx = 1
    feature_idx = 1
    def stage_feature_to_channels(stage):
      nonlocal channel_idx, feature_idx
      nblocks = len(stage)

      # residual channels
      full_residual_channels = self.full_channels[channel_idx + 1]
      if stage[0].downsample is not None:
        residual_channels = _make_divisible(full_residual_channels, feature[feature_idx], self.group_size)
        feature_idx += 1
      else:
        residual_channels = channels[channel_idx - 1]

      for i in range(nblocks):
        channels[channel_idx + 1 + i * 2] = residual_channels

      # middle channels
      for i in range(nblocks):
        full_channels = self.full_channels[channel_idx]
        ratio = feature[feature_idx]
        c = _make_divisible(full_channels, ratio, self.group_size)
        channels[channel_idx] = c
        feature_idx += 1 
        channel_idx += 2

    stage_feature_to_channels(self.layer1)
    stage_feature_to_channels(self.layer2)
    stage_feature_to_channels(self.layer3)
    stage_feature_to_channels(self.layer4)

    return channels

  def _extract_model(self, feature):
    channels = self.feature_to_channels(feature)
    self.set_channels(channels)


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
    # Strided convolution
    if hasattr(cns_config, 'strided'):
      # The first convolution layer is strided convolution layer
      _update(cns_bound, 0, cns_config.strided)
      feature_idx = 1
      def stage_set_strided_bound(stage):
        nonlocal feature_idx
        nblocks = len(stage)
        if stage[0].stride != 1:
          _update(cns_bound, feature_idx, cns_config.strided)
          _update(cns_bound, feature_idx + 1, cns_config.strided)
        feature_idx += nblocks
        if stage[0].downsample is not None:
          feature_idx += 1
      stage_set_strided_bound(self.layer1)
      stage_set_strided_bound(self.layer2)
      stage_set_strided_bound(self.layer3)
      stage_set_strided_bound(self.layer4)
    # Residual convolution
    if hasattr(cns_config, 'residual'):
      feature_idx = 1
      def stage_set_residual_bound(stage):
        nonlocal feature_idx
        nblocks = len(stage)
        if stage[0].downsample is None:
          # bound the input channels
          _update(cns_bound, feature_idx - 1, cns_config.residual)
          feature_idx += nblocks
        else:
          _update(cns_bound, feature_idx, cns_config.residual)
          feature_idx += (nblocks + 1)
      stage_set_residual_bound(self.layer1)
      stage_set_residual_bound(self.layer2)
      stage_set_residual_bound(self.layer3)
      stage_set_residual_bound(self.layer4)

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
    named_convs = [(name, m) for name, m in self.named_modules() if isinstance(m, ops.PrunableConv)]
    if hasattr(wp_config, 'strided'):
      for i, (_, m) in enumerate(named_convs):
        if m.stride[0] != 1:
          _update(wp_bound, i, wp_config.strided)
    # Downsample
    if hasattr(wp_config, 'downsample'):
      for i, (name, m) in enumerate(named_convs):
        if 'downsample' in name:
          _update(wp_bound, i, wp_config.downsample)
  
    return np.array(cns_bound + wp_bound)      


def resnet18(*args, **kwargs):
  if len(args) < 2:
    kwargs.update(
      layers = [2, 2, 2, 2]
    )
  if len(args) < 1:
    kwargs.update(
      block = BasicBlock 
    )

  return ResNet(*args, **kwargs)



if __name__ == '__main__':
  model = resnet18(group_size = 4)

  feature = np.random.choice(np.linspace(0.1, 1, 10), size = (32, model.cns_feature_dims()))

  x = torch.randn(1, 3, 224, 224)

  for f in feature:
    model.extract_model(f)
    out = model(x)
    print(model)
