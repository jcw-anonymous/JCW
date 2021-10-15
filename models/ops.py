import torch
import torch.nn.functional as F
import numpy as np

class PrunableConv(torch.nn.Conv2d):
  def __init__(self, *args, **kwargs):
    self.group_size = 4
    if 'group_size' in kwargs:
      self.group_size = kwargs['group_size']
      del kwargs['group_size']

    super(PrunableConv, self).__init__(*args, **kwargs)
    self.is_dw = (self.in_channels == self.groups and self.out_channels == self.groups)
    if not self.is_dw:
      assert (self.groups == 1)
      assert (self.out_channels % self.group_size == 0)
    self.wp_enabled = False
    # wp_density is the density for weight pruning
    self.wp_density = 1.0
    oc, ic, kh, kw = self.weight.shape
    # self.register_buffer('importance', torch.Tensor(self.out_channels // self.group_size, ic * kh * kw)) 
    self.register_buffer('importance', None) 

  def _set_input_channels(self, in_channels):
    self.in_channels = in_channels

  def _set_output_channels(self, out_channels):
    assert(out_channels % self.group_size == 0)
    self.out_channels = out_channels

  def set_input_output_channels(self, in_channels, out_channels):
    if self.is_dw:
      assert (in_channels == out_channels)
      self.groups = in_channels
    self._set_input_channels(in_channels)
    self._set_output_channels(out_channels)

  def update_importance(self):
    assert(not self.is_dw)
    weight = self.weight.data
    oc, ic, kh, kw = weight.shape
    weight_flat = weight.reshape(oc // self.group_size, self.group_size, -1)
    importance = weight_flat.norm(p = 1, dim = 1)
    self.importance = importance

  def forward(self, input):
    if self.bias is not None:
      b = self.bias[:self.out_channels]
    else:
      b = None

    if not self.is_dw:
      w = self.weight[:self.out_channels, :self.in_channels, :, :]
      if self.wp_enabled:
        oc, ic, kh, kw = self.weight.shape
        imp_out_channels = self.out_channels // self.group_size
        imp = self.importance.view(oc // self.group_size, ic, kh, kw)
        imp = imp[:imp_out_channels, :self.in_channels, :, :].flatten()
        mask_flat = torch.zeros_like(imp)
        nremained = int(round(self.wp_density * imp.numel()))
        sorted_idx = imp.argsort(descending = True)
        mask_flat[sorted_idx[:nremained]] = 1
        mask = mask_flat.reshape(imp_out_channels, 1, -1)
        w = w.reshape(imp_out_channels, self.group_size, -1)
        w = w * mask
        w = w.reshape(self.out_channels, self.in_channels, kh, kw)
    else:
      w = self.weight[:self.out_channels, :, :, :]

    return F.conv2d(input, w, b, self.stride, self.padding, 
                    self.dilation, self.groups)

class PrunableBN(torch.nn.Module):
  def __init__(self, num_features, eps = 1e-5,
               momentum = 0.1,
               share_scale = True,
               share_mv = True,
               nmodels = 1):
    super(PrunableBN, self).__init__()
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.share_scale = share_scale
    self.share_mv = share_mv
    self.nmodels = nmodels
    self.model_idx = 0

    if self.share_scale:
      self.weight = torch.nn.Parameter(torch.Tensor(num_features))
      self.bias = torch.nn.Parameter(torch.Tensor(num_features))
    else:
      self.weight = torch.nn.Parameter(torch.Tensor(nmodels, num_features))
      self.bias = torch.nn.Parameter(torch.Tensor(nmodels, num_features))

    if self.share_mv:
      self.register_buffer('running_mean', torch.zeros(num_features))
      self.register_buffer('running_var', torch.ones(num_features))
      self.register_buffer('num_batches_tracked', torch.tensor(0, dtype = torch.long))
    else:
      self.register_buffer('running_mean', torch.zeros(nmodels, num_features))
      self.register_buffer('running_var', torch.ones(nmodels, num_features))
      self.register_buffer('num_batches_tracked', torch.zeros(nmodels, dtype = torch.long))

    self.reset_parameters()

  def reset_parameters(self):
    with torch.no_grad():
      self.weight.fill_(1)
      self.bias.zero_() 

  def get_parameter(self, name):
    tensor = getattr(self, name)
    shared =  (name == 'num_batches_tracked' and tensor.size(0) > 1) \
           or (name != 'num_batches_tracked' and len(tensor.shape) > 1)

    if not shared:
      tensor = tensor[self.model_idx]

    if name == 'num_batches_tracked':
      return tensor
    else:
      return tensor[:self.num_features]
   
  def set_model_index(self, model_idx = 0):
    assert(model_idx < self.nmodels)
    self.model_idx = model_idx

  def set_num_features(self, num_features):
    self.num_features = num_features 


  def reset_running_states(self):
    with torch.no_grad():
      self.running_mean.zero_()
      self.running_var.fill_(1)
      self.num_batches_tracked.zero_()

  def save_running_states(self, clone = True):
    if clone:
      self.running_states = {
        'running_mean': self.running_mean.clone(),
        'running_var': self.running_var.clone(),
        'num_batches_tracked': self.num_batches_tracked.clone()
      }
    else:
      self.running_states = {
        'running_mean': self.running_mean,
        'running_var': self.running_var,
        'num_batches_tracked': self.num_batches_tracked
      }

  def load_running_states(self):
    with torch.no_grad():
      self.running_mean.copy_(self.running_states['running_mean'])
      self.running_var.copy_(self.running_states['running_var'])
      self.num_batches_tracked.copy_(self.running_states['num_batches_tracked'])


  def forward(self, x):
    exponential_average_factor = 0.0
    if self.share_scale:
      weight = self.weight[:self.num_features]
      bias = self.bias[:self.num_features]
    else:
      weight = self.weight[self.model_idx, :self.num_features]
      bias = self.bias[self.model_idx, :self.num_features]

    if self.share_mv:
      running_mean = self.running_mean[:self.num_features]
      running_var = self.running_var[:self.num_features]
      num_batches_tracked = self.num_batches_tracked
    else:
      running_mean = self.running_mean[self.model_idx, :self.num_features]
      running_var = self.running_var[self.model_idx, :self.num_features]
      num_batches_tracked = self.num_batches_tracked[self.model_idx]

    exponential_average_factor = 0.0
    if self.training:
      num_batches_tracked += 1
      if self.momentum is None:
        exponential_average_factor = 1.0 / float(num_batches_tracked)
      else:
        exponential_average_factor = self.momentum 

    return F.batch_norm(x, running_mean, running_var, weight, bias, self.training,
                        exponential_average_factor, self.eps) 

class PrunableLinear(torch.nn.Linear):
  def __init__(self, *args, **kwargs):
    super(PrunableLinear, self).__init__(*args, **kwargs)

  def _set_input_features(self, in_features):
    self.in_features = in_features

  def _set_output_features(self, out_features):
    self.out_features = out_features

  def set_input_output_features(self, in_features, out_features):
    self._set_input_features(in_features)
    self._set_output_features(out_features)

  def forward(self, input):
    b = self.bias[:self.out_features] if self.bias is not None else None
    w = self.weight[:self.out_features, :self.in_features]

    return F.linear(input, w, b)

class PrunableDownsampleA(torch.nn.Module):
  def __init__(self, out_channels):
    super(PrunableDownsampleA, self).__init__()
    self.out_channels = out_channels
    self.avg = torch.nn.AvgPool2d(kernel_size = 1, stride = 2)

  def set_out_channels(self, out_channels):
    self.out_channels = out_channels

  def forward(self, input):
    input = self.avg(input)

    in_channels = input.size(1)

    if in_channels < self.out_channels:
      return torch.cat([input, torch.zeros(input.size(0), self.out_channels - in_channels, input.size(2), input.size(3), device = input.device)], dim = 1)

    return input[:, :self.out_channels, :, :]

class PrunableModel(torch.nn.Module):
  def __init__(self):
    super(PrunableModel, self).__init__()

  def feature_dims(self):
    return self.cns_feature_dims() + self.wp_feature_dims()

  def cns_feature_dims(self):
    raise NotImplementedError

  def wp_feature_dims(self):
    dims = 0
    for m in self.modules():
      if isinstance(m, PrunableConv):
        # skip depthwise convolutions
        if not m.is_dw:
          dims += 1
    return dims

  def enable_conv_wp(self):
    for m in self.modules():
      if isinstance(m, PrunableConv) and (not m.is_dw):
        m.wp_enabled = True

  def disable_conv_wp(self):
    for m in self.modules():
      if isinstance(m, PrunableConv) and (not m.is_dw):
        m.wp_enabled = False

  def update_importance(self):
    for m in self.modules():
      if isinstance(m, PrunableConv) and (not m.is_dw):
        m.update_importance()

  def extract_model(self, feature, model_idx = 0):
    feature_dims = self.feature_dims()
    cns_feature_dims = self.cns_feature_dims()
    wp_feature_dims = self.wp_feature_dims()
    assert(len(feature) == feature_dims or len(feature) == cns_feature_dims or len(feature) == cns_feature_dims + 1)

    if len(feature) == cns_feature_dims:
      feature = np.concatenate([feature, np.ones(wp_feature_dims)])
    elif len(feature) == cns_feature_dims + 1:
      feature = np.concatenate([feature[:cns_feature_dims], feature[-1] * np.ones(wp_feature_dims)])

    if hasattr(self, '_lbound'):
      feature = np.maximum(feature, self._lbound)
    if hasattr(self, '_ubound'):
      feature = np.minimum(feature, self._ubound)

    for m in self.modules():
      if isinstance(m, PrunableBN):
        if not (m.share_scale and m.share_mv):
          m.set_model_index(model_idx)

    self._extract_model(feature[:cns_feature_dims])

    i = 0
    for m in self.modules():
      if isinstance(m, PrunableConv) and not m.is_dw:
        m.wp_density = feature[cns_feature_dims + i]
        i += 1



  def reset_bn_running_states(self):
    for m in self.modules():
      if isinstance(m, PrunableBN):
        m.reset_running_states()

  def save_bn_running_states(self):
    for m in self.modules():
      if isinstance(m, PrunableBN):
        m.save_running_states()

  def load_bn_running_states(self):
    for m in self.modules():
      if isinstance(m, PrunableBN):
        m.load_running_states()

  def _parse_bound(self, bound):
    raise NotImplementedError

  def parse_bound(self, bound):
    if isinstance(bound, (float, int)):
      return bound
    elif isinstance(bound, (list, tuple)) and all([isinstance(d, (float, int)) for d in bound]):
      return bound
    else:
      return self._parse_bound(bound) 

  def conv_info(self, input_shape):
    convs = [m for m in self.modules() if isinstance(m, PrunableConv) and not m.is_dw]

    info = []
    def conv_hook(m, inputs, output):
      _, _, kh, kw = m.weight.shape
      _, _, ih, iw = inputs[0].shape
      ic, oc = m.in_channels, m.out_channels
      stride = m.stride[0]
      info.append([ic, ih, oc, kh, stride, False, m.wp_density])

    handles = []
    for m in convs:
      handle = m.register_forward_hook(conv_hook)
      handles.append(handle)

    device = next(self.parameters()).device
    x = torch.randn(1, *input_shape).to(device)

    out = self(x)

    for handle in handles:
      handle.remove()

    return info

  def init_weights(self):
    for m in self.modules():
      if isinstance(m, (PrunableConv, PrunableLinear)):
        m.reset_parameters()
      if isinstance(m, PrunableBN):
        with torch.no_grad():
          m.weight.fill_(1)
          m.bias.zero_()
          m.reset_running_states()
