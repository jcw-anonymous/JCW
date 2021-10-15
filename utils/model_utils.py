import torch
import models

def load_state_dict(model, state_dict):
  state_dict = state_dict['model'] if 'model' in state_dict else state_dict
  state_dict = {k[7:] if k.startswith('module.') else k : v for k, v in state_dict.items()}
  model.load_state_dict(state_dict)
  return model

def get_model(config, **kwargs):
  if hasattr(config, 'kwargs'):
    kwargs.update(**config.kwargs)
  model = models.__dict__[config.model](**kwargs)
  if hasattr(config, 'load_model_from') and config.load_model_from is not None:
    print('=> Load model from: {}'.format(config.load_model_from))
    ck = torch.load(config.load_model_from)
    ck = ck['model'] if 'model' in ck else ck
    ck = {k[7:] if k.startswith('module.') else k : v for k, v in ck.items()}
    # delete importance in ck
    ck = {k : v for k, v in ck.items() if not k.endswith('importance')}
    model.load_state_dict(ck)

  return model

def get_model_params(model, config):
  if not config.no_bias_decay:
    return model.parameters()

  conv_weights = []
  other_params = []

  param_names = []

  for name, m in model.named_modules():
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
      conv_weights.append(m.weight)
      param_names.append('{}.weight'.format(name))
      if m.bias is not None:
        other_params.append(m.bias)
        param_names.append('{}.bias'.format(name))

  for name, param in model.named_parameters():
    if name not in param_names:
      other_params.append(param)

  assert len(conv_weights) + len(other_params) == len(list(model.parameters()))

  return [{
    'params': conv_weights
  }, {
    'params': other_params,
    'weight_decay': 0
  }]
