from collections import OrderedDict, Iterable
import os
import sys
from importlib import import_module

def cast(value):
  if isinstance(value, (dict, OrderedDict)):
    if '__isdict__' in value and value['__isdict__'] == True:
      del value['__isdict__']
      return {k : cast(v) for k, v in value.items()}
    else:
      return Config(value)

  elif value is None:
    return None

  elif isinstance(value, Config):
    return value

  elif isinstance(value, str):
    return value

  elif isinstance(value, Iterable):
    return [cast(v) for v in value]

  else:
    return value

class Config(object):
  def __init__(self, cfg = None):
    if cfg is not None:
      assert isinstance(cfg, (dict, OrderedDict))
      for n, v in cfg.items():
        assert isinstance(n, str)
        setattr(self, n, v)

  def __setattr__(self, name, value):
    name = name.replace('-', '_')
    super(Config, self).__setattr__(name, cast(value))

  def __getattr__(self, name):
    try:
      name = name.replace('-', '_')
      return super(Config, self).__getattr__(name)
    except:
      raise AttributeError('The Config object has no attribute: {}'.format(name))

  def _setdefault(self, name, value):
    name = name.replace('-', '_')
    if not hasattr(self, name):
      setattr(self, name, value)

  def setdefault(self, *args, **kwargs):
    assert(len(args) % 2 == 0)

    argc = len(args)
    for i in range(0, argc, 2):
      self._setdefault(args[i], args[i + 1])

    for k, v in kwargs.items():
      self._setdefault(k, v)

  @staticmethod
  def from_file(fname):
    use_json = False
    use_yaml = False
    use_py = False

    if fname.endswith('.yaml') or fname.endswith('.yml'):
      use_yaml = True

    if fname.endswith('json'):
      use_json = True

    if fname.endswith('.py'):
      use_py = True

    if use_py:
      module_path = os.path.dirname(fname)
      module_name = os.path.basename(fname)[:-3]
      sys.path.insert(0, module_path)
      m = import_module(module_name)
      return Config({k : v for k, v in m.__dict__.items() if not k.startswith('__')})

    if use_json:
      import json
      with open(fname) as f:
        return Config(json.loads(f.read()))

    if use_yaml:
      import yaml
      with open(fname) as f:
        return Config(yaml.load(f.read()))

    raise ValueError('Cannot guess the file format for the input file: {}'.format(fname))
