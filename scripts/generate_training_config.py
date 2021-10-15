import os
import argparse
import yaml
from copy import deepcopy

import torch

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--base-config', type = str, default = None,
    help = 'The base configuration file for training. This file defines '
           'the important hyper parameters for training, such as training epochs, '
           'learning rate and so on. All the generated configuration files keep '
           'the same settings for these hyper parameters as the base configuration '
           'file, except for the save directory and the architecture of trained models, i.e. the number '
           'of channels and weight sparsity for each layer.'
  )
  parser.add_argument('--search-result', type = str, default = None,
    help = 'The file that saves the search result, i.e., the architectures of searched pareto optimal models.'
  )
  parser.add_argument('--output', type = str, default = None,
    help = 'The output directory to save the generated configuration files.'
  )
  args = parser.parse_args()

  with open(args.base_config) as f:
    base_config = yaml.load(f.read())

  search_result = torch.load(args.search_result)
  searched_models = search_result['models']

  for idx, model in enumerate(searched_models):
    outdir = os.path.join(args.output, 'model-{}'.format(idx))
    config = deepcopy(base_config)
    config['save'] = outdir
    config['MODEL']['EXTRACT']['feature'] = [float(m) for m in model]
    if not os.path.exists(outdir):
      os.makedirs(outdir)

    outf = os.path.join(outdir, 'train.yaml')
    with open(outf, 'w') as f:
      yaml.dump(config, f)
