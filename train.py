from scipy.interpolate import interpn

import argparse

import torch

import utils
import lib

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type = str)
  args = parser.parse_args()

  config = utils.Config.from_file(args.config)

  trainer = lib.create_trainer(config)

  trainer.run()
