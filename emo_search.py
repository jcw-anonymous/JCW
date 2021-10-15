from scipy.interpolate import interpn

import os
import time
import argparse

import torch
import numpy as np

import utils
import lib

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type = str, default = 'search.yaml')
  args = parser.parse_args()

  config = utils.Config.from_file(args.config)

  model_evaluation = lib.create_model_evaluation(config.MODELEVALUATION)
  model_evaluation._model._lbound = model_evaluation._model.parse_bound(config.NSGAII.lbound)
  model_evaluation._model._ubound = model_evaluation._model.parse_bound(config.NSGAII.ubound)

  # EMO re-configuration
  feature_dims = model_evaluation._model.feature_dims()
  config.SEARCH.setdefault('wp_uniform', False)
  if config.SEARCH.wp_uniform:
    cns_feature_dims = model_evaluation._model.cns_feature_dims()
    config.NSGAII.individual_dims = cns_feature_dims + 1
    if isinstance(config.NSGAII.lbound, utils.Config):
      config.NSGAII.lbound = list(model_evaluation._model._lbound[:cns_feature_dims + 1])
      config.NSGAII.lbound[-1] = min(model_evaluation._model._lbound[cns_feature_dims:])
    if isinstance(config.NSGAII.ubound, utils.Config):
      config.NSGAII.ubound = list(model_evaluation._model._ubound[:cns_feature_dims + 1])
      config.NSGAII.ubound[-1] = max(model_evaluation._model._ubound[cns_feature_dims:])
  else:
    config.NSGAII.individual_dims = feature_dims
    if isinstance(config.NSGAII.lbound, utils.Config):
      config.NSGAII.lbound = list(model_evaluation._model._lbound)
    if isinstance(config.NSGAII.ubound, utils.Config):
      config.NSGAII.ubound = list(model_evaluation._model._ubound)
 
  if config.NSGAII.mutation_prob < 0 or config.NSGAII.mutation_prob > 1:
    config.NSGAII.mutation_prob = 1. / config.NSGAII.individual_dims

  emo = lib.NSGA_II(config.NSGAII)

  model_intensity = lib.create_model_intensity(config.MODELINTENSITY)
  if hasattr(config.NSGAII, 'SELECTION'):
    model_evaluation._model.extract_model(emo._lbound)
    min_val = model_intensity.evaluate(model_evaluation._model)
    model_evaluation._model.extract_model(emo._ubound)
    max_val = model_intensity.evaluate(model_evaluation._model)
    config.NSGAII.SELECTION.min_val = min_val
    config.NSGAII.SELECTION.max_val = max_val
    emo._selection_config = config.NSGAII.SELECTION


  def emo_eval(P):
    loss = model_evaluation.evaluate(P)
    # loss = np.random.uniform(0, 1, size = (P.shape[0], 2))
    ret = []
    for i, p in enumerate(P):
      model_evaluation._model.extract_model(p)
      intensity = model_intensity.evaluate(model_evaluation._model)
      ret.append([loss[i][0], intensity])
    return np.array(ret)
  emo._evaluate = emo_eval

  for g in range(emo._ngenerations):
    print('Search step: {} | {}...'.format(g + 1, emo._ngenerations))
    start = time.time()
    emo._step(g)
    
    # print search result
    print('Current result:')
    for p, pscore in zip(emo.P, emo.PScore):
      log = []
      log.append('model:')
      log.extend(['{:.2f}'.format(d) for d in p])
      log.append('score:')
      log.append('({}, {:.2f})'.format(pscore[0], pscore[1]))
      print(' '.join(log))

    # save current result
    state = {
      'models': emo.P,
      'score': emo.PScore
    }
    torch.save(state, os.path.join(config.SEARCH.save, 'search.pth'))

    end = time.time()
    print('Runtime: {:.2f} min'.format((end - start) / 60.))
