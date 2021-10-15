'''
  Baseclass for all model evaluators
'''
class Evaluator(object):
  registry = {}
  def __init__(self, config):
    self.config = config
  def evaluate(self, models):
    raise NotImplementedError
