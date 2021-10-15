import numpy as np

from .emo_utils import SBX, fast_non_dominated_sort, crowding_distance_assignment, PLM

class NSGA_II(object):
  points = None
  text = None
  def __init__(self, config):
    self._init_config(config)
    self._init_info()
    
  def _init_config(self, config):
    self._ngenerations = config.ngenerations
    self._nindividuals = config.nindividuals
    self._individual_dims = config.individual_dims
    self._mutation_prob = config.mutation_prob
    self._plm_mu = config.plm_mu
    self._sbx_mu = config.sbx_mu
    self._sbx_prob = config.sbx_prob
    self._extension_factor = config.extension_factor
    self._nsolutions = config.nsolutions
    self._init_bounds(config)
    self._selection_config = None
    if hasattr(config, 'SELECTION'):
      self._selection_config = config.SELECTION

    self._candidate_list = None
    if hasattr(config, 'candidate_list'):
      self._candidate_list = config.candidate_list

    # configuration for elimination
    self._eliminate_repeat_eps = 0.
    if hasattr(config, 'eliminate_repeat_eps'):
      self._eliminate_repeat_eps = config.eliminate_repeat_eps
    
  def _init_bounds(self, config):
    
    def _make_array(d, size):
      if isinstance(d, (float, int)):
        d = [d]
      assert(isinstance(d, (list, tuple)))
      if len(d) == 1:
        d *= size
      assert(len(d) == size)
      
      return np.array(d)
      
    self._lbound = _make_array(-np.inf, self._individual_dims)
    self._ubound = _make_array(np.inf, self._individual_dims)
    
    if hasattr(config, 'lbound'):
      self._lbound = _make_array(config.lbound, self._individual_dims)
    if hasattr(config, 'ubound'):
      self._ubound = _make_array(config.ubound, self._individual_dims)
      
  def _init_info(self):
    if self._candidate_list is None:
      self.P = np.random.uniform(self._lbound, self._ubound, size = (self._nindividuals, self._individual_dims))
    else:
      cand = np.array(self._candidate_list)
      Ps = []
      for i in range(self._individual_dims):
        mask = ((cand >= self._lbound[i]) & (cand <= self._ubound[i]))
        masked_cand = cand[mask]
        Ps.append(np.random.choice(masked_cand, size = (self._nindividuals, 1)))
      self.P = np.concatenate(Ps, axis = 1)

  def _evaluate(self, population):
    # This function should be implemented by customers
    raise NotImplementedError

  def find_nearest(self, P):
    if self._candidate_list is None:
      return P

    N, dims = P.shape

    P = P.reshape(-1, 1)
    cand = np.array(self._candidate_list)

    err = np.abs(P - cand)
    indices = err.argmin(axis = 1)

    return cand[indices].reshape(N, dims)

  def _eliminate_repeat(self, P, Q):
    # eliminate repeated individuals
    Mp, Np = P.shape
    Mq, Nq = Q.shape

    P = P.reshape(Mp, 1, Np)
    dist = np.abs(P - Q).sum(axis = 2).min(axis = 0) / float(Np)

    return Q[dist > self._eliminate_repeat_eps, :]
 
  def _make_new_pop(self):
    # Make new population through crossover and mutation
    N = self._nindividuals * self._extension_factor
    n = 0
    i =0
    P = self.P
    Q = None
    while n < N and i < 16:
      _Q = SBX(self.P, N, self._sbx_mu, self._sbx_prob)
      _Q = self.find_nearest(_Q)
      _Q = _Q.clip(self._lbound[None, :], self._ubound[None, :])
      _Q = PLM(_Q, self._mutation_prob, self._plm_mu, self._lbound, self._ubound)
      _Q = self.find_nearest(_Q)
      _Q = _Q.clip(self._lbound[None, :], self._ubound[None, :])
      i += 1
      if i < 16:
        _Q = self._eliminate_repeat(P, _Q)
        P = np.concatenate([P, _Q], axis = 0)
      if _Q is None:
        continue
 
      if Q is None:
        Q = _Q
      else:
        if Q.shape[0] + _Q.shape[0] > N:
          _Q = _Q[:N - n, :]
        Q = np.concatenate([Q, _Q], axis = 0)

      n = Q.shape[0]

    self.Q = Q
        
    
  def _get_next_P(self):
    R = np.concatenate([self.P, self.Q], axis = 0)
    RScore = self._evaluate(R)
    #RScore = np.concatenate([self.PScore, self.QScore], axis = 0)
    self.P, self.PScore = self._sorted(R, RScore)

  def _psorted(self, R, RScore, nindividuals = None):
    if nindividuals == None:
      nindividuals = self._nindividuals

    if len(R) <= nindividuals:
      return R, RScore

    if self._selection_config.sample_mode == 'uniform':
      sampled = np.linspace(self._selection_config.min_val,
                            self._selection_config.max_val,
                            self._selection_config.nsamples)
    elif self._selection_config.sample_mode == 'random':
      sampled = self._selection_config.min_val \
              + np.random.uniform(0, 1, size = self._selection_config.nsamples) \
              * (self._selection_config.max_val \
              - self._selection_config.min_val)
      if self._selection_config.sandwich:
        sampled[0] = self._selection_config.min_val
        sampled[-1] = self._selection_config.max_val
    else:
      raise ValueError('Unrecognized sample mode: {}'.format(self._selection_config.sample_mode))    

    RScores = np.abs(RScore[:, [1]] - sampled)


    by_frontier = True
    if hasattr(self._selection_config, 'by_frontier'):
      by_frontier = self._selection_config.by_frontier

    if by_frontier:
      selected_indices = self._select_by_frontier(RScores, RScore, nindividuals)
    else:
      selected_indices = self._select_by_individual(RScores, RScore, nindividuals)
  
    return R[selected_indices, :], RScore[selected_indices, :]

  def _select_by_frontier(self, RScores, RScore, nindividuals):
    selected_indices = set()
    Fs = []
    for i in range(self._selection_config.nsamples):
      rscore = RScore.copy()
      rscore[:, 1] = RScores[:, i]
      F = fast_non_dominated_sort(rscore)
      Fs.append(F)      
   
    stage = 0 
    while len(selected_indices) < nindividuals:
      for i in range(len(Fs)):
        if stage < len(Fs[i]):
          new_indices = set(Fs[i][stage])
          next_indices = selected_indices | new_indices
          if len(next_indices) <= nindividuals:
            selected_indices = next_indices
          elif len(selected_indices) < nindividuals:
            rscore = RScore.copy()
            I = crowding_distance_assignment(rscore[Fs[i][stage], 1].reshape(-1, 1))
            idx = np.argsort(I)
            for j in idx[::-1]:
              selected_indices.add(Fs[i][stage][j])
              if len(selected_indices) >= nindividuals:
                break
        if len(selected_indices) >= nindividuals:
          break
      stage += 1
    
    selected_indices = list(selected_indices)

    return selected_indices

  def _select_by_individual(self, RScores, RScore, nindividuals):
    selected_indices = set()
    Is = []
    for i in range(self._selection_config.nsamples):
      rscore = RScore.copy()
      rscore[:, 1] = RScores[:, i]
      F = fast_non_dominated_sort(rscore)
      sorted_indices = []
      for f in F:
        d = crowding_distance_assignment(RScore[f, 1].reshape(-1, 1))
        idx = np.argsort(d)
        sorted_indices.extend([f[i] for i in idx[::-1]])
      Is.append(sorted_indices)
    
    for i in range(len(RScore)):
      if len(selected_indices) >= nindividuals:
        break
      for j in range(self._selection_config.nsamples):
        selected_indices.add(Is[j][-1 - i])
        if len(selected_indices) >= nindividuals:
          break

    return list(selected_indices)

 
  def _sorted(self, R, RScore, nindividuals = None):
    if self._selection_config is not None:
      return self._psorted(R, RScore, nindividuals)
      
    if nindividuals is None:
      nindividuals = self._nindividuals
    if len(R) <= nindividuals:
      return R, RScore
    F = fast_non_dominated_sort(RScore)
    sorted_idx = []
    i = 0
    while len(sorted_idx) + len(F[i]) <= nindividuals:
      sorted_idx.extend(F[i])
      i += 1
      
    if len(sorted_idx) < nindividuals:
      n = nindividuals - len(sorted_idx)
      I = crowding_distance_assignment(RScore[F[i], :])
      idx = np.argsort(I)
      sorted_idx.extend([F[i][j] for j in idx[-n:]])
      
    return R[sorted_idx, :], RScore[sorted_idx, :]
    
  def _step(self, gen):
    #if gen == 0:
    #  self.PScore = self._evaluate(self.P)
    self._make_new_pop()
    #self.QScore = self._evaluate(self.Q)
    
    self._get_next_P()
    
    if NSGA_II.points is not None:
      NSGA_II.points.set_data(self.PScore[:, 0], self.PScore[:, 1])
    if NSGA_II.text is not None:
      NSGA_II.text.set_text('Generation: {}'.format(gen + 1))
      
  
  def run(self):
    for g in range(self._ngenerations):
      self._step(g)  

