import numpy as np


def isdominate(p, q):
  return all(p <= q) and any(p != q)
  
  
def fast_non_dominated_sort(P):
  # each row of P denotes the benifits of an individual
  nindividuals, _ = P.shape
  
  # S[p] are the individuals that are dominated by p
  # n[p] is the number of individuals that dominate p
  # F: the sorted individuals
  S = [[] for i in range(nindividuals)]
  n = [0 for i in range(nindividuals)]
  F = [[]]
  
  for p in range(nindividuals):
    for q in range(p + 1, nindividuals):
      if isdominate(P[p], P[q]):
        S[p].append(q)
        n[q] += 1
      elif isdominate(P[q], P[p]):
        S[q].append(p)
        n[p] += 1
        
    if n[p] == 0:
      # there is no individual that dominates p
      F[-1].append(p)
      
  while True:
    Q = []
    for p in F[-1]:
      for q in S[p]:
        n[q] -= 1
        if n[q] == 0:
          Q.append(q)
          
    if len(Q) == 0:
      break
    else:
      F.append(Q)
      
  return F
  
def crowding_distance_assignment(P):
  # each row of P denotes the benifits of an individual
  nindividuals, nvalues = P.shape
  sorted_idx = P.argsort(axis = 0)
  
  I = [0. for i in range(nindividuals)]
  for v in range(nvalues):
    I[sorted_idx[0, v]] = np.inf
    I[sorted_idx[-1, v]] = np.inf
  
  for n in range(2, nindividuals - 1):
    for v in range(nvalues):
      prev = sorted_idx[n - 1, v]
      cur = sorted_idx[n, v]
      next = sorted_idx[n + 1, v]
      I[cur] += float(P[next, v] - P[prev, v]) / float(P[sorted_idx[-1, v], v] - P[sorted_idx[0, v], v])
      
  return I
  
def SBX(P, N, mu = 2., prob = 0.5):
  # Simulated binary crossover
  # Each row of P denotes an individual.
  indices = np.random.choice(P.shape[0], size = (N // 2, 2))
  for i in range(N // 2):
    while indices[i, 0] == indices[i, 1]:
      indices[i, 1] = np.random.choice(P.shape[0])
      
  indices1 = indices[:, 0]
  indices2 = indices[:, 1]
  
  p1 = P[indices1, :]
  p2 = P[indices2, :]
  
  _, dims = p1.shape
  
  u = np.random.uniform(0, 1, size = (N // 2, dims))
  
  idx1 = (u <= 0.5)
  idx2 = (u > 0.5)
  
  u[idx1] = np.power(2 * u[idx1], 1. / (mu + 1.))
  u[idx2] = np.power(1. / (2. * (1 - u[idx2])), 1. / (mu + 1.))
  
  q1 = 0.5 * ((1. + u) * p1 + (1. - u) * p2)
  q2 = 0.5 * ((1. - u) * p1 + (1. + u) * p2)
  
  # random swap
  u = np.random.uniform(0, 1, size = (N // 2, dims))
  swap = (u < 0.5)
  c = np.copy(q1[swap])
  q1[swap] = q2[swap]
  q2[swap] = c
  
  # random crossover
  u = np.random.uniform(0, 1, size = (N // 2, dims))
  no_crossover = (u > prob)
  q1[no_crossover] = p1[no_crossover]
  q2[no_crossover] = p2[no_crossover]
  
  
  return np.concatenate([q1, q2], axis = 0)
  
def PLM(P, prob, mu, lbound, ubound):
  # Polynomial mutation operator
  nindividuals, dims = P.shape
  lbound = np.repeat(lbound[None, :], nindividuals, axis = 0)
  ubound = np.repeat(ubound[None, :], nindividuals, axis = 0)
  
  u = np.random.uniform(0, 1, size = (nindividuals, dims))
  do_mutation = (u <= prob)
  do_mutation[lbound == ubound] = False
  
  Q = np.copy(P)
  
  mutatedQ = Q[do_mutation]
  lbound = lbound[do_mutation]
  ubound = ubound[do_mutation]
  
  
  u = np.random.uniform(0, 1, size = mutatedQ.size)
  left = (u < 0.5)
  right = (u >= 0.5)
  
  delta1 = (mutatedQ[left] - lbound[left]) / (ubound[left] - lbound[left])
  delta2 = (ubound[right] - mutatedQ[right]) / (ubound[right] - lbound[right])
  
  deltaq = np.zeros((mutatedQ.size,), dtype = np.float32)
  
  #print(1. - delta1)
  #exit()
  
  deltaq[left] = np.power(2 * u[left] + (1. - 2 * u[left]) * np.power(1. - delta1, 1. + mu), 1. / (1. + mu)) - 1.
  deltaq[right] = 1. - np.power(2 * (1. - u[right]) + 2 * (u[right] - 0.5) * np.power(1 - delta2, 1 + mu), 1. / (1. + mu))
  
  mutatedQ += deltaq * (ubound - lbound)
  mutatedQ.clip(lbound, ubound)
  
  
  Q[do_mutation] = mutatedQ
  
  return Q
