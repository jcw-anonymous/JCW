import torch

def TeacherStudentKLDivLoss(out, teacher_out):
  out_logsoft = out.log_softmax(dim = 1)
  teacher_soft = teacher_out.softmax(dim = 1)
  return - (out_logsoft * teacher_soft).sum() / out.shape[0]

def LabelSmoothKLDivLoss(out, target, label_smooth):
  """
    kl divergence loss with label smooth.
  """
  assert label_smooth < 1
  batch_size, num_classes = out.shape
  smoothed_target = torch.zeros(batch_size, num_classes, dtype = out.dtype, device = out.device)
  smoothed_target.fill_(label_smooth / (num_classes - 1))
  smoothed_target.scatter_(1, target.view(-1, 1), 1 - label_smooth)


  out_logsoft = out.log_softmax(dim = 1)

  return - (out_logsoft * smoothed_target).sum() / out.shape[0] 

def rank_loss(out, target, reduction = 'sum'):
  S = torch.sign(target.view(-1, 1) - target.view(1, -1))
  F = out.view(-1, 1) - out.view(1, -1)
  E = torch.exp(- S * F)
  loss = (1 + E).log().sum()
  scale = 1.
  if reduction == 'mean':
    scale = out.numel()
  return loss / float(scale)

def mse_loss(out, target, reduction = 'mean'):
  err = out - target
  l = err * err

  if reduction == 'mean':
    return l.mean()
  elif reduction == 'sum':
    return l.sum()
  else:
    raise ValueError('Unsupported loss reduction: {}'.format(reduction))
