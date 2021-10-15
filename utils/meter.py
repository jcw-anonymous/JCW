class AvgMeter(object):
  def __init__(self):
    self.n = 0
    self.sum = 0
    self.avg = 0

  def update(self, value):
    self.n += 1
    self.sum += value
    self.avg = self.sum / float(self.n)

  def clear(self):
    self.n = 0
    self.sum = 0
    self.avg = 0

class AccMeter(object):
  def __init__(self):
    self.n = 0
    self.top1 = 0
    self.top5 = 0
    self.acc1 = 0
    self.acc5 = 0

  def update(self, input, target):
    _, topk = input.topk(5, dim = 1)

    topk = topk.long()
    target = target.to(topk.device).long()

    correct = topk.eq(target.view(-1, 1))

    self.n += len(target)
    self.top1 += correct[:, 0].float().sum().item()
    self.top5 += correct.float().sum().item()
   
    self.acc1 = self.top1 / float(self.n)
    self.acc5 = self.top5 / float(self.n)

  def clear(self):
    self.n = 0
    self.top1 = 0
    self.top5 = 0
    self.acc1 = 0
    self.acc5 = 0
