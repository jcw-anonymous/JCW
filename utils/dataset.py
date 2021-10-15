import os

import torch
from torchvision import datasets, transforms
import numpy as np

def get_dataloader(config):
  if config.data == 'ImageNet':
    normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                     std = [0.229, 0.224, 0.225])
    train_transform = []
    augmentation = config.augmentation if hasattr(config, 'augmentation') else None
    scale = [0.2, 1.0]
    train_transform.extend([
      transforms.RandomResizedCrop(224, scale = scale),
      transforms.RandomHorizontalFlip(),
    ]) 
    if augmentation is not None and hasattr(augmentation, 'color_jitter'):
      print('Using color jittor.')
      train_transform.append(transforms.ColorJitter(*augmentation.color_jitter))
    train_transform.extend([
      transforms.ToTensor(),
      normalize
    ])
    train_transform = transforms.Compose(train_transform)

    val_transform = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize
    ])

    train_dir = os.path.join(config.root, 'train')
    val_dir = os.path.join(config.root, 'val')
    train_set = datasets.ImageFolder(root = train_dir, transform = train_transform)
    if not hasattr(config, 'SPLIT'):
      val_set = datasets.ImageFolder(root = val_dir, transform = val_transform)
    else:
      split_config = config.SPLIT 
      val_set = datasets.ImageFolder(root = train_dir, transform = val_transform)
      
      split_config.setdefault('num_classes', 1000)
      num_classes = split_config.num_classes
      
      # randomly choice `num_classes` classes 
      classes = np.random.choice(train_set.classes, size = num_classes, replace = False)
      classes.sort()
      class_to_path = {target : [] for target in classes}
      for path, class_id in train_set.samples:
        target = train_set.classes[class_id]
        if target in classes:
          class_to_path[target].append(path)
      class_to_idx = {classes[i] : i for i in range(len(classes))}
      train_samples = []
      val_samples = []
      for target, pathes in class_to_path.items():
        train_samples_per_class = split_config.train_samples_per_class
        val_samples_per_class = split_config.val_samples_per_class
        selected_pathes = np.random.choice(pathes, train_samples_per_class + val_samples_per_class, replace = False)
        for path in selected_pathes[:train_samples_per_class]:
          train_samples.append((path, class_to_idx[target]))  
        for path in selected_pathes[train_samples_per_class:]:
          val_samples.append((path, class_to_idx[target]))
      train_targets = [s[1] for s in train_samples]
      val_targets = [s[1] for s in val_samples]
      
      # reset train/val dataset 
      train_set.classes = classes
      train_set.class_to_idx = class_to_idx
      train_set.samples = train_samples
      train_set.targets = train_targets
      
      val_set.classes = classes
      val_set.class_to_idx = class_to_idx
      val_set.samples = val_samples
      val_set.targets = val_targets 
  elif config.data == 'CIFAR-10':
    normalize = transforms.Normalize(mean = [m / 255. for m in [125.3, 123.0, 113.9]],
                                     std = [s / 255. for s in [63.0, 62.1, 66.7]])
    train_transform = []
    train_transform.extend([
      transforms.RandomCrop(32, padding = 4),
      transforms.RandomHorizontalFlip()
    ])
    augmentation = config.augmentation if hasattr(config, 'augmentation') else None
    if augmentation is not None and hasattr(augmentation, 'color_jittor'):
      train_transform.append(transforms.ColorJittor(augmentation.color_jittor))

    train_transform.extend([
      transforms.ToTensor(),
      normalize
    ])
    train_transform = transforms.Compose(train_transform)

    val_transform = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])

    train_set = datasets.CIFAR10(root = config.root, train = True, transform = train_transform)
    val_set = datasets.CIFAR10(root = config.root, train = False, transform = val_transform)
  else:
    raise NotImplementedError

  train_loader = torch.utils.data.DataLoader(train_set, batch_size = config.train_batch,
                                             num_workers = config.num_workers,
                                             shuffle = True,
                                             pin_memory = True)
  val_loader = torch.utils.data.DataLoader(val_set, batch_size = config.val_batch,
                                           num_workers = config.num_workers,
                                           shuffle = False,
                                           pin_memory = True)
  return train_loader, val_loader
