# This repository contains the codes for our paper 889: Joint Channel and Weight Pruning for Model Acceleration on Mobile Devices

## Requirements

  + torch
  + torchvision
  + numpy
  + scipy

## Usage
  
  The JCW works in a two-step fashion. i.e. the **search step** and the **training step**. The search step seaches for the layer-wise channel numbers and weight sparsity for Pareto-optimal models. The training steps trains the searched models with ADMM. We give a simple example for resnet18.

## The search step 

  1. **Modify the configuration file**

      First, open the file `experiments/res18-search.yaml`:
      ```shell
      vim experiments/res18-search.yaml
      ```
      Go to the 44th line and find the following codes:
      ```
      DATASET:
        data: ImageNet
        root: /path/to/imagenet
        ...
      ```
      and modify the `root` property of `DATASET` to the path of ImageNet dataset on your machine.

  2. **Apply the search**
      
      After modifying the configuration file, you can simply start the search by:
      ```shell
      python emo_search.py --config experiments/res18-search.yaml | tee experiments/res18-search.log
      ```
      After searching, the search results will be saved in `experiments/search.pth`

## The training step

After searching, we can train the searched models by:

  1. **Modify the base configuration file**

      Open the file `experiments/res18-train.yaml`:
      ```shell
      vim experiments/res18-train.yaml
      ```
      Go to the 5th line, find the following codes:
      ```
      root: &root /path/to/imagenet
      ```
      and modify the root property to the path of ImageNet dataset on your machine.

  2. **Generate configuration files for training**

      After modifying the base configuration file, we are ready to generate the configuration files for training. To do that, simply run the following command:
      ```shell
      python scripts/generate_training_configs.py --base-config experiments/res18-train.yaml --search-result experiments/search.pth --output ./train-configs 
      ``` 
      After running the above command, the training configuration files will be written into ./train-configs/model-{id}/train.yaml.

  3. **Apply the training**
      
      After generating the configuration files, simply run the following command to train one certain model:
      ```shell
      python train.py --config xxxx/xxx/train.yaml | tee xxx/xxx/train.log
      ```
