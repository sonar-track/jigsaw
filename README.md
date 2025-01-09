Unofficial implementation of self-supervised learning methods addressed in the paper 

**Self-Supervised Learning for Sonar Image Classification**

*Alan Preciado-Grijalva, Bilal Wehbe, Miguel Bande Firvida, Matias Valdenegro-Toro; Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2022, pp. 1499-1508*

# Installation

Assuming that Docker is available on host system, first clone the repository

```
git clone git@github.com:sonar-track/jigsaw.git
```

Then open `docker` folder 

```
cd jigsaw/docker
```

and build the image

```
./build.sh
```

Open and edit the `run.sh` script. In this example script, I map the development folder, to where the `jigsaw` repo is also cloned, into the docker container's `/home/trainer/dev`. You are free to change to suit your need.

From the command line, 

```
./run.sh jigsaw
```

Inside the container, navigate to the repository home folder and install the package

```
pip install -e .
```

# Training

After installation, the entry point `jigsaw.py` is accessible. At this moment the script provides a single `train` option. To initialize the training process, first create a default configuration file `.YAML` by calling `jigsaw.py` without arguments:

```
mkdir jigsaw_training
cd jigsaw_training
jigsaw.py
```

This is my example configuration to train a Jigsaw model from UATD dataset. To learn more about the dataset, checkout their paper https://arxiv.org/abs/2212.00352. 

```
experiment_unique_id: jigsaw-mobinetv3s-192x192
train:
  batch_size: 64
  brightness_delta: 0.3
  checkpoint_dirpath: auto
  contrast_range:
  - 0.8
  - 1.2
  crop_range:
  - 0.8
  - 3.0
  dropout: 0.4
  epochs: 200
  image_height: 1024
  image_width: 1024
  initial_epoch: 0
  log_dirpath: auto
  num_cells_x: 3
  num_cells_y: 3
  num_classes: 10
  optimizer: adam
  patch_height: 192
  patch_width: 192
  patience: 10
  preset_model: mobilenetv3small
  seed: 9999
  shuffle: true
  train_image_dirpath: /home/trainer/dev/sonar/datasets/uatd/train
  val_image_dirpath: /home/trainer/dev/sonar/datasets/uatd/test1
  trained_backbone_filepath: auto
  trained_classifier_filepath: auto
```

Given that you have setup paths for the datasets, to train the jigsaw model,

```
jigsaw.py -c default-config.yml train
```

After training finishes, the resulting trained model is saved at `./jigsaw_training/generated/jigsaw-mobilenetv3s-192x192/train/backbone`.