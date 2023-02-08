# LCDNet: Deep Loop Closure Detection and Point Cloud Registration for LiDAR SLAM  (IEEE T-RO 2022)

Official PyTorch implementation of LCDNet.

[![](imgs/video-preview.png)](https://www.youtube.com/watch?v=nAvTdEFRh_s)

## Installation

You can install LCDNet locally on your machine, or use the provided Dockerfile to run it in a container. The `environment_lcdnet.yml` file is meant to be used with docker, as it contains version of packages that are specific to a CUDA version. We don't recommend using it for local installation.

### Local Installation

1. Install [PyTorch](https://pytorch.org/) (make sure to select the correct cuda version)
2. Install the requirements
```pip install -r requirements.txt```
3. Install [spconv](https://github.com/traveller59/spconv) <= 2.1.25 (make sure to select the correct cuda version, for example ```pip install spconv-cu113==2.1.25``` for cuda 11.3)
4. Install [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
5. Install [faiss-cpu](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md) - NOTE: avoid installing faiss via pip, use the conda version, or build it from source alternatively.

Tested in the following environments:
* Ubuntu 18.04/20.04/22.04
* cuda 10.2/11.1/11.3
* pytorch 1.8/1.9/1.10
* Open3D 0.12.0

#### Note
We noticed that the RANSAC implementation in Open3D version >=0.15 achieves bad results. We tested our code with Open3D versions between 0.12.0 and 0.14.2, please use one of these versions, as results might be very different otherwise.

We also noticed that spconv version 2.2 or higher is not compatible with the pretrained weights provided with this repository. Spconv version 2.1.25 or lower is required to properly load the pretrained model.

### Docker

1. Install Docker and NVIDIA-Docker (see [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) for instructions)
2. Download the pretrained model (see [Pretrained model](#pretrained-model) section) in the same folder as the Dockerfile
3. Build the docker image ```docker build --tag lcdnet -f Dockerfile .```
4. Run the docker container ```docker run --gpus all -it --rm -v KITTI_ROOT:/data/KITTI lcdnet```
5. From inside the container, activate the anaconda environment ```conda activate lcdnet``` and change directory to the LCDNet folder ```cd LCDNet```
7. Run the training or evaluation scripts (see [Training](#training) and [Evaluation](#evaluation) sections). The weights of the pretrained model are copied inside the container under ```/pretreined_models/LCDNet-kitti360.tar```.

## Preprocessing

### KITTI

Download the [SemanticKITTI](http://semantic-kitti.org/dataset.html#download) dataset and generate the loop ground truths:

```python -m data_process.generate_loop_GT_KITTI --root_folder KITTI_ROOT```

where KITTI_ROOT is the path where you downloaded and extracted the SemanticKITTI dataset.

NOTE: although the semantic labels are not required to run LCDNet, we use the improved ground truth poses provided with the SemanticKITTI dataset.

### KITTI-360

Download the [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/download.php) dataset (raw velodyne scans, calibrations and vehicle poses) and generate the loop ground truths:

```python -m data_process.generate_loop_GT_KITTI360 --root_folder KITTI360_ROOT```

where KITTI360_ROOT is the path where you downloaded and extracted the KITTI-360 dataset.

### Optional: Ground Plane Removal

To achieve better results, it is suggested to preprocess the datasets by removing the ground plane:

```python -m data_process.remove_ground_plane_kitti --root_folder KITT_ROOT```

```python -m data_process.remove_ground_plane_kitti360 --root_folder KITT_ROOT360```

If you skip this step, please remove the option ```--without_ground``` in all the following steps.

## Training

The training script will use all the available GPUs, if you want to use only a subset of the GPUs, use the environment variable ```CUDA_VISIBLE_DEVICES```. If you don't know how to do that, check [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars).

To train on the KITTI dataset:

```python -m training_KITTI_DDP --root_folder KITTI_ROOT --dataset kitti --batch_size B --without_ground```

To train on the KITTI-360 dataset:

```python -m training_KITTI_DDP --root_folder KITTI360_ROOT --dataset kitti360 --batch_size B --without_ground```

To track the training progress using [Weights & Biases](https://wandb.ai/), add the argument ```--wandb```.
The per-GPU batch size B must be at least 2, and a GPU with at least 8GB of RAM is required (12GB or more is preferred). In our experiments we used a batch size of 6 on 4 x 24GB GPUs, for a total batch size of 24.

The network's weights will be saved in the folder ```./checkpoints``` (you can change this folder with the argument ```--checkpoints_dest```), inside a subfolder named with the starting date and time of the training (format ```%d-%m-%Y_%H-%M-%S```), for example: ```20-02-2022_16-38-24```

## Evaluation

### Loop Closure

To evaluate the loop closure performance of the trained model on the KITTI dataset:

```python -m evaluation.inference_loop_closure --root_folder KITTI_ROOT --dataset kitti --validation_sequence 08 --weights_path WEIGHTS --without_ground```

where WEIGHTS is the path of the pretrained model, for example ```./checkpoints/20-02-2022_16-38-24/checkpoint_last_iter.tar```

Similarly, on the KITTI-360 dataset:

```python -m evaluation.inference_loop_closure --root_folder KITTI360_ROOT --dataset kitti360 --validation_sequence 2013_05_28_drive_0002_sync --weights_path WEIGHTS --without_ground```

### Point Cloud Registration

To evaluate the loop closure performance of the trained model on the KITTI and KITTI-360 dataset:

```python -m evaluation.inference_yaw_general --root_folder KITTI_ROOT --dataset kitti --validation_sequence 08 --weights_path WEIGHTS --ransac --without_ground```

```python -m evaluation.inference_yaw_general --root_folder KITTI360_ROOT --dataset kitti360 --validation_sequence 2013_05_28_drive_0002_sync --weights_path WEIGHTS --ransac --without_ground```

To evaluate LCDNet (fast), remove the ```--ransac``` argument.

## Pretrained Model

A model pretrained on the KITTI-360 dataset can be found [here](https://drive.google.com/file/d/176dQn6QhFoolu3bcGvyGuBxaCQY42kNn/view?usp=sharing)

## Paper

"LCDNet: Deep Loop Closure Detection and Point Cloud Registration for LiDAR SLAM"
* [IEEEXplore](https://ieeexplore.ieee.org/document/9723505)
* [Arxiv](https://arxiv.org/abs/2103.05056)
* [Video](https://www.youtube.com/watch?v=nAvTdEFRh_s)

If you use LCDnet, please cite:
```
@ARTICLE{cattaneo2022tro,
  author={Cattaneo, Daniele and Vaghi, Matteo and Valada, Abhinav},
  journal={IEEE Transactions on Robotics}, 
  title={LCDNet: Deep Loop Closure Detection and Point Cloud Registration for LiDAR SLAM}, 
  year={2022},
  volume={},
  number={},
  pages={1-20},
  doi={10.1109/TRO.2022.3150683}
 }
```

## Contacts
* [Daniele Cattaneo](https://rl.uni-freiburg.de/people/cattaneo)
* [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)

## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.
