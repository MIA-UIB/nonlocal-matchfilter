# Learned Nonlocal Feature Matching and Filtering for RAW Image Denoising                                  

[Marco Sánchez Beeckman](https://orcid.org/0000-0002-5949-0775) and [Antoni Buades](https://orcid.org/0000-0001-9832-3358)

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2604.17453)

## Abstract

> Being one of the oldest and most basic problems in image processing, image denoising has seen a resurgence spurred by rapid advances in deep learning.
Yet, most modern denoising architectures make limited use of the technical knowledge acquired researching the classical denoisers that came before the mainstream use of neural networks, instead relying on depth and large parameter counts.
This poses a challenge not only for understanding the properties of such networks, but also for deploying them on real devices which may present resource constraints and diverse noise profiles.
Tackling both issues, we propose an architecture dedicated to RAW-to-RAW denoising that incorporates the interpretable structure of classical self-similarity-based denoisers into a fully learnable neural network.
Our design centers on a novel nonlocal block that parallels the established pipeline of neighbor matching, collaborative filtering and aggregation popularized by nonlocal patch-based methods, operating on learned multiscale feature representations.
This built-in nonlocality efficiently expands the receptive field, sufficing a single block per scale with a moderate number of neighbors to obtain high-quality results.
Training the network on a curated dataset with clean real RAW data and modeled synthetic noise while conditioning it on a noise level map yields a sensor-agnostic denoiser that generalizes effectively to unseen devices.
Both quantitative and visual results on benchmarks and in-the-wild photographs position our method as a practical and interpretable solution for real-world RAW denoising, achieving results competitive with state-of-the-art convolutional and transformer-based denoisers while using significantly fewer parameters.

## Network architecture

![Network architecture](https://arxiv.org/html/2604.17453v1/x1.png)

## Environment and dependencies

The project has been developed using the following environment:

- Python 3.12
- PyTorch 2.6 w/ CUDA 12.4
- Dependencies stated in `pyproject.toml`, which include two custom PyTorch CUDA extensions that need to be compiled with the same CUDA version as PyTorch.

This repository includes a `Containerfile` that takes care of the environment setup.
We recommend using `podman` or `docker` to run the code in order to avoid issues that may arise due to incompatible dependency versions.

## Quick start (container)

To run the code within a container with a NVIDIA GPU, you will need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The following examples use Podman as a container manager, for which NVIDIA recommends using the [Container Device Interface](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html).
Please, follow the provided links for a simple guide on how to use them.

**1) Build image**

```bash
podman build --tag nonlocalmf .
```

The `Containerfile` sets some environment variables inside the container that specify where the application expects to find the datasets and log the execution details (we have used [Aim](https://github.com/aimhubio/aim) as experiment tracker).
You will need to bind mount the corresponding directories in your host machine to these directories in the container.
In both Docker and Podman, this is done with the `-v` flag.

**2) Training example**

```bash
podman run --rm --device nvidia.com/gpu=all \
  -v /host/path/to/data/png:/data/png \
  -v /host/path/to/data/raw:/data/raw \
  -v $PWD/logs:/nonlocal-matchfilter/logs \
  -v $PWD/results:/aim \
  nonlocalmf \
  train +experiment=rawnoise_image_nlmatchfilter_ms
```

**3) Test example**

```bash
podman run --rm --device nvidia.com/gpu=all \
  -v /host/path/to/data/png:/data/png \
  -v /host/path/to/data/raw:/data/raw \
  -v $PWD/logs:/nonlocal-matchfilter/logs \
  -v $PWD/results:/aim \
  -v $PWD/weights:/weights
  nonlocalmf \
  test +experiment=gauss_image_nlmatchfilter_ms model.network.neighbours.scale1=[5,5] model.network.neighbours.scale2=[3,5] model.network.neighbours.scale3=[3,3] ckpt_path=/weights/raw_25_15_9.ckpt
```

If you use Docker, replace runtime flags with your Docker GPU setup (typically `--gpus all`).

## Experiments

The project uses [Hydra](https://hydra.cc/docs/intro/) for configuration management.
We provide four experiment presets, which are located in `conf/experiment/`:

- gauss_image_nlmatchfilter.yaml
- gauss_image_nlmatchfilter_ms.yaml
- rawnoise_image_nlmatchfilter.yaml
- rawnoise_image_nlmatchfilter_ms.yaml

Run them with:
```bash
train +experiment=<experiment_name>
test +experiment=<experiment_name> ckpt_path=<path_to_ckpt>
```

You can override any value in the base configuration from the command line, e.g.

```bash
train +experiment=gauss_image_nlmatchfilter model/network=nlmatchfilter_cherel trainer.max_epochs=2000 data.loader.train.batch_size=4
```

## Datasets

All the datasets used for training and testing in the paper are public.
In `src/nonlocal_matchfilter/data/image_datasets.py` we provide PyTorch `Dataset` classes to load them.
We do not redistribute them; please, download them from their original sources and organize the files according to the expected structure.

All the images in the RAW datasets are expected to be `tiff` files, packed into 4 channels in RGBG format.
They also need metadata `npy` files containing their noise profiles and ISP information to process them for visualization.
We provide such files in the repository's Releases, alongside some pretrained network weights.

## Citation

```
@misc{Sanchezbeeckman2026NonlocalMatchFilter,
    title={Learned Nonlocal Feature Matching and Filtering for RAW Image Denoising}, 
    author={Marco Sánchez-Beeckman and Antoni Buades},
    year={2026},
    eprint={2604.17453},
    archivePrefix={arXiv},
    primaryClass={eess.IV},
    url={https://arxiv.org/abs/2604.17453}, 
}
```
