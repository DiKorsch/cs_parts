# L1-SVM based parts extraction

Code for the paper "[Classification-Specific Parts for Improving Fine-Grained Visual Categorization](https://arxiv.org/abs/1909.07075)"

## Installation
Clone the repository and initialize the submodules
```bash
git clone git@github.com:cvjena/cs_parts.git
cd cs_parts
git submodule init
git submodule update
```


### python3.7
```bash
conda env create -n cs_parts python~=3.7.0
conda activate cs_parts
conda install -c conda-forge -c nvidia cudatoolkit=11.0.3 \
 cudatoolkit-dev=11.0.3 nccl cudnn

pip install --no-cache-dir cupy-cuda110~=7.8.0
python -c "import cupy; cupy.show_config(); print(cupy.zeros(8) + 1)"
# should display something like this:
# CuPy Version          : 7.8.0
# CUDA Root             : /home/korsch/.miniconda3/envs/cs_parts
# CUDA Build Version    : 11000
# CUDA Driver Version   : 12000
# CUDA Runtime Version  : 11000
# cuBLAS Version        : 11200
# cuFFT Version         : 10201
# cuRAND Version        : 10201
# cuSOLVER Version      : (10, 6, 0)
# cuSPARSE Version      : 11101
# NVRTC Version         : (11, 0)
# cuDNN Build Version   : 8002
# cuDNN Version         : 8201
# NCCL Build Version    : 2708
# NCCL Runtime Version  : 2708
# CUB Version           : Enabled
# cuTENSOR Version      : None
# [1. 1. 1. 1. 1. 1. 1. 1.]
pip install -r requirements.txt
```

### python3.9
```bash
conda env create -n cs_parts python~=3.9.0
conda activate cs_parts
conda install -c conda-forge -c nvidia cudatoolkit=11.0.3 \
 cudatoolkit-dev=11.0.3 nccl cudnn

# installs cupy version 7.8.0.post1 directly from source, since the
# wheels are only built for python3.7
pip install --no-cache-dir -e git+https://github.com/cupy/cupy.git@3e3635d802eda54a4b8c96d0126c646e97c3d239#egg=cupy
python -c "import cupy; cupy.show_config(); print(cupy.zeros(8) + 1)"
# should display something like this:
# CuPy Version          : 7.8.0
# CUDA Root             : /home/korsch/.miniconda3/envs/cs_parts
# CUDA Build Version    : 11000
# CUDA Driver Version   : 12000
# CUDA Runtime Version  : 11000
# cuBLAS Version        : 11200
# cuFFT Version         : 10201
# cuRAND Version        : 10201
# cuSOLVER Version      : (10, 6, 0)
# cuSPARSE Version      : 11101
# NVRTC Version         : (11, 0)
# cuDNN Build Version   : 8002
# cuDNN Version         : 8201
# NCCL Build Version    : 2708
# NCCL Runtime Version  : 2708
# CUB Version           : Enabled
# cuTENSOR Version      : None
# [1. 1. 1. 1. 1. 1. 1. 1.]
pip install -r requirements.txt
```

## Running the experiments

### Download the datasets and models

1. Download the needed datasets. Set up the according paths in the `data_info.yml` file.
2. Download the [fine-tuned models](models) or copy your own models to the `models` folder

### Start an experiment

You could either start the whole pipeline for the default dataset (`CUB200`):

```bash
./run.sh
```

or set according datasets (and GPUs) manually:

```bash
GPU=0 DATASET=NAB ./run.sh
GPU=1 DATASETS=FLOWERS ./run.sh
GPU=1 BATCH_SIZE=16 DATASETS=CARS ./run.sh
```

The extracted features (`features.npz`), the trained L1-SVM, and part locations (`part_locs.txt`) will be stored in `outputs/<DATASET>/<MODEL_TYPE>/<timestamp>` folder.

You can also restart the experiment using already extracted features and/or trained L1-SVM by setting the `--checkpoint` parameter, e.g.:

```bash
./run.sh --checkpoint outputs/CUB200/cvmodelz.InceptionV3/2023-10-05-11.27.16.471332713
```

## Citation
You are welcome to use our code in your research! If you do so please cite it as:

```bibtex
@inproceedings{Korsch19_CSPARTS,
  title = {Classification-Specific Parts for Improving Fine-Grained Visual Categorization},
  booktitle = {German Conference on Pattern Recognition (GCPR)},
  author = {Dimitri Korsch and Paul Bodesheim and Joachim Denzler},
  pages = {62--75},
  year = {2019},
}
```

## License
This work is licensed under a [GNU Affero General Public License][agplv3].

[![AGPLv3][agplv3-image]][agplv3]

[agplv3]: https://www.gnu.org/licenses/agpl-3.0.html
[agplv3-image]: https://www.gnu.org/graphics/agplv3-88x31.png
